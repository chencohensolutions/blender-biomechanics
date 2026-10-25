"""Train muscle activations that hold the bone chain up against gravity (no bpy).

Loads a chain package exported from Blender, runs the chain solver headless, and
optimizes either a static activation vector or a small state-feedback network
with a simple evolution strategy. The objective rewards the height of the top
bone and an upright posture, and penalizes effort, strain, and penetration.

Run from the workspace .venv:

    python tools/train_posture.py --package <dir> --mode static --generations 60
    python tools/train_posture.py --package <dir> --mode policy --generations 200

Outputs `posture_policy.npz` in the package directory (weights plus metadata)
and a JSON log of the best score per generation.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import functools

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001  progress must reach a redirected log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _addon_import import load_addon_module  # noqa: E402

chain_package = load_addon_module("chain_package")
chain_solver = load_addon_module("chain_solver")
posture_policy = load_addon_module("posture_policy")
Policy = posture_policy.Policy
static_controller = posture_policy.static_controller

POLICY_FILE = posture_policy.POLICY_FILE
LOG_FILE = "posture_training_log.json"


# ----------------------------------------------------------------------------- environment


class PostureEnv:
    """Headless chain under gravity; actions are muscle group activations in [0, 1]."""

    FAIL_PENALTY = -10.0

    def __init__(self, package_dir, steps=40, gravity_mm=None, iterations=None, substeps=None, effort=0.02, perturb_mm=0.0, perturb_deg=0.0, seed=0, fail_cm=5.0):
        self.meta, self.arrays = chain_package.load_package(package_dir)
        self.field_cache = {}
        self.steps = steps
        self.iterations = iterations
        self.substeps = substeps
        self.effort = effort
        self.perturb_mm = perturb_mm
        self.perturb_deg = perturb_deg
        self.fail_cm = fail_cm
        self.rng = np.random.default_rng(seed)
        self.units_per_mm = self.meta["units_per_mm"]
        gravity = None
        if gravity_mm is not None:
            gravity = [0.0, 0.0, -gravity_mm * self.units_per_mm]
        self.gravity = gravity
        self.solver = None
        self.top = self.meta["bodies"].index(self.meta["top"])
        self.state = None
        self.reset()

    @property
    def group_count(self):
        return len(self.meta["muscle_groups"])

    def reset(self):
        self.solver = chain_package.build_solver(
            self.meta, self.arrays, self.field_cache, iterations=self.iterations, substeps=self.substeps, gravity=self.gravity
        )
        self.state = posture_policy.ChainState(self.solver, self.units_per_mm)
        self.rest_top = self.solver.bodies[self.top].position.copy()
        self.perturb()
        return self.observe()

    def perturb(self):
        """Drag one random free body away from rest, as a user would, then release it."""
        if self.perturb_mm <= 0.0 and self.perturb_deg <= 0.0:
            return
        free = [body for body in self.solver.bodies if not body.fixed]
        body = free[self.rng.integers(len(free))]
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        target = body.position + direction * self.perturb_mm * self.units_per_mm * abs(self.rng.normal())
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(self.perturb_deg) * self.rng.normal()
        half = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
        target_quaternion = chain_solver.quat_multiply(half, body.quaternion)
        pose = chain_solver.PoseConstraint(body, target, target_quaternion, position_compliance=0.02, rotation_compliance=0.02)
        # The pose step limit (1 mm per sweep) and the contacts keep the drag physical.
        self.solver.pose_constraints = [pose]
        for _drag in range(6):
            self.solver.solve()
        self.solver.pose_constraints = []

    def observe(self):
        return self.state.observe()

    def step(self, activations):
        self.solver.activations[:] = np.clip(activations, 0.0, 1.0)
        self.solver.solve()
        return self.observe()

    def score(self, activations_used):
        """Height of the top bone relative to rest (cm), upright bonus, minus penalties."""
        top = self.solver.bodies[self.top].position
        height_cm = (top[2] - self.rest_top[2]) / self.units_per_mm / 10.0
        if not np.all(np.isfinite([body.position for body in self.solver.bodies])) or abs(height_cm) > 100.0:
            return -100.0  # the solver diverged: worst possible outcome
        drift_cm = np.linalg.norm((top[:2] - self.rest_top[:2])) / self.units_per_mm / 10.0
        strain = min(self.solver.max_strain(), 5.0)
        penetration_mm = self.solver.max_penetration() / self.units_per_mm
        effort = float(np.mean(np.square(activations_used)))
        return height_cm - 0.3 * drift_cm - 0.2 * strain - 0.5 * max(penetration_mm - 1.0, 0.0) - self.effort * effort * 10.0

    def height_cm(self):
        return (self.solver.bodies[self.top].position[2] - self.rest_top[2]) / self.units_per_mm / 10.0

    def failed(self):
        """The top bone has dropped by fail_cm or more (or the solver diverged)."""
        return (not self.solver.is_finite()) or self.height_cm() <= -self.fail_cm

    def rollout(self, controller, episodes=1):
        """Mean score over episodes; with perturbations each episode starts differently.

        An episode that drops the top bone by `fail_cm` ends at once as a failure: its
        score is FAIL_PENALTY plus a small credit for the steps survived, always below
        any non-failing score.
        """
        total = 0.0
        for _episode in range(episodes):
            observation = self.reset()
            used = []
            failed_at = None
            for step in range(self.steps):
                activations = controller(observation)
                used.append(activations)
                observation = self.step(activations)
                if self.failed():
                    failed_at = step + 1
                    break
            if failed_at is not None:
                total += self.FAIL_PENALTY + 2.0 * failed_at / self.steps
            else:
                total += self.score(np.mean(used, axis=0))
        return total / episodes


# ----------------------------------------------------------------------------- evolution strategy

_worker_env = None
_worker_mode = None
_worker_policy = None


_worker_episodes = 1


def _worker_init(package_dir, steps, gravity_mm, iterations, substeps, mode, hidden, effort, perturb_mm, perturb_deg, episodes, seed, fail_cm):
    global _worker_env, _worker_mode, _worker_policy, _worker_episodes
    _worker_env = PostureEnv(package_dir, steps, gravity_mm, iterations, substeps, effort, perturb_mm, perturb_deg, seed + os.getpid(), fail_cm)
    _worker_mode = mode
    _worker_policy = Policy(len(_worker_env.observe()), _worker_env.group_count, hidden) if mode == "policy" else None
    _worker_episodes = episodes


def _evaluate(vector):
    controller = static_controller(vector) if _worker_mode == "static" else _worker_policy.controller(vector)
    return _worker_env.rollout(controller, _worker_episodes)


def evolve(args):
    env = PostureEnv(args.package, args.steps, args.gravity, args.iterations, args.substeps, args.effort, args.perturb_mm, args.perturb_deg, args.seed, args.fail_cm)
    observation_size = len(env.observe())
    if args.mode == "static":
        dimension = env.group_count
        mean = np.full(dimension, -2.0)  # sigmoid(-2) = 0.12: start with light tone
    else:
        policy = Policy(observation_size, env.group_count, args.hidden)
        dimension = policy.size
        mean = np.zeros(dimension)
        # Bias toward light tone at start.
        mean[-env.group_count :] = -2.0
    sigma = args.sigma
    population = args.population
    rng = np.random.default_rng(args.seed)
    print(f"mode={args.mode} dimension={dimension} groups={env.group_count} observation={observation_size} population={population}")
    baseline = env.rollout(static_controller(np.full(env.group_count, -20.0)), args.episodes)
    print(f"passive score (no muscles): {baseline:.3f}")

    pool = mp.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(
            args.package, args.steps, args.gravity, args.iterations, args.substeps, args.mode, args.hidden, args.effort,
            args.perturb_mm, args.perturb_deg, args.episodes, args.seed, args.fail_cm,
        ),
    )
    log = []
    best_vector, best_score = mean.copy(), -np.inf
    started = time.perf_counter()
    try:
        for generation in range(args.generations):
            noise = rng.standard_normal((population // 2, dimension))
            noise = np.concatenate((noise, -noise))  # antithetic
            candidates = mean + sigma * noise
            scores = np.array(pool.map(_evaluate, list(candidates)))
            order = np.argsort(scores)[::-1]
            elite = order[: max(2, population // 4)]
            # Weighted recombination of the elite.
            weights = np.log(len(elite) + 0.5) - np.log(np.arange(1, len(elite) + 1))
            weights /= weights.sum()
            mean = (weights[:, None] * candidates[elite]).sum(axis=0)
            if scores[order[0]] > best_score:
                best_score, best_vector = float(scores[order[0]]), candidates[order[0]].copy()
            sigma *= 0.98
            log.append({"generation": generation, "best": float(scores[order[0]]), "mean": float(scores.mean()), "sigma": sigma})
            elapsed = time.perf_counter() - started
            print(f"gen {generation:3d}  best {scores[order[0]]:7.3f}  mean {scores.mean():7.3f}  overall {best_score:7.3f}  sigma {sigma:.3f}  {elapsed:6.0f}s")
            # Save every generation so Blender's Follow Training view can replay the current best.
            _save(args, env, best_vector, best_score, observation_size, log, generation)
    finally:
        pool.close()
        pool.join()
    _save(args, env, best_vector, best_score, observation_size, log, args.generations)
    return best_vector, best_score


def _save(args, env, vector, score, observation_size, log, generation):
    gravity_mm = -env.solver.gravity[2] / env.units_per_mm
    target = os.path.join(args.package, POLICY_FILE)
    temporary = target + ".tmp.npz"
    posture_policy.save_policy(
        temporary, args.mode, vector, score, observation_size, env.group_count, args.hidden, args.steps, gravity_mm, generation
    )
    os.replace(temporary, target)  # atomic: a reader never sees a half-written file
    with open(os.path.join(args.package, LOG_FILE), "w", encoding="utf-8") as handle:
        json.dump(log, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, help="chain package directory exported from Blender")
    parser.add_argument("--mode", choices=("static", "policy"), default="static")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40, help="solver steps per rollout")
    parser.add_argument("--gravity", type=float, default=None, help="override gravity in mm per step")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--effort", type=float, default=0.02)
    parser.add_argument("--perturb-mm", dest="perturb_mm", type=float, default=0.0, help="random push of 3 bodies at episode start (mm)")
    parser.add_argument("--perturb-deg", dest="perturb_deg", type=float, default=0.0, help="random tilt of those bodies (degrees)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes averaged per evaluation (use >1 with perturbations)")
    parser.add_argument("--fail-cm", dest="fail_cm", type=float, default=5.0, help="top-bone drop that ends an episode as a failure")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _vector, score = evolve(args)
    print(f"best score {score:.3f}; saved {os.path.join(args.package, POLICY_FILE)}")


if __name__ == "__main__":
    main()
