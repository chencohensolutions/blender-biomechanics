"""Posture control through a learned dynamics surrogate (no bpy).

The chain solver is exact but slow (hundreds of ms per step). This pipeline
learns it once and then trains the posture controller against the learned
model with gradients, thousands of episodes per second:

    collect   roll the real solver under random activations and pushes, saving
              (observation, activations) -> next observation transitions
    fit       train an MLP dynamics model on those transitions (PyTorch)
    optimize  train the posture policy by backpropagation through unrolled
              surrogate rollouts (height up, effort down, fail below -5 cm)
    evaluate  score a policy in the real solver over perturbed episodes
    iterate   rounds of collect (half the episodes under the current policy),
              fit, optimize; the policy file is only replaced when its real
              solver score improves, so Blender always sees the best so far

The policy has the same shape and file format as tools/train_posture.py, so
Blender's Posture Controller loads it unchanged.

    python tools/train_posture_surrogate.py collect  --package <dir> --episodes 300
    python tools/train_posture_surrogate.py fit      --package <dir>
    python tools/train_posture_surrogate.py optimize --package <dir>
    python tools/train_posture_surrogate.py evaluate --package <dir>
"""

import argparse
import functools
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from _addon_import import load_addon_module  # noqa: E402

posture_policy = load_addon_module("posture_policy")

DATASET_FILE = "dynamics_dataset.npz"
MODEL_FILE = "dynamics_model.npz"
LATEST_FILE = "posture_policy_latest.npz"
GRAVITY_MM = 0.3
PERTURB_MM = 8.0
PERTURB_DEG = 4.0
STEPS = 40
FAIL_CM = 5.0


LIVE_FILE = "surrogate_live.json"


class Live:
    """Training state snapshot for tools/surrogate_dashboard.py, rewritten atomically.

    One JSON document per package: phase, fit curve, optimize curve, the current
    policy weights with sample activations (so the dashboard can draw the network
    with its values), and the real-solver scores per round.
    """

    def __init__(self, package):
        import json

        self.json = json
        self.path = os.path.join(package, LIVE_FILE)
        self.data = {"phase": "", "round": 0, "fit": [], "optimize": [], "real_scores": [], "meta": {}, "dataset": 0}
        if os.path.exists(self.path):
            try:
                with open(self.path) as handle:
                    self.data.update(json.load(handle))
            except (OSError, ValueError):
                pass

    def set_meta(self, env):
        self.data["meta"] = {
            "bodies": [body.name for body in env.state.free],
            "groups": [" ".join(group) for group in env.meta["muscle_groups"]],
            "top": env.meta["top"],
        }

    def update(self, **fields):
        self.data.update(fields)
        self.data["time"] = time.time()
        temporary = self.path + ".tmp"
        with open(temporary, "w") as handle:
            self.json.dump(self.data, handle)
        os.replace(temporary, self.path)


def _train_posture_module():
    spec = importlib.util.spec_from_file_location("train_posture", TOOLS / "train_posture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(package, seed, gravity_mm=GRAVITY_MM):
    tp = _train_posture_module()
    return tp.PostureEnv(package, STEPS, gravity_mm, None, None, 0.02, PERTURB_MM, PERTURB_DEG, seed, FAIL_CM)


# ----------------------------------------------------------------------------- collect


def _ou_activations(rng, group_count, steps, mean=0.15, sigma=0.15, theta=0.3):
    """Smooth random activation sequences (Ornstein-Uhlenbeck), clipped to [0, 1]."""
    values = np.clip(rng.normal(mean, sigma, group_count), 0.0, 1.0)
    out = []
    for _step in range(steps):
        values = values + theta * (mean - values) + sigma * 0.5 * rng.normal(size=group_count)
        values = np.clip(values, 0.0, 1.0)
        out.append(values.copy())
    return np.array(out)


def _collect_worker(args):
    package, seed, episodes, policy = args
    env = make_env(package, seed)
    rng = np.random.default_rng(seed + 1000)
    controller = None
    if policy is not None:
        controller = posture_policy.controller_from_policy(policy, len(env.observe()), env.group_count)
    observations, activations, next_observations, heights = [], [], [], []
    for episode in range(episodes):
        observation = env.reset()
        # Odd episodes follow the current policy plus exploration noise, so the data
        # covers the states the policy actually visits; even episodes stay random.
        follow = controller is not None and episode % 2 == 1
        sigma = rng.uniform(0.02, 0.1) if follow else rng.uniform(0.05, 0.25)
        plan = _ou_activations(rng, env.group_count, STEPS, mean=0.0 if follow else rng.uniform(0.0, 0.4), sigma=sigma)
        for step in range(STEPS):
            activation = plan[step]
            if follow:
                activation = np.clip(controller(observation) + plan[step] - plan[step].mean(), 0.0, 1.0)
            following = env.step(activation)
            observations.append(observation)
            activations.append(activation)
            next_observations.append(following)
            heights.append(env.height_cm())
            observation = following
            if not env.solver.is_finite() or env.height_cm() < -3.0 * FAIL_CM:
                break
    return np.array(observations), np.array(activations), np.array(next_observations), np.array(heights)


def collect(args, policy=None):
    workers = args.workers
    per_worker = max(1, args.episodes // workers)
    started = time.perf_counter()
    live = Live(args.package)
    live.update(phase="collect", episodes=per_worker * workers, on_policy=policy is not None)
    with mp.Pool(workers) as pool:
        parts = pool.map(_collect_worker, [(args.package, args.seed + index, per_worker, policy) for index in range(workers)])
    observations = np.concatenate([part[0] for part in parts])
    activations = np.concatenate([part[1] for part in parts])
    next_observations = np.concatenate([part[2] for part in parts])
    heights = np.concatenate([part[3] for part in parts])
    path = os.path.join(args.package, DATASET_FILE)
    if args.append and os.path.exists(path):
        with np.load(path) as old:
            observations = np.concatenate([old["observations"], observations])
            activations = np.concatenate([old["activations"], activations])
            next_observations = np.concatenate([old["next_observations"], next_observations])
            heights = np.concatenate([old["heights"], heights])
    np.savez_compressed(path, observations=observations, activations=activations, next_observations=next_observations, heights=heights)
    live.update(dataset=len(observations))
    print(f"collected {per_worker * workers} episodes in {time.perf_counter() - started:.0f} s; dataset now {len(observations)} transitions -> {path}")
    print(f"height range {heights.min():.1f} .. {heights.max():.1f} cm; mean |delta obs| {np.abs(next_observations - observations).mean():.4f}")


# ----------------------------------------------------------------------------- fit


def _torch():
    import torch

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    return torch


class Dynamics:
    """MLP: normalized [observation, activations] -> normalized delta observation."""

    def __init__(self, torch, observation_size, group_count, hidden=256, layers=3):
        self.torch = torch
        sizes = [observation_size + group_count] + [hidden] * layers + [observation_size]
        modules = []
        for index in range(len(sizes) - 1):
            modules.append(torch.nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                modules.append(torch.nn.SiLU())
        self.net = torch.nn.Sequential(*modules)
        self.stats = None

    def set_stats(self, observations, activations, deltas):
        t = self.torch
        eps = 1e-6
        self.stats = {
            "obs_mean": t.tensor(observations.mean(0), dtype=t.float32),
            "obs_std": t.tensor(observations.std(0) + eps, dtype=t.float32),
            "act_mean": t.tensor(activations.mean(0), dtype=t.float32),
            "act_std": t.tensor(activations.std(0) + eps, dtype=t.float32),
            "delta_mean": t.tensor(deltas.mean(0), dtype=t.float32),
            "delta_std": t.tensor(deltas.std(0) + eps, dtype=t.float32),
        }

    def step(self, observation, activation):
        s = self.stats
        x = self.torch.cat(((observation - s["obs_mean"]) / s["obs_std"], (activation - s["act_mean"]) / s["act_std"]), dim=-1)
        delta = self.net(x) * s["delta_std"] + s["delta_mean"]
        return observation + delta

    def save(self, path):
        arrays = {f"stat_{key}": value.numpy() for key, value in self.stats.items()}
        for name, parameter in self.net.state_dict().items():
            arrays[f"param_{name}"] = parameter.numpy()
        np.savez(path, **arrays)

    @classmethod
    def load(cls, torch, path, observation_size, group_count, hidden=256, layers=3):
        model = cls(torch, observation_size, group_count, hidden, layers)
        with np.load(path) as data:
            model.stats = {key[5:]: torch.tensor(data[key]) for key in data.files if key.startswith("stat_")}
            state = {key[6:]: torch.tensor(data[key]) for key in data.files if key.startswith("param_")}
        model.net.load_state_dict(state)
        return model


def fit(args):
    torch = _torch()
    with np.load(os.path.join(args.package, DATASET_FILE)) as data:
        observations = data["observations"].astype(np.float32)
        activations = data["activations"].astype(np.float32)
        next_observations = data["next_observations"].astype(np.float32)
    deltas = next_observations - observations
    count = len(observations)
    split = int(count * 0.9)
    rng = np.random.default_rng(0)
    order = rng.permutation(count)
    train_idx, test_idx = order[:split], order[split:]
    model_path = os.path.join(args.package, MODEL_FILE)
    if args.warm and os.path.exists(model_path):
        model = Dynamics.load(torch, model_path, observations.shape[1], activations.shape[1], args.hidden_dyn, args.layers)
    else:
        model = Dynamics(torch, observations.shape[1], activations.shape[1], args.hidden_dyn, args.layers)
        model.set_stats(observations[train_idx], activations[train_idx], deltas[train_idx])
    obs_t, act_t, delta_t = (torch.tensor(a) for a in (observations, activations, deltas))
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    batch = 512
    started = time.perf_counter()
    live = Live(args.package)
    live.update(phase="fit", fit=[], dataset=count)
    for epoch in range(args.epochs):
        permutation = torch.tensor(rng.permutation(train_idx))
        total = 0.0
        for start in range(0, len(permutation), batch):
            idx = permutation[start : start + batch]
            predicted = model.step(obs_t[idx], act_t[idx]) - obs_t[idx]
            loss = torch.mean(((predicted - delta_t[idx]) / model.stats["delta_std"]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        with torch.no_grad():
            idx = torch.tensor(test_idx)
            predicted = model.step(obs_t[idx], act_t[idx]) - obs_t[idx]
            test = float(torch.mean(((predicted - delta_t[idx]) / model.stats["delta_std"]) ** 2))
            raw_mm = float(torch.mean(torch.abs(predicted - delta_t[idx])[:, : len(predicted[0]) // 2])) * 10.0
        live.data["fit"].append({"epoch": epoch, "train": total / split, "test": test, "mm": raw_mm})
        live.update()
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train {total / split:.4f}  test {test:.4f}  (normalized MSE; mean position error {raw_mm:.3f} mm/step)  {time.perf_counter() - started:.0f}s")
    model.save(model_path)
    print(f"saved {model_path}")


# ----------------------------------------------------------------------------- optimize


class TorchPolicy:
    def __init__(self, torch, observation_size, group_count, hidden=32):
        self.torch = torch
        self.w1 = torch.nn.Parameter(torch.randn(observation_size, hidden) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.randn(hidden, group_count) * 0.1)
        self.b2 = torch.nn.Parameter(torch.full((group_count,), -2.0))
        self.hidden = hidden

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def __call__(self, observation):
        h = self.torch.tanh(observation @ self.w1 + self.b1)
        return self.torch.sigmoid(h @ self.w2 + self.b2)

    def vector(self):
        return np.concatenate([p.detach().numpy().ravel() for p in self.parameters()])


def optimize(args):
    torch = _torch()
    with np.load(os.path.join(args.package, DATASET_FILE)) as data:
        observations = data["observations"].astype(np.float32)
        activations = data["activations"].astype(np.float32)
        heights = data["heights"].astype(np.float32)
    observation_size, group_count = observations.shape[1], activations.shape[1]
    model = Dynamics.load(torch, os.path.join(args.package, MODEL_FILE), observation_size, group_count, args.hidden_dyn, args.layers)
    for parameter in model.net.parameters():
        parameter.requires_grad_(False)
    env = make_env(args.package, 0)
    # Height of the top bone is its z offset in the observation (cm); find its index.
    top_index = [body.name for body in env.state.free].index(env.meta["top"])
    height_column = top_index * 3 + 2
    # Starting states: observed states that have not yet failed, plus the rest pose.
    pool = observations[heights > -2.0]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    policy = TorchPolicy(torch, observation_size, group_count, args.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    started = time.perf_counter()
    live = Live(args.package)
    live.set_meta(env)
    live.update(phase="optimize", optimize=[])
    # A fixed pushed state for the dashboard: the lowest non-failed state in the pool.
    sample_state = pool[np.argmin(pool[:, height_column])]

    def snapshot(iteration):
        with torch.no_grad():
            samples = {}
            for label, state in (("neutral", np.zeros(observation_size, np.float32)), ("pushed", sample_state)):
                x = torch.tensor(state)
                hidden = torch.tanh(x @ policy.w1 + policy.b1)
                output = torch.sigmoid(hidden @ policy.w2 + policy.b2)
                samples[label] = {"input": state.tolist(), "hidden": hidden.tolist(), "output": output.tolist()}
        rounded = lambda tensor: [[round(v, 4) for v in row] for row in tensor.detach().tolist()]  # noqa: E731
        live.update(
            iteration=iteration,
            weights={"w1": rounded(policy.w1), "b1": [round(v, 4) for v in policy.b1.tolist()], "w2": rounded(policy.w2), "b2": [round(v, 4) for v in policy.b2.tolist()]},
            samples=samples,
        )

    for iteration in range(args.iterations):
        starts = pool[rng.choice(len(pool), args.batch)]
        starts[: args.batch // 4] = 0.0  # a quarter of the batch starts at rest
        observation = torch.tensor(starts)
        total_height = 0.0
        total_effort = 0.0
        fail = 0.0
        for _step in range(args.horizon):
            activation = policy(observation)
            observation = model.step(observation, activation)
            height = observation[:, height_column]
            # Reward height only up to a small gain: the chain cannot really rise much
            # above rest, and an uncapped reward drives the policy into states the
            # surrogate has never seen, where it extrapolates freely.
            total_height = total_height + torch.clamp(height, max=args.height_cap).mean()
            total_effort = total_effort + (activation ** 2).mean()
            fail = fail + torch.sigmoid((-FAIL_CM - height) * 2.0).mean()
        objective = total_height / args.horizon - args.effort * total_effort / args.horizon - 10.0 * fail / args.horizon
        loss = -objective
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        live.data["optimize"].append(
            {"iter": iteration, "objective": objective.item(), "height": total_height.item() / args.horizon, "effort": total_effort.item() / args.horizon, "fail": fail.item() / args.horizon}
        )
        if iteration % args.live_every == 0 or iteration == args.iterations - 1:
            snapshot(iteration)
        if iteration % 50 == 0 or iteration == args.iterations - 1:
            print(
                f"iter {iteration:4d}  objective {objective.item():7.3f}  mean height {total_height.item() / args.horizon:6.2f} cm  "
                f"effort {total_effort.item() / args.horizon:.3f}  fail {fail.item() / args.horizon:.3f}  {time.perf_counter() - started:.0f}s"
            )
    vector = policy.vector()
    live.update(phase="evaluate")
    score = _evaluate_vector(args.package, vector, args.hidden, episodes=args.eval_episodes)
    live.data["real_scores"].append({"round": live.data.get("round", 0), "score": score})
    live.update(phase="evaluated")
    print(f"real-solver score over {args.eval_episodes} perturbed episodes: {score:.3f}")
    return vector, score


def _save_if_better(args, vector, score, observation_size, group_count, generation):
    """Write posture_policy.npz (what Blender loads) only when the real score improved."""
    path = os.path.join(args.package, posture_policy.POLICY_FILE)
    previous = None
    if os.path.exists(path):
        existing = posture_policy.load_policy(path)
        if existing["mode"] == "policy" and existing["observation_size"] == observation_size and existing["hidden"] == args.hidden:
            previous = existing["score"]
    if previous is not None and score <= previous and not args.overwrite:
        print(f"kept existing policy (score {previous:.3f} >= {score:.3f})")
        return False
    posture_policy.save_policy(path, "policy", vector, score, observation_size, group_count, args.hidden, STEPS, GRAVITY_MM, generation)
    print(f"saved {path} (score {score:.3f})")
    return True


def iterate(args):
    """Collect under the current policy, refit, re-optimize; repeat and keep the best real score."""
    # Data is collected under the most recent surrogate policy even when it scored
    # worse than the best one: the surrogate must learn the states that policy visits
    # so that the next optimization cannot exploit them again.
    latest_path = os.path.join(args.package, LATEST_FILE)
    if not os.path.exists(latest_path):
        best_path = os.path.join(args.package, posture_policy.POLICY_FILE)
        if os.path.exists(best_path):
            latest_path = best_path
    for round_index in range(args.rounds):
        print(f"=== round {round_index} ===")
        Live(args.package).update(round=round_index)
        policy = posture_policy.load_policy(latest_path) if os.path.exists(latest_path) else None
        if policy is not None and (policy["mode"] != "policy" or policy["hidden"] != args.hidden):
            policy = None
        args.seed += 100
        args.append = True
        collect(args, policy)
        args.warm = True
        fit(args)
        vector, score = optimize(args)
        env = make_env(args.package, 0)
        observation_size, group_count = len(env.observe()), env.group_count
        latest_path = os.path.join(args.package, LATEST_FILE)
        posture_policy.save_policy(latest_path, "policy", vector, score, observation_size, group_count, args.hidden, STEPS, GRAVITY_MM, round_index + 1)
        _save_if_better(args, vector, score, observation_size, group_count, round_index + 1)


def _evaluate_vector(package, vector, hidden, episodes=8, seed=123):
    env = make_env(package, seed)
    controller = posture_policy.Policy(len(env.observe()), env.group_count, hidden).controller(vector)
    return env.rollout(controller, episodes)


def evaluate(args):
    policy = posture_policy.load_policy(os.path.join(args.package, posture_policy.POLICY_FILE))
    env = make_env(args.package, args.seed)
    controller = posture_policy.controller_from_policy(policy, len(env.observe()), env.group_count)
    passive = env.rollout(posture_policy.static_controller(np.full(env.group_count, -20.0)), args.eval_episodes)
    score = env.rollout(controller, args.eval_episodes)
    print(f"passive {passive:.3f}   policy {score:.3f}   (episodes {args.eval_episodes}, fail at -{FAIL_CM:g} cm)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "fit", "optimize", "evaluate", "iterate"))
    parser.add_argument("--package", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dyn", dest="hidden_dyn", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=STEPS)
    parser.add_argument("--effort", type=float, default=0.2)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=8)
    parser.add_argument("--height-cap", dest="height_cap", type=float, default=1.0, help="cm of height gain rewarded at most")
    parser.add_argument("--append", action="store_true", help="collect: add to the existing dataset")
    parser.add_argument("--warm", action="store_true", help="fit: continue from the saved model")
    parser.add_argument("--overwrite", action="store_true", help="optimize: save even if the real score did not improve")
    parser.add_argument("--rounds", type=int, default=5, help="iterate: collect/fit/optimize rounds")
    parser.add_argument("--live-every", dest="live_every", type=int, default=10, help="optimize: iterations between dashboard snapshots")
    args = parser.parse_args()
    if args.command == "optimize":
        vector, score = optimize(args)
        env = make_env(args.package, 0)
        _save_if_better(args, vector, score, len(env.observe()), env.group_count, 0)
    else:
        {"collect": collect, "fit": fit, "evaluate": evaluate, "iterate": iterate}[args.command](args)


if __name__ == "__main__":
    main()
