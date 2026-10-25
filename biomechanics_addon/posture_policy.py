"""Posture controller shared by headless training and Blender (bpy-free).

A controller maps the chain state to muscle group activations in [0, 1]. Two
forms exist: a static activation vector (one posture) and a two-layer tanh
network on the observation. Both are stored in `posture_policy.npz`.

The observation is, for every free body in solver order, its position offset
from the rest pose in centimeters followed by its rotation vector from the rest
orientation in radians, so a trained network reacts to the chain being pushed
out of its rest pose.
"""

import numpy as np

from .chain_solver import quat_conjugate, quat_multiply

POLICY_FILE = "posture_policy.npz"


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


class ChainState:
    """Rest pose bookkeeping for observations of one solver."""

    def __init__(self, solver, units_per_mm):
        self.free = [body for body in solver.bodies if not body.fixed]
        self.rest_positions = np.array([body.position for body in self.free])
        self.rest_quaternions = np.array([body.quaternion for body in self.free])
        self.units_per_mm = units_per_mm

    @property
    def size(self):
        return 6 * len(self.free)

    def observe(self):
        positions = np.array([body.position for body in self.free])
        offsets = (positions - self.rest_positions) / self.units_per_mm / 10.0  # cm
        rotations = []
        for body, rest in zip(self.free, self.rest_quaternions):
            relative = quat_multiply(body.quaternion, quat_conjugate(rest))
            if relative[0] < 0.0:
                relative = -relative
            rotations.append(2.0 * relative[1:])
        return np.concatenate((offsets.ravel(), np.array(rotations).ravel()))


class Policy:
    """Two-layer tanh network: observation -> activations in [0, 1]."""

    def __init__(self, observation_size, group_count, hidden=32):
        self.shapes = [(observation_size, hidden), (hidden,), (hidden, group_count), (group_count,)]
        self.size = sum(int(np.prod(shape)) for shape in self.shapes)

    def unpack(self, vector):
        parts = []
        offset = 0
        for shape in self.shapes:
            count = int(np.prod(shape))
            parts.append(np.asarray(vector[offset : offset + count]).reshape(shape))
            offset += count
        return parts

    def controller(self, vector):
        w1, b1, w2, b2 = self.unpack(vector)

        def controller(observation):
            hidden = np.tanh(observation @ w1 + b1)
            return sigmoid(hidden @ w2 + b2)

        return controller


def static_controller(vector):
    activations = sigmoid(np.asarray(vector, dtype=np.float64))

    def controller(_observation):
        return activations

    return controller


def save_policy(path, mode, vector, score, observation_size, group_count, hidden, steps, gravity_mm, generation=0):
    np.savez(
        path,
        mode=mode,
        vector=np.asarray(vector, dtype=np.float64),
        score=float(score),
        hidden=int(hidden),
        observation_size=int(observation_size),
        group_count=int(group_count),
        steps=int(steps),
        gravity_mm=float(gravity_mm),
        generation=int(generation),
    )


def load_policy(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "mode": str(data["mode"]),
            "vector": data["vector"],
            "score": float(data["score"]),
            "hidden": int(data["hidden"]),
            "observation_size": int(data["observation_size"]),
            "group_count": int(data["group_count"]),
            "steps": int(data["steps"]),
            "gravity_mm": float(data["gravity_mm"]),
            "generation": int(data["generation"]) if "generation" in data.files else 0,
        }


def controller_from_policy(policy, observation_size, group_count):
    """Build a controller; raises ValueError if the policy does not fit this chain."""
    if policy["group_count"] != group_count:
        raise ValueError(f"Policy has {policy['group_count']} muscle groups, chain has {group_count}")
    if policy["mode"] == "static":
        return static_controller(policy["vector"])
    if policy["observation_size"] != observation_size:
        raise ValueError(f"Policy expects {policy['observation_size']} observation values, chain gives {observation_size}")
    return Policy(observation_size, group_count, policy["hidden"]).controller(policy["vector"])
