"""Chain package: everything the chain solver needs, saved to disk, loadable without bpy.

A package is a directory with `chain.json` (names, labels, settings) and
`chain.npz` (arrays). Bone distance fields are referenced by their cached `.npz`
paths; the loader reads them and rebuilds `ChainSolver` objects, so posture
optimization and policy training can run in a plain Python process and the
result can be brought back into Blender, where the same solver runs live.
"""

import json
import os

import numpy as np

from .chain_solver import Body, ChainSolver, CollisionPair, DistanceConstraint, MuscleElement, PoseConstraint
from .distance_field import DistanceField

PACKAGE_JSON = "chain.json"
PACKAGE_NPZ = "chain.npz"
MODE_CODES = {"both": 0, "tension": 1, "compression": 2}
MODE_NAMES = {code: name for name, code in MODE_CODES.items()}


class CenteredField:
    """A DistanceField in object-local coordinates queried in body-centered coordinates."""

    def __init__(self, field, center):
        self.field = field
        self.center = np.asarray(center, dtype=np.float64)

    def sample(self, points):
        return self.field.sample(np.asarray(points) + self.center)

    def sample_with_gradient(self, points):
        return self.field.sample_with_gradient(np.asarray(points) + self.center)


def save_package(directory, meta, arrays):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, PACKAGE_JSON), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    np.savez_compressed(os.path.join(directory, PACKAGE_NPZ), **arrays)


def load_package(directory):
    with open(os.path.join(directory, PACKAGE_JSON), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(os.path.join(directory, PACKAGE_NPZ)) as data:
        arrays = {key: data[key] for key in data.files}
    return meta, arrays


def build_solver(meta, arrays, field_cache=None, iterations=None, substeps=None, gravity=None):
    """Rebuild bodies, constraints, muscles, collision pairs, and a ChainSolver from a package."""
    field_cache = {} if field_cache is None else field_cache
    bodies = []
    for index, name in enumerate(meta["bodies"]):
        body = Body(
            name,
            arrays["body_position"][index],
            arrays["body_quaternion"][index],
            mass=float(arrays["body_mass"][index]),
            radius=float(arrays["body_radius"][index]),
            fixed=bool(arrays["body_fixed"][index]),
        )
        body.radius = float(arrays["body_radius"][index])
        bodies.append(body)

    constraints = []
    for index in range(len(arrays["c_a"])):
        constraints.append(
            DistanceConstraint(
                bodies[int(arrays["c_a"][index])],
                arrays["c_local_a"][index],
                bodies[int(arrays["c_b"][index])],
                arrays["c_local_b"][index],
                float(arrays["c_rest"][index]),
                compliance=float(arrays["c_compliance"][index]),
                slack=float(arrays["c_slack"][index]),
                name=meta["constraint_names"][index],
                mode=MODE_NAMES[int(arrays["c_mode"][index])],
            )
        )

    muscles = []
    for index in range(len(arrays["m_a"])):
        muscles.append(
            MuscleElement(
                bodies[int(arrays["m_a"][index])],
                arrays["m_local_a"][index],
                bodies[int(arrays["m_b"][index])],
                arrays["m_local_b"][index],
                int(arrays["m_group"][index]),
                float(arrays["m_strength"][index]),
                name=meta["muscle_names"][index],
            )
        )

    fields = {}
    for body_index, path in meta["field_paths"].items():
        body_index = int(body_index)
        if path not in field_cache:
            with np.load(path) as data:
                field_cache[path] = DistanceField.from_npz_dict(data)
        fields[body_index] = CenteredField(field_cache[path], arrays["body_center"][body_index])

    pairs = []
    for index, (a, b) in enumerate(zip(arrays["pair_a"], arrays["pair_b"])):
        a, b = int(a), int(b)
        pairs.append(
            CollisionPair(
                bodies[a],
                arrays[f"samples_{a}"],
                fields[a],
                bodies[b],
                arrays[f"samples_{b}"],
                fields[b],
                margin=float(arrays["pair_margin"][index]),
            )
        )

    holds = []
    for hold in meta.get("holds", []):
        body = bodies[int(hold["body"])]
        holds.append(
            PoseConstraint(
                body,
                body.position.copy(),
                body.quaternion.copy(),
                position_compliance=float(hold.get("position_compliance", 1e6)),
                rotation_compliance=float(hold["rotation_compliance"]),
                max_position_step=0.001,
            )
        )
    solver = ChainSolver(
        bodies,
        constraints,
        pairs,
        holds,
        iterations=meta["iterations"] if iterations is None else iterations,
        substeps=meta["substeps"] if substeps is None else substeps,
        gravity=meta["gravity"] if gravity is None else gravity,
        muscles=muscles,
        group_count=len(meta["muscle_groups"]),
    )
    return solver


def package_from_session(session, scene_settings):
    """Serialize a live ChainSession (Blender side) into (meta, arrays)."""
    solver = session.solver
    bodies = solver.bodies
    index_of = {body.name: index for index, body in enumerate(bodies)}
    meta = {
        "bodies": [body.name for body in bodies],
        "chain": [name for name in session.bodies if not session.bodies[name].fixed or name == scene_settings["base"]],
        "base": scene_settings["base"],
        "top": scene_settings["top"],
        "units_per_mm": scene_settings["units_per_mm"],
        "iterations": solver.iterations,
        "substeps": solver.substeps,
        "gravity": [float(value) for value in solver.gravity],
        "constraint_names": [constraint.name for constraint in solver.distance_constraints],
        "muscle_names": [muscle.name for muscle in solver.muscles],
        "muscle_groups": [list(key) for key in session.muscle_groups],
        "muscle_group_labels": scene_settings.get("group_labels", []),
        "field_paths": {str(index_of[name]): path for name, path in scene_settings["field_paths"].items()},
        "holds": [
            {"body": index_of[hold.body.name], "rotation_compliance": hold.rotation_compliance, "position_compliance": hold.position_compliance}
            for hold in getattr(session, "holds", [])
        ],
    }
    arrays = {
        "body_position": np.array([body.position for body in bodies]),
        "body_quaternion": np.array([body.quaternion for body in bodies]),
        "body_mass": np.array([body.mass for body in bodies]),
        "body_radius": np.array([body.radius for body in bodies]),
        "body_fixed": np.array([body.fixed for body in bodies]),
        "body_center": np.array([session.centers.get(body.name, np.zeros(3)) for body in bodies]),
        "c_a": solver.c_a,
        "c_b": solver.c_b,
        "c_local_a": solver.c_local_a,
        "c_local_b": solver.c_local_b,
        "c_rest": solver.c_rest,
        "c_compliance": solver.c_compliance,
        "c_slack": np.array([constraint.slack for constraint in solver.distance_constraints]),
        "c_mode": solver.c_mode,
        "m_a": solver.m_a,
        "m_b": solver.m_b,
        "m_local_a": solver.m_local_a,
        "m_local_b": solver.m_local_b,
        "m_group": solver.m_group,
        "m_strength": solver.m_strength,
        "pair_a": np.array([pair.body_a.index for pair in solver.collision_pairs], dtype=np.int64),
        "pair_b": np.array([pair.body_b.index for pair in solver.collision_pairs], dtype=np.int64),
        "pair_margin": np.array([pair.margin for pair in solver.collision_pairs]),
    }
    for pair in solver.collision_pairs:
        arrays[f"samples_{pair.body_a.index}"] = pair.samples_a
        arrays[f"samples_{pair.body_b.index}"] = pair.samples_b
    return meta, arrays
