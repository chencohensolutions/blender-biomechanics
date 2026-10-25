"""Interactive bone chain: drag one Rigid vertebra and the chain follows.

The chain is the run of vertebrae above a fixed base bone (default sacrum) up
to a top bone (default T01). Every ligament and disc string whose bones are all
in the chain (or the base) becomes a distance constraint with its rest length
taken from the Original pose; adjacent bones get a collision pair backed by their
signed distance fields. The handle bone is pulled toward a target Empty.

Each depsgraph update of the target runs one solve and writes the bones; a
timer keeps solving while the pose is still moving, so a held target settles
to the static equilibrium of ligaments, discs, and contacts. Ribs of chain
vertebrae follow their vertebra rigidly.
"""

import os
import time

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np
from bpy.app.handlers import persistent  # pyright: ignore[reportMissingImports]
from mathutils import Matrix  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, local_vertices, world_vertices
from .bone_fields import field_for_object, field_path, has_cached_field, surface_samples
from .chain_package import package_from_session, save_package
from .posture_policy import ChainState, controller_from_policy, load_policy
from .chain_solver import (
    Body,
    ChainSolver,
    CollisionPair,
    DistanceConstraint,
    MuscleElement,
    PoseConstraint,
    matrix_to_quat,
)
from .muscle_atlas import SPINE, THORACIC
from .muscle_strings import (
    _restore_string_color,
    _set_string_color,
    apply_delta_colors,
    cached_object,
    is_delta_active,
    muscle_collections,
    string_info,
)

FEMUR_BASE = "femurs"
# Ball joints are zero-length links with a whisper of compliance: fully rigid ones
# oscillate against the ligaments at the solver's sweep budget.
JOINT_COMPLIANCE = 1e-4
FEMURS = ("femur L", "femur R")
PELVIS = ("ilium L", "ilium R", "sacrum")
STERNUM = "strenum"
TARGET_NAME = "Biomechanics Chain Target"
TARGET_PROPERTY = "biomechanics_chain_target"
SETTLE_INTERVAL = 0.03
SETTLE_THRESHOLD = 3e-5  # Blender units (0.03 mm) of maximum body movement per solve
MAX_SETTLE_SOLVES = 150

_session = None
_solving = False
_force_colors = False
FORCE_BINS = 12
MUSCLE_COLOR_LOW = (0.92, 0.86, 0.86)
MUSCLE_COLOR_HIGH = (1.0, 0.08, 0.04)
LIGAMENT_COLOR_LOW = (0.86, 0.88, 0.94)
LIGAMENT_COLOR_HIGH = (0.05, 0.3, 1.0)


class SolverDiverged(RuntimeError):
    pass


class ChainSession:
    def __init__(self, scene, handle=None):
        """handle: the Rigid vertebra to drive, or None for free mode (grab any chain bone)."""
        self.scene_key = scene.as_pointer()
        self.objects = {}
        self.centers = {}
        self.bodies = {}
        self.start_matrices = {}
        self.followers = []
        self.free_mode = handle is None
        self.handle_name = None if handle is None else handle.name
        self.solver = None
        self.pose = None
        self.pinned_top = None
        self.muscle_groups = []  # (muscle key, side) per activation index
        self.anchors = {}  # fixed bodies added only to anchor muscles
        self.field_paths = {}  # body name -> cached field file, for export
        self.controller = None  # posture controller: observation -> activations
        self.state = None
        self.policy_score = None
        self.policy_generation = None
        self.policy_steps = None  # episode length the policy was trained for
        self.replay_budget = None  # solves left in a Follow Training replay
        self.force_scale = 0.0  # running maximum ligament force, for Force Colors
        self.last_solve_ms = 0.0
        self.settle_count = 0
        self.last_movement = 0.0
        self.written = {}  # object name -> matrix the solver last wrote
        self._build(scene, handle)

    def _build(self, scene, handle):
        # Matrices assigned by a previous restore or reset are only reliable after evaluation.
        bpy.context.view_layer.update()
        rigid = scene.biomechanics_target_collection
        by_base = {}
        for obj in rigid.objects:
            if obj.type == "MESH":
                by_base.setdefault(base_object_name(obj.name), obj)
        base = scene.biomechanics_chain_base
        top = scene.biomechanics_chain_top
        pelvis_free = base == FEMUR_BASE
        fixed_names, pelvis = (set(FEMURS), list(FEMURS) + list(PELVIS)) if pelvis_free else ({base}, [base])
        chain = chain_levels("sacrum" if pelvis_free else base, top)
        # The sternum is carried by the costal cartilages of whichever ribs are in the chain.
        extras = [STERNUM] if STERNUM in by_base and any(level in chain for level in THORACIC) else []
        grabbable = chain + (list(PELVIS) if pelvis_free else []) + extras
        if handle is not None and base_object_name(handle.name) not in grabbable:
            raise ValueError(f"{handle.name} is not between {base} and {top}")
        missing = [name for name in pelvis + chain if name not in by_base]
        if missing:
            raise ValueError(f"Missing chain bones: {', '.join(missing)}")

        bones = {}
        pinned = set(fixed_names)
        if scene.biomechanics_chain_pin_top and (handle is None or chain[-1] != base_object_name(handle.name)):
            pinned.add(chain[-1])
            self.pinned_top = chain[-1]
        for name in pelvis + chain + extras:
            obj = by_base[name]
            vertices = local_vertices(obj)
            center = vertices.mean(axis=0)
            extent = vertices.max(axis=0) - vertices.min(axis=0)
            matrix = np.array(obj.matrix_world, dtype=np.float64)
            position = matrix[:3, :3] @ center + matrix[:3, 3]
            body = Body(
                name,
                position,
                matrix_to_quat(matrix[:3, :3]),
                mass=1.0,
                radius=float(np.linalg.norm(extent)) / 2.0,
                fixed=(name in pinned),
            )
            body.radius = float(np.linalg.norm(extent)) / 2.0
            bones[name] = body
            self.objects[name] = obj.name
            self.centers[name] = center
            self.start_matrices[obj.name] = obj.matrix_world.copy()
            self.written[obj.name] = obj.matrix_world.copy()
        self.bodies = bones

        constraints = self._ligament_constraints(scene, by_base)
        if pelvis_free:
            constraints += self._pelvic_joints(scene, by_base)
        muscles = self._muscle_elements(scene, by_base, chain)
        pairs = []
        margin = scene.biomechanics_chain_margin_mm / _units(scene)
        contacts = list(zip(["sacrum" if pelvis_free else base] + chain, chain))
        # With the pelvis free, hips, sacroiliac joints, and the symphysis are ball joints
        # (see _pelvic_joints); bone contacts there would only fight them.
        for lower, upper in contacts:
            pairs.append(self._collision_pair(scene, by_base, lower, upper, margin))

        self.holds = []
        if pelvis_free and scene.biomechanics_chain_pelvis_hold > 0.0 and "sacrum" in bones:
            # Stand-in for the hip and balance muscles: the pelvis keeps its place and
            # tilt unless pushed (the trunk's weight otherwise sinks it through the hips).
            sacrum = bones["sacrum"]
            self.holds.append(
                PoseConstraint(
                    sacrum,
                    sacrum.position.copy(),
                    sacrum.quaternion.copy(),
                    position_compliance=scene.biomechanics_chain_pelvis_hold,
                    rotation_compliance=scene.biomechanics_chain_pelvis_hold,
                    max_position_step=0.001,
                )
            )
        self.solver = ChainSolver(
            list(bones.values()) + list(self.anchors.values()),
            constraints,
            pairs,
            list(self.holds),
            iterations=scene.biomechanics_chain_iterations,
            substeps=scene.biomechanics_chain_substeps,
            gravity=_gravity_vector(scene),
            muscles=muscles,
            group_count=len(self.muscle_groups),
        )
        self.constraint_count = len(constraints)
        self.muscle_count = len(muscles)
        self.state = ChainState(self.solver, 1.0 / _units(scene))
        self.set_tone(scene.biomechanics_muscle_tone)
        self._collect_followers(scene, rigid, chain)
        if handle is not None:
            self.set_handle(scene, base_object_name(handle.name))

    def set_handle(self, scene, name):
        """Make the chain bone `name` the driven one; a pinned top bone is freed while held."""
        body = self.bodies[name]
        if self.pinned_top is not None:
            top = self.bodies[self.pinned_top]
            top.fixed = name != self.pinned_top
            top.inverse_mass = 0.0 if top.fixed else 1.0
            top.inverse_inertia = 0.0 if top.fixed else 1.0 / (0.4 * top.radius * top.radius)
        self.handle_name = self.objects[name]
        self.pose = PoseConstraint(
            body,
            body.position.copy(),
            body.quaternion.copy(),
            position_compliance=scene.biomechanics_chain_handle_compliance,
            rotation_compliance=scene.biomechanics_chain_handle_compliance,
        )
        self.solver.pose_constraints = [self.pose] + list(self.holds)
        self.solver._build_arrays()

    def grabbed_bone(self, updated_objects):
        """Return the chain object the user moved (differs from what the solver wrote), or None."""
        for name, object_name in self.objects.items():
            if self.bodies[name].fixed and name != self.pinned_top:
                continue
            obj = bpy.data.objects.get(object_name)
            if obj is None or obj not in updated_objects:
                continue
            written = self.written.get(object_name)
            if written is not None and not _matrices_match(obj.matrix_world, written):
                return obj
        return None

    def _ligament_constraints(self, scene, by_base):
        source = scene.biomechanics_source_collection
        ligament_sets = muscle_collections(scene, "ligament")
        rigid_strings = ligament_sets.get(scene.biomechanics_target_collection.name)
        source_strings = ligament_sets.get(source.name) if source else None
        if rigid_strings is None or source_strings is None:
            raise ValueError("Build Ligaments & Discs first")
        # Rest lengths come from the Original pose only on request; by default the
        # pose at session start is the equilibrium, so the chain holds still until dragged.
        rest_strings = source_strings if scene.biomechanics_chain_rest_from_original else rigid_strings
        source_by_key = {}
        for obj in rest_strings.objects:
            info = string_info(obj)
            if info is not None:
                source_by_key[(info["muscle"], info["side"], info["index"])] = info

        constraints = []
        for obj in rigid_strings.objects:
            info = string_info(obj)
            if info is None:
                continue
            points = [self._muscle_attachment(scene, by_base, bone, index, chain=None) for bone, index in info["attachments"]]
            if any(point is None for point in points) or all(point[0].fixed for point in points):
                continue
            rest_info = source_by_key.get((info["muscle"], info["side"], info["index"]))
            if rest_info is None:
                continue
            rest_points = [_world_vertex(bone, index) for bone, index in rest_info["attachments"]]
            if any(point is None for point in rest_points):
                continue
            for segment in range(len(points) - 1):
                (body_a, local_a), (body_b, local_b) = points[segment], points[segment + 1]
                if body_a is body_b:
                    continue
                rest = float(np.linalg.norm(rest_points[segment + 1] - rest_points[segment]))
                constraints.append(
                    DistanceConstraint(
                        body_a,
                        local_a,
                        body_b,
                        local_b,
                        rest,
                        compliance=info.get("compliance", 0.5),
                        unilateral=info.get("unilateral", True),
                        slack=info.get("slack", 0.0),
                        name=obj.name,
                        mode=info.get("mode"),
                    )
                )
        return constraints

    def _pelvic_joints(self, scene, by_base):
        """Near-rigid zero-length links: hips at the fitted femoral head centers, sacroiliac
        joints at the interosseous ligament centers, pubic symphysis at its ligament center."""
        joints = self._hip_joints(scene, by_base)
        body = self.bodies
        centers = self._string_centers(scene)
        for side in ("L", "R"):
            center = centers.get(("Interosseous sacroiliac ligament", side))
            if center is not None and "sacrum" in body and f"ilium {side}" in body:
                a, b = body["sacrum"], body[f"ilium {side}"]
                joints.append(DistanceConstraint(a, a.local_point(center), b, b.local_point(center), 0.0, compliance=JOINT_COMPLIANCE, name=f"si joint {side}", mode="both"))
        center = centers.get(("Pubic symphysis ligaments", "M"))
        if center is not None and "ilium L" in body and "ilium R" in body:
            a, b = body["ilium L"], body["ilium R"]
            joints.append(DistanceConstraint(a, a.local_point(center), b, b.local_point(center), 0.0, compliance=JOINT_COMPLIANCE, name="pubic joint", mode="both"))
        return joints

    def _string_centers(self, scene):
        """Mean world point of all attachments of each Rigid ligament label, by (label, side)."""
        centers = {}
        collection = muscle_collections(scene, "ligament").get(scene.biomechanics_target_collection.name)
        if collection is None:
            return centers
        groups = {}
        for obj in collection.objects:
            info = string_info(obj)
            if info is None:
                continue
            for bone, index in info["attachments"]:
                point = _world_vertex(bone, index)
                if point is not None:
                    groups.setdefault((info["label"], info["side"]), []).append(point)
        for key, points in groups.items():
            centers[key] = np.mean(points, axis=0)
        return centers

    def _hip_joints(self, scene, by_base):
        """Near-rigid zero-length links at the femoral head centers: ball joints that keep
        the pelvis seated on the femurs while leaving rotation free."""
        joints = []
        for side in ("L", "R"):
            femur, ilium = by_base.get(f"femur {side}"), by_base.get(f"ilium {side}")
            if femur is None or ilium is None or f"ilium {side}" not in self.bodies:
                continue
            field = field_for_object(ilium, scene)
            femur_world = world_vertices(femur)
            local_in_ilium = (femur_world - np.array(ilium.matrix_world.translation)) @ np.array(ilium.matrix_world.to_3x3())
            distances = field.sample(local_in_ilium)
            articular = femur_world[np.abs(distances) < 0.006]
            if len(articular) < 50:
                print(f"Biomechanics: hip {side}: only {len(articular)} articular vertices, no ball joint")
                continue
            center, radius = _fit_sphere(articular)
            femur_body = self.bodies[f"femur {side}"]
            ilium_body = self.bodies[f"ilium {side}"]
            joints.append(
                DistanceConstraint(
                    ilium_body,
                    ilium_body.local_point(center),
                    femur_body,
                    femur_body.local_point(center),
                    0.0,
                    compliance=JOINT_COMPLIANCE,
                    name=f"hip joint {side}",
                    mode="both",
                )
            )
            self.hip_radius_mm = getattr(self, "hip_radius_mm", {})
            self.hip_radius_mm[side] = radius * _units(scene)
        return joints

    def _anchor_body(self, obj):
        """A fixed body for a bone outside the chain that muscles pull against."""
        name = base_object_name(obj.name)
        body = self.anchors.get(name)
        if body is None:
            vertices = local_vertices(obj)
            center = vertices.mean(axis=0)
            matrix = np.array(obj.matrix_world, dtype=np.float64)
            body = Body(name, matrix[:3, :3] @ center + matrix[:3, 3], matrix_to_quat(matrix[:3, :3]), fixed=True)
            body.radius = 0.05
            self.anchors[name] = body
            self.centers[name] = center
        return body

    def _muscle_attachment(self, scene, by_base, bone_name, index, chain):
        """Map a muscle attachment to (body, local offset); rib points ride on their vertebra."""
        base = base_object_name(bone_name)
        obj = bpy.data.objects.get(bone_name)
        if obj is None:
            return None
        if base in self.bodies:
            return self.bodies[base], _local_offset(obj, index, self.centers[base])
        if " rib " in f" {base} ":
            vertebra = base.split(" rib ")[0]
            if vertebra not in self.bodies:
                return None
            leader = by_base[vertebra]
            # Ribs follow their vertebra rigidly, so express the rib vertex in the vertebra frame.
            local = leader.matrix_world.inverted() @ obj.matrix_world @ obj.data.vertices[index].co
            return self.bodies[vertebra], np.array(local, dtype=np.float64) - self.centers[vertebra]
        if base in ("sacrum", "ilium L", "ilium R", "femur L", "femur R", STERNUM):
            body = self._anchor_body(by_base[base]) if base in by_base else None
            if body is None:
                return None
            return body, _local_offset(by_base[base], index, self.centers[base])
        return None

    def _muscle_elements(self, scene, by_base, chain):
        muscle_sets = muscle_collections(scene, "muscle")
        strings = muscle_sets.get(scene.biomechanics_target_collection.name)
        if strings is None:
            return []
        strength = scene.biomechanics_muscle_strength_mm / _units(scene)
        group_index = {}
        elements = []
        for obj in strings.objects:
            info = string_info(obj)
            if info is None:
                continue
            points = [self._muscle_attachment(scene, by_base, bone, index, chain) for bone, index in info["attachments"]]
            if any(point is None for point in points):
                continue
            if all(point[0].fixed for point in points):
                continue
            key = (info["muscle"], info["side"])
            if key not in group_index:
                group_index[key] = len(self.muscle_groups)
                self.muscle_groups.append(key)
            for (body_a, local_a), (body_b, local_b) in zip(points, points[1:]):
                if body_a is body_b:
                    continue
                elements.append(MuscleElement(body_a, local_a, body_b, local_b, group_index[key], strength, name=obj.name))
        return elements

    def set_tone(self, tone):
        """Uniform activation for every muscle group (manual test of the actuators)."""
        if self.solver is not None and self.controller is None:
            self.solver.activations[:] = float(tone)

    def set_controller(self, path):
        """Load a trained posture policy; None clears it and returns to Muscle Tone."""
        if path is None:
            self.controller = None
            self.policy_score = None
            return
        policy = load_policy(path)
        self.controller = controller_from_policy(policy, self.state.size, len(self.muscle_groups))
        self.policy_score = policy["score"]
        self.policy_generation = policy["generation"]
        self.policy_steps = policy["steps"]

    def _collision_pair(self, scene, by_base, name_a, name_b, margin):
        obj_a, obj_b = by_base[name_a], by_base[name_b]
        field_a = field_for_object(obj_a, scene)
        field_b = field_for_object(obj_b, scene)
        self.field_paths[name_a] = field_path(scene, obj_a, 1.0)
        self.field_paths[name_b] = field_path(scene, obj_b, 1.0)
        samples_a = surface_samples(obj_a) - self.centers[name_a]
        samples_b = surface_samples(obj_b) - self.centers[name_b]
        # Fields are in object-local coordinates; bodies are centered, so shift the field origin.
        pair = CollisionPair(
            self.bodies[name_a],
            samples_a,
            _CenteredField(field_a, self.centers[name_a]),
            self.bodies[name_b],
            samples_b,
            _CenteredField(field_b, self.centers[name_b]),
            margin=margin,
        )
        # Bones that already touch at rest keep their rest clearance instead of being pushed apart.
        rest_clearance = min(
            float(field_b.sample((surface_samples(obj_a) @ _relative(obj_a, obj_b)[0].T + _relative(obj_a, obj_b)[1])).min()),
            float(field_a.sample((surface_samples(obj_b) @ _relative(obj_b, obj_a)[0].T + _relative(obj_b, obj_a)[1])).min()),
        )
        pair.margin = min(margin, max(rest_clearance - 1e-5, -1.0))
        return pair

    def _follower_leader(self, base, chain):
        """Which chain body a non-chain bone rides on, or None to leave it where it is."""
        if " rib " in f" {base} " or " cartilage " in f" {base} ":
            vertebra = base.split(" ")[0]
            return vertebra if vertebra in chain else None
        head_neck = base in ("skull", "mandible", "hyoid") or (len(base) == 2 and base[0] == "C" and base[1].isdigit())
        if head_neck:
            return "T01" if "T01" in chain else None
        if base.startswith("clavicle"):
            return STERNUM if STERNUM in self.bodies else ("T01" if "T01" in chain else None)
        if base.split(" ")[0] in ("scapula", "humerus", "radius", "ulna", "hand"):
            for level in ("T04", "T03", "T02", "T01"):
                if level in chain:
                    return level
        return None

    def _collect_followers(self, scene, rigid, chain):
        if not scene.biomechanics_chain_move_ribs:
            return
        for obj in rigid.objects:
            if obj.type != "MESH" or obj.hide_get():
                continue
            base = base_object_name(obj.name)
            if base in self.bodies or base in self.anchors:
                continue
            leader_name = self._follower_leader(base, chain)
            if leader_name is None or leader_name not in self.objects:
                continue
            leader = bpy.data.objects[self.objects[leader_name]]
            self.followers.append((obj.name, leader_name, leader.matrix_world.inverted() @ obj.matrix_world))
            self.start_matrices[obj.name] = obj.matrix_world.copy()

    def set_target(self, target_matrix):
        if self.pose is None:
            return
        matrix = np.array(target_matrix, dtype=np.float64)
        center = self.centers[base_object_name(self.handle_name)]
        self.pose.target_position = matrix[:3, :3] @ center + matrix[:3, 3]
        self.pose.target_quaternion = matrix_to_quat(matrix[:3, :3])

    def solve_and_apply(self, scene):
        started = time.perf_counter()
        before = {name: body.position.copy() for name, body in self.bodies.items()}
        self.solver.iterations = scene.biomechanics_chain_iterations
        self.solver.substeps = scene.biomechanics_chain_substeps
        self.solver.gravity = _gravity_vector(scene)
        if self.controller is not None:
            self.solver.activations[:] = np.clip(self.controller(self.state.observe()), 0.0, 1.0)
        self.solver.solve()
        if not self.solver.is_finite() or self._runaway():
            raise SolverDiverged("chain solver diverged")
        if _force_colors:
            apply_force_colors(self)
        self.last_movement = max(
            float(np.linalg.norm(body.position - before[name])) for name, body in self.bodies.items()
        )
        for name, body in self.bodies.items():
            if body.fixed:
                continue
            obj = bpy.data.objects.get(self.objects[name])
            if obj is None:
                continue
            rotation = body.rotation
            translation = body.position - rotation @ self.centers[name]
            matrix = Matrix.Identity(4)
            for row in range(3):
                for column in range(3):
                    matrix[row][column] = float(rotation[row, column])
                matrix[row][3] = float(translation[row])
            obj.matrix_world = matrix
            self.written[obj.name] = matrix.copy()
        for follower_name, leader_name, relative in self.followers:
            follower = bpy.data.objects.get(follower_name)
            # Use the matrix just written: a bone-parented leader's matrix_world reads
            # stale until the depsgraph re-evaluates.
            leader_matrix = self.written.get(self.objects[leader_name])
            if follower is not None and leader_matrix is not None:
                follower.matrix_world = leader_matrix @ relative
        self.last_solve_ms = (time.perf_counter() - started) * 1000.0

    def _runaway(self, limit=0.5):
        """True when any body is more than `limit` meters from where it started."""
        for name, body in self.bodies.items():
            start = self.start_matrices.get(self.objects[name])
            if start is not None and np.linalg.norm(body.position - np.array(start.translation) - body.rotation @ self.centers[name]) > limit:
                return True
        return False

    def restore(self):
        for name, matrix in self.start_matrices.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.matrix_world = matrix

    def status(self, scene):
        units = _units(scene)
        return {
            "penetration_mm": self.solver.max_penetration() * units,
            "strain_percent": self.solver.max_strain() * 100.0,
            "solve_ms": self.last_solve_ms,
            "bodies": len(self.bodies),
            "constraints": self.constraint_count,
            "muscles": self.muscle_count,
            "muscle_groups": len(self.muscle_groups),
            "activation_mean": float(self.solver.activations.mean()) if len(self.solver.activations) else 0.0,
            "controller": self.controller is not None,
            "policy_score": self.policy_score,
            "policy_generation": self.policy_generation,
            "height_cm": self.top_height_cm(scene),
        }

    def top_height_cm(self, scene):
        """Top chain bone height relative to session start, in cm (negative = lower)."""
        top = scene.biomechanics_chain_top
        body = self.bodies.get(top)
        start = self.start_matrices.get(self.objects.get(top, ""))
        if body is None or start is None:
            return 0.0
        start_center = np.array(start.to_3x3()) @ self.centers[top] + np.array(start.translation)
        return float(body.position[2] - start_center[2]) * _units(scene) / 10.0


class _CenteredField:
    """A DistanceField queried in body-centered coordinates."""

    def __init__(self, field, center):
        self.field = field
        self.center = np.asarray(center, dtype=np.float64)

    def sample(self, points):
        return self.field.sample(np.asarray(points) + self.center)

    def sample_with_gradient(self, points):
        return self.field.sample_with_gradient(np.asarray(points) + self.center)


def _fit_sphere(points):
    """Least-squares sphere through points: returns (center, radius)."""
    points = np.asarray(points, dtype=np.float64)
    a = np.hstack((2.0 * points, np.ones((len(points), 1))))
    b = (points ** 2).sum(axis=1)
    solution, _residuals, _rank, _sv = np.linalg.lstsq(a, b, rcond=None)
    center = solution[:3]
    radius = float(np.sqrt(solution[3] + center @ center))
    return center, radius


def _relative(obj_from, obj_to):
    """(rotation, translation) mapping obj_from local points into obj_to local space."""
    matrix = np.array(obj_to.matrix_world.inverted() @ obj_from.matrix_world, dtype=np.float64)
    return matrix[:3, :3], matrix[:3, 3]


def _units(scene):
    return (scene.unit_settings.scale_length or 1.0) * 1000.0


def _gravity_vector(scene):
    """World -Z displacement per solver round, from the panel's mm setting."""
    if not scene.biomechanics_chain_gravity:
        return np.zeros(3)
    return np.array([0.0, 0.0, -scene.biomechanics_chain_gravity_mm / _units(scene)])


def _world_vertex(bone_name, index):
    bone = bpy.data.objects.get(bone_name)
    if bone is None or bone.type != "MESH" or index >= len(bone.data.vertices):
        return None
    return np.array(bone.matrix_world @ bone.data.vertices[index].co, dtype=np.float64)


def _local_offset(obj, index, center):
    return np.array(obj.data.vertices[index].co, dtype=np.float64) - center


def chain_levels(base, top):
    """Vertebra names above base up to and including top, lowest first."""
    order = list(reversed(SPINE))  # L5 ... T01
    if base == "sacrum":
        start = 0
    elif base in order:
        start = order.index(base) + 1
    else:
        raise ValueError(f"Unknown base bone {base}")
    if top not in order:
        raise ValueError(f"Unknown top bone {top}")
    end = order.index(top) + 1
    if end <= start:
        raise ValueError("Top bone must be above the base bone")
    return order[start:end]


def chain_bone_names(scene):
    base = scene.biomechanics_chain_base
    try:
        if base == FEMUR_BASE:
            return list(FEMURS) + list(PELVIS) + chain_levels("sacrum", scene.biomechanics_chain_top)
        return [base] + chain_levels(base, scene.biomechanics_chain_top)
    except ValueError:
        return []


def session():
    return _session


def is_active():
    return _session is not None


def _target_object(scene):
    target = scene.biomechanics_chain_target_object
    return target if target is not None and target.name in bpy.data.objects else None


def start_session(context, handle):
    global _session
    scene = context.scene
    stop_session(scene)
    _session = ChainSession(scene, handle)
    target = bpy.data.objects.new(TARGET_NAME, None)
    target.empty_display_type = "ARROWS"
    target.empty_display_size = 0.12
    target.show_in_front = True
    target[TARGET_PROPERTY] = True
    scene.collection.objects.link(target)
    target.matrix_world = handle.matrix_world.copy()
    scene.biomechanics_chain_target_object = target
    _hide_original_strings(scene)
    for obj in context.view_layer.objects:
        obj.select_set(False)
    target.select_set(True)
    context.view_layer.objects.active = target
    _session.last_movement = float("inf")
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE_INTERVAL)
    return _session


def stop_session(scene, restore=False):
    global _session
    if _session is not None and restore:
        _session.restore()
    if _session is not None and _force_colors:
        _restore_force_colors(_session)
    _session = None
    _show_original_strings()
    if bpy.app.timers.is_registered(_settle):
        bpy.app.timers.unregister(_settle)
    target = _target_object(scene)
    if target is not None:
        bpy.data.objects.remove(target, do_unlink=True)
    scene.biomechanics_chain_target_object = None


def _solve_now(scene):
    global _solving
    if _session is None or _solving:
        return
    _solving = True
    try:
        if not _session.free_mode:
            target = _target_object(scene)
            if target is None:
                return
            _session.set_target(target.matrix_world)
        try:
            _session.solve_and_apply(scene)
        except SolverDiverged:
            _recover_session(scene)
    finally:
        _solving = False


def _recover_session(scene):
    """Put the bones back where the session started and rebuild it (controller kept)."""
    global _session
    old = _session
    old.restore()
    bpy.context.view_layer.update()
    path = bpy.path.abspath(scene.biomechanics_posture_policy_path) if old.controller is not None else None
    rebuilt = ChainSession(scene, None if old.free_mode else bpy.data.objects.get(old.handle_name))
    if path is not None:
        try:
            rebuilt.set_controller(path)
        except (OSError, ValueError, KeyError):
            pass
    rebuilt.force_scale = 0.0
    _session = rebuilt
    print("Biomechanics: chain solver diverged; pose restored and session rebuilt")


def _kick_settle():
    if _session is None:
        return
    _session.settle_count = 0
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE_INTERVAL)


def _settle():
    scene = bpy.context.scene
    if _session is None:
        return None
    if _session.replay_budget is not None and _session.settle_count >= _session.replay_budget:
        return None  # the replay ran the trained episode length; hold the pose
    if _session.last_movement <= SETTLE_THRESHOLD or _session.settle_count >= MAX_SETTLE_SOLVES:
        return None
    _session.settle_count += 1
    _solve_now(scene)
    _redraw()
    return SETTLE_INTERVAL


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _matrices_match(first, second, tolerance=1e-6):
    return all(abs(first[row][col] - second[row][col]) <= tolerance for row in range(4) for col in range(4))


@persistent
def chain_depsgraph_update(_scene, depsgraph):
    if _session is None or _solving:
        return
    scene = bpy.context.scene
    updated = {getattr(update.id, "original", None) or update.id for update in depsgraph.updates}
    if _session.free_mode:
        grabbed = _session.grabbed_bone(updated)
        if grabbed is None:
            return
        if grabbed.name != _session.handle_name:
            _session.set_handle(scene, base_object_name(grabbed.name))
        _session.set_target(grabbed.matrix_world)
        _solve_now(scene)
        _kick_settle()
        return
    target = _target_object(scene)
    if target is None:
        return
    handle = bpy.data.objects.get(_session.handle_name)
    if handle is not None and handle in updated and target not in updated:
        # The user grabbed the vertebra itself: its pose becomes the new target.
        written = _session.written.get(handle.name)
        if written is not None and not _matrices_match(handle.matrix_world, written):
            target.matrix_world = handle.matrix_world.copy()
        else:
            return
    elif target not in updated:
        return
    _solve_now(scene)
    _session.settle_count = 0
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE_INTERVAL)


def update_tone(scene, _context):
    if _session is None:
        return
    _session.set_tone(scene.biomechanics_muscle_tone)
    update_gravity(scene, _context)


def update_gravity(scene, _context):
    if _session is None:
        return
    _session.settle_count = 0
    _session.last_movement = float("inf")
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE_INTERVAL)


_hidden_original_strings = []


def _hide_original_strings(scene):
    """Original strings never move; hide them so they are not mistaken for stuck fibers."""
    global _hidden_original_strings
    source = scene.biomechanics_source_collection
    if source is None or not scene.biomechanics_chain_hide_original_strings:
        return
    for _bones, _kind, collection in string_collections_of(scene, source.name):
        layer = _layer_collection(bpy.context.view_layer.layer_collection, collection)
        if layer is not None and not layer.hide_viewport:
            layer.hide_viewport = True
            _hidden_original_strings.append(collection.name)


def _show_original_strings():
    global _hidden_original_strings
    for name in _hidden_original_strings:
        collection = bpy.data.collections.get(name)
        layer = _layer_collection(bpy.context.view_layer.layer_collection, collection) if collection else None
        if layer is not None:
            layer.hide_viewport = False
    _hidden_original_strings = []


def _layer_collection(layer_collection, collection):
    if layer_collection.collection == collection:
        return layer_collection
    for child in layer_collection.children:
        found = _layer_collection(child, collection)
        if found is not None:
            return found
    return None


def string_collections_of(scene, bones_name):
    from .muscle_strings import string_collections

    return [(bones, kind, collection) for bones, kind, collection in string_collections(scene) if bones == bones_name]


def start_free_mode(context):
    """Grab-any-bone mode: no target Empty; the moved vertebra drives the chain."""
    global _session
    scene = context.scene
    stop_session(scene)
    _session = ChainSession(scene, None)
    _hide_original_strings(scene)
    _session.last_movement = float("inf")
    _kick_settle()
    return _session


def chain_ready(scene):
    """Return (ready, reason) for starting a session with the current settings."""
    rigid = scene.biomechanics_target_collection
    if rigid is None or scene.biomechanics_source_collection is None:
        return False, "Choose the Original and Rigid collections"
    if scene.biomechanics_target_collection.name not in muscle_collections(scene, "ligament"):
        return False, "Build Ligaments & Discs first"
    try:
        base = scene.biomechanics_chain_base
        chain_levels("sacrum" if base == FEMUR_BASE else base, scene.biomechanics_chain_top)
    except ValueError as error:
        return False, str(error)
    return True, ""


def fields_missing(scene):
    rigid = scene.biomechanics_target_collection
    if rigid is None:
        return []
    by_base = {base_object_name(obj.name): obj for obj in rigid.objects if obj.type == "MESH"}
    return [name for name in chain_bone_names(scene) if name in by_base and not has_cached_field(by_base[name], scene)]


class BIOMECHANICS_OT_precompute_chain_fields(bpy.types.Operator):
    """Compute and cache the signed distance field of every bone in the chain"""

    bl_idname = "biomechanics.precompute_chain_fields"
    bl_label = "Precompute Collision Fields"

    @classmethod
    def poll(cls, context):
        return context.scene.biomechanics_target_collection is not None

    def execute(self, context):
        scene = context.scene
        rigid = scene.biomechanics_target_collection
        by_base = {base_object_name(obj.name): obj for obj in rigid.objects if obj.type == "MESH"}
        computed = 0
        started = time.perf_counter()
        for name in chain_bone_names(scene):
            obj = by_base.get(name)
            if obj is None:
                continue
            if not has_cached_field(obj, scene):
                computed += 1
            field_for_object(obj, scene, report=lambda text: print(text))
        self.report({"INFO"}, f"{computed} fields computed in {time.perf_counter() - started:.1f} s; the rest were cached")
        return {"FINISHED"}


class BIOMECHANICS_OT_start_chain(bpy.types.Operator):
    """Drag the active Rigid vertebra; ligaments, discs, and collisions move the chain with it"""

    bl_idname = "biomechanics.start_chain"
    bl_label = "Start Chain Interaction"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and chain_ready(context.scene)[0]

    def execute(self, context):
        scene = context.scene
        handle = context.active_object
        rigid = scene.biomechanics_target_collection
        if rigid.objects.get(handle.name) is not handle:
            self.report({"ERROR"}, f"Select a vertebra in {rigid.name}")
            return {"CANCELLED"}
        try:
            chain = start_session(context, handle)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Chain of {len(chain.bodies) - 1} bones with {chain.constraint_count} constraints; move {TARGET_NAME}",
        )
        return {"FINISHED"}


def export_package(scene, directory):
    """Write the active session as a bpy-free chain package."""
    from .muscle_atlas import muscle_strings

    if _session is None:
        raise ValueError("Start Move Bones first")
    group_of = {string["muscle"]: string["group"] for string in muscle_strings()}
    settings = {
        "base": scene.biomechanics_chain_base,
        "top": scene.biomechanics_chain_top,
        "units_per_mm": 1.0 / _units(scene),
        "field_paths": dict(_session.field_paths),
        "group_labels": [group_of.get(muscle, "") for muscle, _side in _session.muscle_groups],
    }
    meta, arrays = package_from_session(_session, settings)
    save_package(directory, meta, arrays)
    return meta


def _force_material(prefix, index, low, high):
    name = f"Biomechanics Force {prefix} {index:02d}"
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    weight = index / (FORCE_BINS - 1)
    color = tuple(l + (h - l) * weight for l, h in zip(low, high)) + (1.0,)
    if any(abs(c - w) > 1e-4 for c, w in zip(material.diffuse_color, color)):
        material.diffuse_color = color
    return material


def apply_force_colors(session):
    """Tint muscle strings red by activation and ligament strings blue by carried force."""
    solver = session.solver
    muscle_materials = [_force_material("Muscle", i, MUSCLE_COLOR_LOW, MUSCLE_COLOR_HIGH) for i in range(FORCE_BINS)]
    ligament_materials = [_force_material("Ligament", i, LIGAMENT_COLOR_LOW, LIGAMENT_COLOR_HIGH) for i in range(FORCE_BINS)]
    forces = {}
    for constraint, lagrange in zip(solver.distance_constraints, solver.c_lagrange):
        forces[constraint.name] = max(forces.get(constraint.name, 0.0), abs(float(lagrange)))
    current_max = max(forces.values(), default=0.0)
    # Running maximum with slow decay keeps the scale steady while bones move.
    session.force_scale = max(current_max, session.force_scale * 0.98)
    scale = session.force_scale if session.force_scale > 1e-12 else 1.0
    for name, force in forces.items():
        obj = cached_object(name)
        if obj is not None:
            _set_string_color(obj, ligament_materials[min(FORCE_BINS - 1, int(round(force / scale * (FORCE_BINS - 1))))])
    activations = solver.activations
    seen = set()
    for muscle in solver.muscles:
        if muscle.name in seen:
            continue
        seen.add(muscle.name)
        obj = cached_object(muscle.name)
        if obj is not None:
            level = float(np.clip(activations[muscle.group], 0.0, 1.0))
            _set_string_color(obj, muscle_materials[int(round(level * (FORCE_BINS - 1)))])


def _restore_force_colors(session):
    if session is None:
        return
    names = {c.name for c in session.solver.distance_constraints} | {m.name for m in session.solver.muscles}
    for name in names:
        obj = cached_object(name)
        if obj is not None:
            _restore_string_color(obj)
    if is_delta_active():
        apply_delta_colors(bpy.context.scene)


def is_force_colors_active():
    return _force_colors


class BIOMECHANICS_OT_toggle_force_colors(bpy.types.Operator):
    """Toggle Force Colors: muscles red by activation, ligaments and discs blue by carried force"""

    bl_idname = "biomechanics.toggle_force_colors"
    bl_label = "Force Colors"

    @classmethod
    def poll(cls, _context):
        return is_active() or _force_colors

    def execute(self, context):
        global _force_colors
        if _force_colors:
            _force_colors = False
            _restore_force_colors(_session)
        else:
            _force_colors = True
            if _session is not None:
                _session.force_scale = 0.0
                apply_force_colors(_session)
        _redraw()
        return {"FINISHED"}


def is_controller_active():
    return _session is not None and _session.controller is not None


FOLLOW_INTERVAL = 1.0
_followed_mtime = None


def _policy_mtime(scene):
    try:
        return os.stat(bpy.path.abspath(scene.biomechanics_posture_policy_path)).st_mtime
    except OSError:
        return None


def replay_current_policy(context):
    """Reset the chain to its start pose, reload the policy, and let it play out."""
    global _session
    scene = context.scene
    if _session is None:
        return
    path = bpy.path.abspath(scene.biomechanics_posture_policy_path)
    _session.restore()
    bpy.context.view_layer.update()
    was_force = _force_colors
    _session = ChainSession(scene, None)
    _session.set_controller(path)
    if was_force:
        _session.force_scale = 0.0
    _session.last_movement = float("inf")
    # Play exactly the episode the trainer scored, then hold.
    _session.replay_budget = _session.policy_steps
    _kick_settle()


def _follow_training():
    """Timer: replay whenever the trainer writes a new best policy."""
    global _followed_mtime
    scene = bpy.context.scene
    if _session is None or _session.controller is None or not scene.biomechanics_follow_training:
        return None
    mtime = _policy_mtime(scene)
    if mtime is not None and mtime != _followed_mtime:
        _followed_mtime = mtime
        try:
            replay_current_policy(bpy.context)
        except (OSError, ValueError, KeyError) as error:
            print(f"Follow Training: {error}")
        _redraw()
    return FOLLOW_INTERVAL


def update_follow_training(scene, _context):
    global _followed_mtime
    if scene.biomechanics_follow_training:
        _followed_mtime = None
        if not bpy.app.timers.is_registered(_follow_training):
            bpy.app.timers.register(_follow_training, first_interval=FOLLOW_INTERVAL)
    elif bpy.app.timers.is_registered(_follow_training):
        bpy.app.timers.unregister(_follow_training)


class BIOMECHANICS_OT_reset_rigid_to_original(bpy.types.Operator):
    """Move every Rigid bone exactly onto its Original twin (neutral pose)"""

    bl_idname = "biomechanics.reset_rigid_to_original"
    bl_label = "Reset Rigid to Original"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.biomechanics_source_collection is not None and scene.biomechanics_target_collection is not None

    def execute(self, context):
        from .alignment import mesh_alignment

        scene = context.scene
        if _session is not None:
            stop_session(scene)
        source = scene.biomechanics_source_collection
        by_base = {base_object_name(obj.name): obj for obj in source.objects if obj.type == "MESH"}
        moved = 0
        skipped = 0
        for obj in scene.biomechanics_target_collection.objects:
            if obj.type != "MESH":
                continue
            twin = by_base.get(base_object_name(obj.name))
            if twin is None or len(twin.data.vertices) != len(obj.data.vertices):
                skipped += 1
                continue
            try:
                obj.matrix_world = mesh_alignment(obj, twin) @ obj.matrix_world
            except ValueError:
                skipped += 1
                continue
            moved += 1
        _redraw()
        self.report({"INFO"}, f"Aligned {moved} Rigid bones onto Original" + (f"; skipped {skipped}" if skipped else ""))
        return {"FINISHED"}


class BIOMECHANICS_OT_toggle_posture_controller(bpy.types.Operator):
    """Toggle the trained posture controller: muscles fire from the chain state every solve"""

    bl_idname = "biomechanics.toggle_posture_controller"
    bl_label = "Posture Controller"

    @classmethod
    def poll(cls, _context):
        return is_active()

    def execute(self, context):
        scene = context.scene
        if _session.controller is not None:
            _session.set_controller(None)
            _session.set_tone(scene.biomechanics_muscle_tone)
        else:
            path = bpy.path.abspath(scene.biomechanics_posture_policy_path)
            try:
                _session.set_controller(path)
            except (OSError, ValueError, KeyError) as error:
                self.report({"ERROR"}, f"Cannot load posture policy: {error}")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Posture controller on (training score {_session.policy_score:.2f})")
            update_follow_training(scene, context)
        update_gravity(scene, context)
        _redraw()
        return {"FINISHED"}


class BIOMECHANICS_OT_export_chain_package(bpy.types.Operator):
    """Save the active chain (bones, ligaments, muscles, collision fields) for headless training"""

    bl_idname = "biomechanics.export_chain_package"
    bl_label = "Export Chain Package"

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    @classmethod
    def poll(cls, _context):
        return is_active()

    def invoke(self, context, _event):
        if not self.directory:
            self.directory = bpy.path.abspath(context.scene.biomechanics_chain_package_dir)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            meta = export_package(context.scene, self.directory)
        except (ValueError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        context.scene.biomechanics_chain_package_dir = self.directory
        self.report(
            {"INFO"},
            f"Exported {len(meta['bodies'])} bodies, {len(meta['constraint_names'])} constraints, "
            f"{len(meta['muscle_names'])} muscle strings to {self.directory}",
        )
        return {"FINISHED"}


class BIOMECHANICS_OT_toggle_move_bones(bpy.types.Operator):
    """Toggle Move Bones: grab any vertebra of the chain with G or R and the chain follows"""

    bl_idname = "biomechanics.toggle_move_bones"
    bl_label = "Move Bones"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_active() or chain_ready(context.scene)[0]

    def execute(self, context):
        scene = context.scene
        if _session is not None and _session.free_mode:
            stop_session(scene)
            _redraw()
            return {"FINISHED"}
        try:
            chain = start_free_mode(context)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Move Bones on: grab any of {len(chain.bodies) - 1} chain bones with G or R")
        return {"FINISHED"}


def is_free_mode():
    return _session is not None and _session.free_mode


class BIOMECHANICS_OT_select_chain_target(bpy.types.Operator):
    """Select the chain target so G / R move it"""

    bl_idname = "biomechanics.select_chain_target"
    bl_label = "Select Target"

    @classmethod
    def poll(cls, context):
        return is_active() and not is_free_mode() and _target_object(context.scene) is not None

    def execute(self, context):
        target = _target_object(context.scene)
        for obj in context.view_layer.objects:
            obj.select_set(False)
        target.hide_set(False)
        target.select_set(True)
        context.view_layer.objects.active = target
        return {"FINISHED"}


class BIOMECHANICS_OT_stop_chain(bpy.types.Operator):
    """Stop chain interaction and keep the current pose"""

    bl_idname = "biomechanics.stop_chain"
    bl_label = "Stop"
    bl_options = {"REGISTER", "UNDO"}

    restore: bpy.props.BoolProperty(name="Restore Start Pose", default=False)

    @classmethod
    def poll(cls, _context):
        return is_active()

    def execute(self, context):
        stop_session(context.scene, restore=self.restore)
        _redraw()
        return {"FINISHED"}


@persistent
def _reset_on_load(_dummy):
    global _session, _force_colors
    _session = None
    _force_colors = False
    if bpy.app.timers.is_registered(_settle):
        bpy.app.timers.unregister(_settle)


def _remove_handlers():
    for handlers, function in (
        (bpy.app.handlers.depsgraph_update_post, chain_depsgraph_update),
        (bpy.app.handlers.load_post, _reset_on_load),
    ):
        for handler in tuple(handlers):
            if getattr(handler, "__name__", "") == function.__name__ and getattr(handler, "__module__", "") == __name__:
                handlers.remove(handler)


def register_chain():
    _remove_handlers()
    bpy.app.handlers.depsgraph_update_post.append(chain_depsgraph_update)
    bpy.app.handlers.load_post.append(_reset_on_load)


def unregister_chain():
    global _session
    if _session is not None:
        # A reload must not leave the bones wherever the last solve put them.
        try:
            _session.restore()
        except ReferenceError:
            pass
    _session = None
    for timer in (_settle, _follow_training):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    _remove_handlers()
