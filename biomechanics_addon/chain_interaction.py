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

import time

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np
from bpy.app.handlers import persistent  # pyright: ignore[reportMissingImports]
from mathutils import Matrix  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, local_vertices
from .bone_fields import field_for_object, has_cached_field, surface_samples
from .chain_solver import Body, ChainSolver, CollisionPair, DistanceConstraint, PoseConstraint, matrix_to_quat
from .muscle_atlas import SPINE
from .muscle_strings import muscle_collections, string_info

TARGET_NAME = "Biomechanics Chain Target"
TARGET_PROPERTY = "biomechanics_chain_target"
SETTLE_INTERVAL = 0.03
SETTLE_THRESHOLD = 3e-5  # Blender units (0.03 mm) of maximum body movement per solve
MAX_SETTLE_SOLVES = 150

_session = None
_solving = False


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
        self.last_solve_ms = 0.0
        self.settle_count = 0
        self.last_movement = 0.0
        self.written = {}  # object name -> matrix the solver last wrote
        self._build(scene, handle)

    def _build(self, scene, handle):
        rigid = scene.biomechanics_target_collection
        by_base = {}
        for obj in rigid.objects:
            if obj.type == "MESH":
                by_base.setdefault(base_object_name(obj.name), obj)
        base = scene.biomechanics_chain_base
        top = scene.biomechanics_chain_top
        if base not in by_base:
            raise ValueError(f"Base bone {base} is not in {rigid.name}")
        chain = chain_levels(base, top)
        if handle is not None and base_object_name(handle.name) not in chain:
            raise ValueError(f"{handle.name} is not between {base} and {top}")
        missing = [name for name in chain if name not in by_base]
        if missing:
            raise ValueError(f"Missing chain bones: {', '.join(missing)}")

        bones = {}
        pinned = {base}
        if scene.biomechanics_chain_pin_top and (handle is None or chain[-1] != base_object_name(handle.name)):
            pinned.add(chain[-1])
            self.pinned_top = chain[-1]
        for name in [base] + chain:
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
        pairs = []
        margin = scene.biomechanics_chain_margin_mm / _units(scene)
        ordered = [base] + chain
        for lower, upper in zip(ordered, ordered[1:]):
            pairs.append(self._collision_pair(scene, by_base, lower, upper, margin))

        self.solver = ChainSolver(
            bones.values(),
            constraints,
            pairs,
            [],
            iterations=scene.biomechanics_chain_iterations,
            substeps=scene.biomechanics_chain_substeps,
            gravity=_gravity_vector(scene),
        )
        self.constraint_count = len(constraints)
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
        self.solver.pose_constraints = [self.pose]
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
            names = [base_object_name(bone) for bone, _index in info["attachments"]]
            if any(name not in self.bodies for name in names):
                continue
            rest_info = source_by_key.get((info["muscle"], info["side"], info["index"]))
            if rest_info is None:
                continue
            rest_points = [_world_vertex(bone, index) for bone, index in rest_info["attachments"]]
            if any(point is None for point in rest_points):
                continue
            for segment in range(len(names) - 1):
                name_a, name_b = names[segment], names[segment + 1]
                if name_a == name_b:
                    continue
                index_a = info["attachments"][segment][1]
                index_b = info["attachments"][segment + 1][1]
                local_a = _local_offset(by_base[name_a], index_a, self.centers[name_a])
                local_b = _local_offset(by_base[name_b], index_b, self.centers[name_b])
                rest = float(np.linalg.norm(rest_points[segment + 1] - rest_points[segment]))
                constraints.append(
                    DistanceConstraint(
                        self.bodies[name_a],
                        local_a,
                        self.bodies[name_b],
                        local_b,
                        rest,
                        compliance=info.get("compliance", 0.5),
                        unilateral=info.get("unilateral", True),
                        slack=info.get("slack", 0.0),
                        name=obj.name,
                    )
                )
        return constraints

    def _collision_pair(self, scene, by_base, name_a, name_b, margin):
        obj_a, obj_b = by_base[name_a], by_base[name_b]
        field_a = field_for_object(obj_a, scene)
        field_b = field_for_object(obj_b, scene)
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

    def _collect_followers(self, scene, rigid, chain):
        if not scene.biomechanics_chain_move_ribs:
            return
        for obj in rigid.objects:
            if obj.type != "MESH":
                continue
            base = base_object_name(obj.name)
            if " rib " not in f" {base} ":
                continue
            vertebra = base.split(" rib ")[0]
            if vertebra not in chain or vertebra not in self.objects:
                continue
            leader = bpy.data.objects[self.objects[vertebra]]
            self.followers.append((obj.name, vertebra, leader.matrix_world.inverted() @ obj.matrix_world))
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
        self.solver.solve()
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
        for follower_name, vertebra, relative in self.followers:
            follower = bpy.data.objects.get(follower_name)
            leader = bpy.data.objects.get(self.objects[vertebra])
            if follower is not None and leader is not None:
                follower.matrix_world = leader.matrix_world @ relative
        self.last_solve_ms = (time.perf_counter() - started) * 1000.0

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
        }


class _CenteredField:
    """A DistanceField queried in body-centered coordinates."""

    def __init__(self, field, center):
        self.field = field
        self.center = np.asarray(center, dtype=np.float64)

    def sample(self, points):
        return self.field.sample(np.asarray(points) + self.center)

    def sample_with_gradient(self, points):
        return self.field.sample_with_gradient(np.asarray(points) + self.center)


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
    try:
        return [scene.biomechanics_chain_base] + chain_levels(scene.biomechanics_chain_base, scene.biomechanics_chain_top)
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
    _session = None
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
        _session.solve_and_apply(scene)
    finally:
        _solving = False


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


def update_gravity(scene, _context):
    if _session is None:
        return
    _session.settle_count = 0
    _session.last_movement = float("inf")
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE_INTERVAL)


def start_free_mode(context):
    """Grab-any-bone mode: no target Empty; the moved vertebra drives the chain."""
    global _session
    scene = context.scene
    stop_session(scene)
    _session = ChainSession(scene, None)
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
        chain_levels(scene.biomechanics_chain_base, scene.biomechanics_chain_top)
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
    global _session
    _session = None
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
    _session = None
    if bpy.app.timers.is_registered(_settle):
        bpy.app.timers.unregister(_settle)
    _remove_handlers()
