import bpy  # pyright: ignore[reportMissingImports]
import math
from bpy.app.handlers import persistent  # pyright: ignore[reportMissingImports]
from mathutils import Euler, Matrix, Quaternion, Vector  # pyright: ignore[reportMissingImports]

from . import neural_constraints


TARGET_NAME = "Biomechanics Target"
ATTACHED_BONE_MAP = {
    "ilium R": ("femur R",),
}
_is_solving = False
_attached_bone_cache = {}
_neural_pose_cache = {}
_last_solved_matrices = {}


def _matrix_from_values(values):
    return Matrix(tuple(tuple(values[index : index + 4]) for index in range(0, 16, 4)))


def _matrix_values(matrix):
    return tuple(value for row in matrix for value in row)


def neutral_matrix(scene):
    return _matrix_from_values(scene.biomechanics_neutral_matrix)


def set_neutral_matrix(scene, matrix):
    scene.biomechanics_neutral_matrix = _matrix_values(matrix)


def _rigid_matrix(matrix):
    location, rotation, _scale = matrix.decompose()
    return Matrix.LocRotScale(location, rotation, Vector((1.0, 1.0, 1.0)))


def prepare_attached_bones(scene, leader):
    _attached_bone_cache.pop(scene.as_pointer(), None)
    rigid_collection = scene.biomechanics_target_collection
    if rigid_collection is None:
        return 0

    follower_names = ATTACHED_BONE_MAP.get(leader.name, ())
    followers = []
    leader_inverse = _rigid_matrix(leader.matrix_world).inverted_safe()
    for follower_name in follower_names:
        follower = rigid_collection.objects.get(follower_name)
        if follower is None or follower.type != "MESH":
            continue
        followers.append(
            {
                "name": follower.name,
                "relative_matrix": leader_inverse @ follower.matrix_world,
            }
        )

    if followers:
        _attached_bone_cache[scene.as_pointer()] = {
            "leader": leader.name,
            "followers": followers,
        }
    return len(followers)


def clear_attached_bones(scene=None):
    if scene is None:
        _attached_bone_cache.clear()
    else:
        _attached_bone_cache.pop(scene.as_pointer(), None)


def clear_interaction_state(scene=None):
    if scene is None:
        _last_solved_matrices.clear()
        _neural_pose_cache.clear()
    else:
        scene_key = scene.as_pointer()
        _last_solved_matrices.pop(scene_key, None)
        _neural_pose_cache.pop(scene_key, None)


def _move_attached_bones(scene, leader, leader_matrix):
    if not scene.biomechanics_move_attached_bones:
        return
    cache = _attached_bone_cache.get(scene.as_pointer())
    if cache is None or cache["leader"] != leader.name:
        return
    for follower_data in cache["followers"]:
        follower = bpy.data.objects.get(follower_data["name"])
        if follower is not None:
            follower.matrix_world = _rigid_matrix(leader_matrix) @ follower_data["relative_matrix"]


def _matrices_match(first, second, tolerance=1e-5):
    return all(
        abs(first[row][column] - second[row][column]) <= tolerance
        for row in range(4)
        for column in range(4)
    )


def _pivot_rotated_matrix(neutral, rotation, rotation_center):
    """Return the neutral moving matrix rotated around the trained world-space pivot."""
    _neutral_location, neutral_rotation, _neutral_scale = neutral.decompose()
    world_rotation = (
        neutral_rotation.to_matrix().to_4x4()
        @ rotation.to_matrix().to_4x4()
        @ neutral_rotation.inverted().to_matrix().to_4x4()
    )
    return (
        Matrix.Translation(rotation_center)
        @ world_rotation
        @ Matrix.Translation(-rotation_center)
        @ neutral
    )


def _neural_constrained_matrix(scene, manipulated, target_matrix):
    registry = getattr(scene, "biomechanics_neural_models", ())
    model_entries = [
        entry for entry in registry if entry.moving_object_name == manipulated.name
    ]
    legacy_manifest_path = None
    if not model_entries and scene.biomechanics_neural_model_path:
        # Compatibility with scenes saved before the per-pair registry was introduced.
        legacy_manifest_path = scene.biomechanics_neural_model_path
    if not model_entries and legacy_manifest_path is None:
        if registry:
            scene.biomechanics_neural_model_status = (
                f"No trained neural model moves {manipulated.name}; using scalar limits"
            )
            return None
        return None

    rigid_collection = scene.biomechanics_target_collection
    last_problem = None
    candidates = [(entry.manifest_path, entry.fixed_object_name) for entry in model_entries]
    if legacy_manifest_path is not None:
        candidates.append((legacy_manifest_path, None))
    for manifest_path, fixed_name in candidates:
        try:
            if fixed_name is None:
                manifest = neural_constraints.load_manifest(bpy.path.abspath(manifest_path))
                fixed_name = manifest["fixed_object"]["name"]
            fixed_object = (
                rigid_collection.objects.get(fixed_name)
                if rigid_collection is not None
                else None
            )
            if fixed_object is None or fixed_object.type != "MESH":
                last_problem = f"fixed bone {fixed_name} is unavailable in Rigid"
                continue
            manifest, onnx_path, reasons = neural_constraints.load_cached_model(
                bpy.context,
                bpy.path.abspath(manifest_path),
                fixed_object,
                manipulated,
            )
        except neural_constraints.ModelLoadError as exc:
            last_problem = str(exc)
            continue
        if reasons:
            last_problem = "; ".join(reasons)
            continue
        break
    else:
        if last_problem:
            scene.biomechanics_neural_model_status = (
                "Neural model unavailable; using scalar limits: " + last_problem
            )
        return None

    cache_key = scene.as_pointer()
    cache = _neural_pose_cache.get(cache_key)
    if (
        cache is None
        or cache["path"] != str(onnx_path)
        or cache["session_generation"] != neural_constraints.session_cache_generation()
    ):
        cache = {
            "path": str(onnx_path),
            "manifest": manifest,
            "session": neural_constraints.get_session(onnx_path),
            "accepted_pose": (0.0,) * 6,
            "session_generation": neural_constraints.session_cache_generation(),
        }
        _neural_pose_cache[cache_key] = cache

    neutral = _matrix_from_values(manifest["neutral_moving_matrix"])
    target_rotation = neutral.to_quaternion().inverted() @ target_matrix.to_quaternion()
    target_rotation.normalize()
    _neutral_location, neutral_rotation, neutral_scale = neutral.decompose()
    rotation_center = Vector(manifest["rotation_center_world"])
    rotation_only_matrix = _pivot_rotated_matrix(neutral, target_rotation, rotation_center)
    true_world_translation = target_matrix.translation - rotation_only_matrix.translation
    local_translation = neutral_rotation.inverted() @ true_world_translation
    millimeters_per_blender_unit = (scene.unit_settings.scale_length or 1.0) * 1000.0
    target_pose = tuple(
        local_translation[index]
        * (neutral_scale[index] if manifest["pose_transform_version"] == 1 else 1.0)
        * millimeters_per_blender_unit
        for index in range(3)
    ) + tuple(math.degrees(value) for value in target_rotation.to_euler("XYZ"))

    clamped_target_pose = neural_constraints.clamp_pose_to_range(target_pose, manifest)
    target_was_clamped = clamped_target_pose != target_pose
    target_pose = clamped_target_pose

    try:
        projected_pose, prediction = neural_constraints.project_pose(
            cache["session"],
            manifest,
            cache["accepted_pose"],
            target_pose,
        )
    except neural_constraints.ModelLoadError as exc:
        scene.biomechanics_neural_model_status = (
            f"Neural projection unavailable; using scalar limits: {exc}"
        )
        return None

    cache["accepted_pose"] = projected_pose
    projection_text = "projected" if target_was_clamped or projected_pose != target_pose else "accepted"
    scene.biomechanics_neural_model_status = (
        f"Neural {projection_text}: {prediction['validity_probability']:.3f} valid, "
        f"clearance {prediction['clearance_mm']:.2f} mm"
    )
    translation = Vector(projected_pose[:3]) / millimeters_per_blender_unit
    rotation = Euler(tuple(math.radians(value) for value in projected_pose[3:]), "XYZ").to_quaternion()
    local_translation = Vector(
        tuple(
            translation[index] / neutral_scale[index]
            if manifest["pose_transform_version"] == 1
            else translation[index]
            for index in range(3)
        )
    )
    return (
        Matrix.Translation(neutral_rotation @ local_translation)
        @ _pivot_rotated_matrix(neutral, rotation, rotation_center)
    )


def constrained_matrix(scene, target_matrix, manipulated=None):
    if manipulated is not None:
        neural_matrix = _neural_constrained_matrix(scene, manipulated, target_matrix)
        if neural_matrix is not None:
            return neural_matrix

    neutral = neutral_matrix(scene)
    neutral_location, neutral_rotation, neutral_scale = neutral.decompose()
    target_location, target_rotation, _target_scale = target_matrix.decompose()

    local_translation = neutral_rotation.inverted() @ (target_location - neutral_location)
    scale_length = scene.unit_settings.scale_length or 1.0
    translation_limit = scene.biomechanics_translation_limit_mm / (1000.0 * scale_length)
    local_translation = Vector(
        tuple(max(-translation_limit, min(translation_limit, value)) for value in local_translation)
    )

    local_rotation = neutral_rotation.inverted() @ target_rotation
    local_rotation.normalize()
    angle = local_rotation.angle
    rotation_limit = scene.biomechanics_rotation_limit
    if angle > rotation_limit and angle > 0.0:
        local_rotation = Quaternion(local_rotation.axis, rotation_limit)

    solved_location = neutral_location + neutral_rotation @ local_translation
    solved_rotation = neutral_rotation @ local_rotation
    return Matrix.LocRotScale(solved_location, solved_rotation, neutral_scale)


def solve_scene(scene):
    global _is_solving

    if _is_solving or not scene.biomechanics_live_solve:
        return
    if scene.biomechanics_neural_training_active:
        return
    target = scene.biomechanics_pose_target
    manipulated = scene.biomechanics_manipulated_object
    if target is None or manipulated is None:
        return

    _is_solving = True
    try:
        scene_key = scene.as_pointer()
        last_solved = _last_solved_matrices.get(scene_key)
        direct_mesh_edit = (
            bpy.context.active_object is manipulated
            and (last_solved is None or not _matrices_match(manipulated.matrix_world, last_solved))
        )
        if direct_mesh_edit:
            target.matrix_world = manipulated.matrix_world
        requested_matrix = target.matrix_world
        solved_matrix = constrained_matrix(scene, requested_matrix, manipulated)
        if not _matrices_match(manipulated.matrix_world, solved_matrix):
            manipulated.matrix_world = solved_matrix
        _last_solved_matrices[scene_key] = solved_matrix.copy()
        _move_attached_bones(scene, manipulated, solved_matrix)
    finally:
        _is_solving = False


def remove_pose_targets(scene=None):
    for obj in list(bpy.data.objects):
        if not obj.get("biomechanics_pose_target"):
            continue
        if scene is not None and scene not in obj.users_scene:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)


@persistent
def solve_interaction(_scene, _depsgraph):
    updated_ids = {
        getattr(update.id, "original", None) or update.id
        for update in _depsgraph.updates
    }
    for scene in bpy.data.scenes:
        neural_constraints.invalidate_model_cache(scene, _depsgraph)
        target = scene.biomechanics_pose_target
        manipulated = scene.biomechanics_manipulated_object
        if target is None or manipulated is None:
            continue
        if target not in updated_ids and manipulated not in updated_ids:
            continue
        solve_scene(scene)


def register_interaction():
    for handler in tuple(bpy.app.handlers.depsgraph_update_post):
        if (
            getattr(handler, "__name__", None) == solve_interaction.__name__
            and getattr(handler, "__module__", None) == solve_interaction.__module__
        ):
            bpy.app.handlers.depsgraph_update_post.remove(handler)
    bpy.app.handlers.depsgraph_update_post.append(solve_interaction)
    for scene in bpy.data.scenes:
        if scene.biomechanics_training_moving_object is not None:
            prepare_attached_bones(scene, scene.biomechanics_training_moving_object)


def unregister_interaction():
    for handler in tuple(bpy.app.handlers.depsgraph_update_post):
        if (
            getattr(handler, "__name__", None) == solve_interaction.__name__
            and getattr(handler, "__module__", None) == solve_interaction.__module__
        ):
            bpy.app.handlers.depsgraph_update_post.remove(handler)
    clear_attached_bones()
    clear_interaction_state()
    remove_pose_targets()