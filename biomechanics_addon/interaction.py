import bpy
import json
import math
from bpy.app.handlers import persistent
from mathutils import Euler, Matrix, Quaternion, Vector


TARGET_NAME = "Biomechanics Target"
POINT_TARGET_NAME = "Biomechanics Point Target"
ATTACHED_BONE_MAP = {
    "ilium R": ("femur R",),
}
_is_solving = False
_trained_pose_cache = {}
_attached_bone_cache = {}


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


def prepare_trained_constraints(scene, manipulated):
    clear_trained_constraints(scene)
    text = scene.biomechanics_training_result_text
    if text is None:
        return False, "No training dataset selected; using scalar limits"

    try:
        result = json.loads(text.as_string())
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "Training dataset is not valid JSON; using scalar limits"

    if result.get("format_version") != 1:
        return False, "Unsupported training format; using scalar limits"
    if result.get("moving_object") != manipulated.name:
        return False, "Training dataset belongs to a different moving bone"

    trained_neutral = _matrix_from_values(result.get("neutral_moving_matrix", ()))
    if not _matrices_match(trained_neutral, manipulated.matrix_world):
        return False, "Moving bone changed since training; using scalar limits"
    fixed_object = bpy.data.objects.get(result.get("fixed_object", ""))
    if fixed_object is None:
        return False, "Trained fixed bone is missing; using scalar limits"
    trained_fixed = _matrix_from_values(result.get("neutral_fixed_matrix", ()))
    if not _matrices_match(trained_fixed, fixed_object.matrix_world):
        return False, "Fixed bone changed since training; using scalar limits"

    scale_length = scene.unit_settings.scale_length or 1.0
    millimeters_per_blender_unit = scale_length * 1000.0
    candidates = []
    for sample in result.get("samples", ()):
        if not sample.get("valid"):
            continue
        translation_mm = sample["translation_mm"]
        rotation_degrees = sample["rotation_degrees"]
        translation = Vector(translation_mm) / millimeters_per_blender_unit
        rotation = Euler(
            tuple(math.radians(value) for value in rotation_degrees),
            "XYZ",
        ).to_quaternion()
        local_matrix = Matrix.LocRotScale(translation, rotation, Vector((1.0, 1.0, 1.0)))
        candidates.append((local_matrix, translation, rotation))

    if not candidates:
        return False, "Training dataset contains no valid poses"
    if len(candidates) == 1:
        return False, "Training dataset contains only one valid pose; retrain with nonzero ranges"

    _trained_pose_cache[scene.as_pointer()] = {
        "neutral": trained_neutral,
        "candidates": candidates,
        "millimeters_per_blender_unit": millimeters_per_blender_unit,
        "moving_object": manipulated.name,
    }
    prepare_attached_bones(scene, manipulated)
    return True, f"Using {len(candidates):,} trained valid poses"


def clear_trained_constraints(scene=None):
    if scene is None:
        _trained_pose_cache.clear()
    else:
        _trained_pose_cache.pop(scene.as_pointer(), None)


def _trained_constrained_matrix(scene, manipulated, target_matrix):
    cache = _trained_pose_cache.get(scene.as_pointer())
    if cache is None or cache["moving_object"] != manipulated.name:
        return None

    neutral = cache["neutral"]
    target_local = neutral.inverted_safe() @ target_matrix
    target_translation, target_rotation, _target_scale = target_local.decompose()
    millimeters_per_blender_unit = cache["millimeters_per_blender_unit"]
    joint_radius_mm = 50.0

    def pose_distance(candidate):
        _matrix, translation, rotation = candidate
        translation_distance_mm = (
            (translation - target_translation).length * millimeters_per_blender_unit
        )
        rotation_distance_mm = rotation.rotation_difference(target_rotation).angle * joint_radius_mm
        return translation_distance_mm ** 2 + rotation_distance_mm ** 2

    local_matrix, _translation, _rotation = min(cache["candidates"], key=pose_distance)
    return neutral @ local_matrix


def _trained_point_constrained_matrix(scene, manipulated, target_location, local_point):
    cache = _trained_pose_cache.get(scene.as_pointer())
    if cache is None or cache["moving_object"] != manipulated.name:
        return None

    neutral = cache["neutral"]

    def point_distance(candidate):
        local_matrix, _translation, _rotation = candidate
        candidate_point = (neutral @ local_matrix) @ local_point
        return (candidate_point - target_location).length_squared

    local_matrix, _translation, _rotation = min(cache["candidates"], key=point_distance)
    return neutral @ local_matrix


def constrained_matrix(scene, target_matrix, manipulated=None):
    trained_matrix = None
    if manipulated is not None:
        trained_matrix = _trained_constrained_matrix(scene, manipulated, target_matrix)
    if trained_matrix is not None:
        return trained_matrix

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
    target = scene.biomechanics_pose_target
    manipulated = scene.biomechanics_manipulated_object
    cache = _trained_pose_cache.get(scene.as_pointer())
    if manipulated is None and cache is not None:
        manipulated = bpy.data.objects.get(cache["moving_object"])
    if manipulated is None:
        return

    _is_solving = True
    try:
        if target is not None and target.get("biomechanics_point_target"):
            local_point = Vector(target["biomechanics_local_point"])
            solved_matrix = _trained_point_constrained_matrix(
                scene,
                manipulated,
                target.matrix_world.translation,
                local_point,
            )
            if solved_matrix is None:
                return
        else:
            requested_matrix = target.matrix_world if target is not None else manipulated.matrix_world
            solved_matrix = constrained_matrix(scene, requested_matrix, manipulated)
        if not _matrices_match(manipulated.matrix_world, solved_matrix):
            manipulated.matrix_world = solved_matrix
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
    for scene in bpy.data.scenes:
        solve_scene(scene)


def register_interaction():
    if solve_interaction not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(solve_interaction)
    for scene in bpy.data.scenes:
        moving_object = scene.biomechanics_training_moving_object
        if moving_object is not None and scene.biomechanics_training_result_text is not None:
            prepare_trained_constraints(scene, moving_object)


def unregister_interaction():
    if solve_interaction in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(solve_interaction)
    clear_trained_constraints()
    clear_attached_bones()
    remove_pose_targets()