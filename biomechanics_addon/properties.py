import bpy
import json
import math

from .alignment import base_object_name


PREVIOUS_ALPHA_KEY = "_biomechanics_previous_alpha"
PREVIOUS_TRANSPARENCY_KEY = "_biomechanics_previous_transparency"
PREVIOUS_IN_FRONT_KEY = "_biomechanics_previous_in_front"
PREVIOUS_DISPLAY_TYPE_KEY = "_biomechanics_previous_display_type"
_previous_viewport_color_modes = {}


def _view_3d_spaces():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    yield space


def _restore_viewport_color_modes():
    for space in _view_3d_spaces():
        previous_mode = _previous_viewport_color_modes.pop(space.as_pointer(), None)
        if previous_mode is not None:
            space.shading.color_type = previous_mode
    _previous_viewport_color_modes.clear()


def _restore_transparency_overlay():
    for obj in bpy.data.objects:
        if PREVIOUS_ALPHA_KEY not in obj:
            continue

        color = list(obj.color)
        color[3] = obj[PREVIOUS_ALPHA_KEY]
        obj.color = color
        obj.show_transparent = bool(obj[PREVIOUS_TRANSPARENCY_KEY])
        obj.show_in_front = bool(obj[PREVIOUS_IN_FRONT_KEY])
        obj.display_type = obj[PREVIOUS_DISPLAY_TYPE_KEY]
        del obj[PREVIOUS_ALPHA_KEY]
        del obj[PREVIOUS_TRANSPARENCY_KEY]
        del obj[PREVIOUS_IN_FRONT_KEY]
        del obj[PREVIOUS_DISPLAY_TYPE_KEY]

    _restore_viewport_color_modes()


def refresh_transparency_overlay(scene, excluded_pair_name=None):
    _restore_transparency_overlay()

    mode = scene.biomechanics_transparency_mode
    if mode == "NONE":
        return
    if mode == "ORIGINAL":
        collection = scene.biomechanics_source_collection
    else:
        collection = scene.biomechanics_target_collection
    if collection is None:
        return

    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        if base_object_name(obj.name) == excluded_pair_name:
            continue

        obj[PREVIOUS_ALPHA_KEY] = obj.color[3]
        obj[PREVIOUS_TRANSPARENCY_KEY] = obj.show_transparent
        obj[PREVIOUS_IN_FRONT_KEY] = obj.show_in_front
        obj[PREVIOUS_DISPLAY_TYPE_KEY] = obj.display_type
        if scene.biomechanics_overlay_wireframe:
            obj.display_type = "WIRE"
        else:
            color = list(obj.color)
            color[3] = scene.biomechanics_overlay_opacity
            obj.color = color
            obj.show_transparent = True
        if scene.biomechanics_overlay_on_top:
            obj.show_in_front = True

    if scene.biomechanics_overlay_wireframe:
        return

    for space in _view_3d_spaces():
        if space.shading.type != "SOLID":
            continue
        _previous_viewport_color_modes[space.as_pointer()] = space.shading.color_type
        space.shading.color_type = "OBJECT"


def _update_transparency_overlay(scene, context):
    selected_object = context.active_object if context is not None else bpy.context.active_object
    source_collection = scene.biomechanics_source_collection
    excluded_pair_name = None
    if (
        selected_object is not None
        and selected_object.type == "MESH"
        and source_collection is not None
        and source_collection.objects.get(selected_object.name) is selected_object
    ):
        excluded_pair_name = base_object_name(selected_object.name)

    refresh_transparency_overlay(scene, excluded_pair_name)


def _update_training_result(scene, _context):
    text = scene.biomechanics_training_result_text
    rigid_collection = scene.biomechanics_target_collection
    if text is None or rigid_collection is None:
        return
    try:
        result = json.loads(text.as_string())
    except (TypeError, ValueError, json.JSONDecodeError):
        scene.biomechanics_training_summary = "Selected training data is invalid"
        return

    fixed_object = rigid_collection.objects.get(result.get("fixed_object", ""))
    moving_object = rigid_collection.objects.get(result.get("moving_object", ""))
    if fixed_object is not None and fixed_object.type == "MESH":
        scene.biomechanics_training_fixed_object = fixed_object
    if moving_object is not None and moving_object.type == "MESH":
        scene.biomechanics_training_moving_object = moving_object
    summary = result.get("summary", {})
    if "valid_samples" in summary and "total_samples" in summary:
        scene.biomechanics_training_summary = (
            f'{summary["valid_samples"]} valid / {summary["total_samples"]} samples'
        )


def register_properties():
    bpy.types.Scene.biomechanics_training_fixed_object = bpy.props.PointerProperty(
        name="Fixed Bone",
        description="Rigid mesh that remains fixed during geometric training",
        type=bpy.types.Object,
    )
    bpy.types.Scene.biomechanics_training_moving_object = bpy.props.PointerProperty(
        name="Moving Bone",
        description="Rigid mesh sampled around its neutral pose",
        type=bpy.types.Object,
    )
    bpy.types.Scene.biomechanics_training_translation_mm = bpy.props.FloatProperty(
        name="Translation Range (mm)",
        description="Sample from minus to plus this local translation on each axis",
        min=0.0,
        default=3.0,
        precision=2,
    )
    bpy.types.Scene.biomechanics_training_rotation = bpy.props.FloatProperty(
        name="Rotation Range",
        description="Sample from minus to plus this local rotation on each axis",
        subtype="ANGLE",
        min=0.0,
        max=math.pi,
        default=math.radians(3.0),
    )
    bpy.types.Scene.biomechanics_training_translation_steps = bpy.props.IntProperty(
        name="Translation Steps",
        description="Samples per translation axis",
        min=1,
        max=7,
        default=3,
    )
    bpy.types.Scene.biomechanics_training_rotation_steps = bpy.props.IntProperty(
        name="Rotation Steps",
        description="Samples per rotation axis",
        min=1,
        max=7,
        default=3,
    )
    bpy.types.Scene.biomechanics_training_clearance_mm = bpy.props.FloatProperty(
        name="Minimum Clearance (mm)",
        description="Reject non-colliding poses closer than this surface clearance",
        min=0.0,
        default=0.0,
        precision=3,
    )
    bpy.types.Scene.biomechanics_training_proxy_faces = bpy.props.IntProperty(
        name="Proxy Faces",
        description="Maximum faces per temporary collision proxy",
        min=250,
        max=20000,
        default=2000,
    )
    bpy.types.Scene.biomechanics_training_maximum_samples = bpy.props.IntProperty(
        name="Maximum Samples",
        description="Reject training workloads above this hard sample limit",
        min=1,
        max=1000000,
        default=20000,
    )
    bpy.types.Scene.biomechanics_training_seed = bpy.props.IntProperty(
        name="Sampling Seed",
        description="Deterministic seed recorded in the exported package for background dataset sampling",
        min=0,
        default=12345,
    )
    bpy.types.Scene.biomechanics_training_export_directory = bpy.props.StringProperty(
        name="Training Folder",
        description="Folder for terminal training inputs and results",
        subtype="DIR_PATH",
        default="//biomechanics_training",
    )
    bpy.types.Scene.biomechanics_training_result_text = bpy.props.PointerProperty(
        name="Training Data",
        type=bpy.types.Text,
        update=_update_training_result,
    )
    bpy.types.Scene.biomechanics_training_summary = bpy.props.StringProperty(
        name="Training Summary",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_onnx_status = bpy.props.StringProperty(
        name="Neural Model Status",
        description="Result of the last ONNX Runtime compatibility check",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_model_path = bpy.props.StringProperty(
        name="Neural Model",
        description="Path to the active model_manifest.json bundle, if any",
        subtype="FILE_PATH",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_model_status = bpy.props.StringProperty(
        name="Neural Model Import Status",
        description="Result of the last neural model import/compatibility check",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_manipulated_object = bpy.props.PointerProperty(
        name="Manipulated Bone",
        type=bpy.types.Object,
    )
    bpy.types.Scene.biomechanics_pose_target = bpy.props.PointerProperty(
        name="Pose Target",
        type=bpy.types.Object,
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neutral_matrix = bpy.props.FloatVectorProperty(
        name="Neutral Matrix",
        size=16,
        default=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_translation_limit_mm = bpy.props.FloatProperty(
        name="Translation Limit (mm)",
        description="Maximum local translation from the neutral pose on each axis, in millimeters",
        min=0.0,
        default=5.0,
        precision=2,
    )
    bpy.types.Scene.biomechanics_rotation_limit = bpy.props.FloatProperty(
        name="Rotation Limit",
        description="Maximum total rotation from the neutral pose",
        subtype="ANGLE",
        min=0.0,
        max=math.pi,
        default=math.radians(5.0),
    )
    bpy.types.Scene.biomechanics_live_solve = bpy.props.BoolProperty(
        name="Live Solve",
        description="Apply the constrained target pose while the target is moved",
        default=True,
    )
    bpy.types.Scene.biomechanics_move_attached_bones = bpy.props.BoolProperty(
        name="Move Attached Bones",
        description="Move explicitly mapped follower bones with the constrained leader",
        default=True,
    )
    bpy.types.Scene.biomechanics_source_collection = bpy.props.PointerProperty(
        name="Original Collection",
        type=bpy.types.Collection,
        update=_update_transparency_overlay,
    )
    bpy.types.Scene.biomechanics_target_collection = bpy.props.PointerProperty(
        name="Rigid Collection",
        type=bpy.types.Collection,
        update=_update_transparency_overlay,
    )
    bpy.types.Scene.biomechanics_transparency_mode = bpy.props.EnumProperty(
        name="Comparison Collection",
        description="Choose which collection uses the comparison display",
        items=(
            ("NONE", "None", "Show both collections normally"),
            ("ORIGINAL", "Original", "Apply comparison display to the original collection"),
            ("RIGID", "Rigid", "Apply comparison display to the rigid collection"),
        ),
        default="NONE",
        update=_update_transparency_overlay,
    )
    bpy.types.Scene.biomechanics_overlay_opacity = bpy.props.FloatProperty(
        name="Overlay Opacity",
        description="Set the opacity of the transparent collection overlay",
        subtype="FACTOR",
        min=0.0,
        max=1.0,
        default=0.25,
        update=_update_transparency_overlay,
    )
    bpy.types.Scene.biomechanics_overlay_wireframe = bpy.props.BoolProperty(
        name="Mesh Wireframe",
        description="Show the comparison collection as mesh wireframe instead of transparent",
        default=False,
        update=_update_transparency_overlay,
    )
    bpy.types.Scene.biomechanics_overlay_on_top = bpy.props.BoolProperty(
        name="Overlay On Top",
        description="Display the transparent collection in front of other objects",
        default=True,
        update=_update_transparency_overlay,
    )

    for scene in bpy.data.scenes:
        if "biomechanics_rigid_transparent" in scene:
            del scene["biomechanics_rigid_transparent"]
        if scene.biomechanics_source_collection is None:
            scene.biomechanics_source_collection = bpy.data.collections.get("Collection")
        if scene.biomechanics_target_collection is None:
            scene.biomechanics_target_collection = bpy.data.collections.get("rigid")
        rigid_collection = scene.biomechanics_target_collection
        if rigid_collection is not None:
            result_text = scene.biomechanics_training_result_text
            if result_text is not None:
                try:
                    training_result = json.loads(result_text.as_string())
                except (TypeError, ValueError, json.JSONDecodeError):
                    training_result = {}
                saved_fixed = rigid_collection.objects.get(
                    training_result.get("fixed_object", "")
                )
                saved_moving = rigid_collection.objects.get(
                    training_result.get("moving_object", "")
                )
                if saved_fixed is not None and saved_fixed.type == "MESH":
                    scene.biomechanics_training_fixed_object = saved_fixed
                if saved_moving is not None and saved_moving.type == "MESH":
                    scene.biomechanics_training_moving_object = saved_moving
            else:
                scene.biomechanics_training_moving_object = None
            if scene.biomechanics_training_fixed_object is None:
                scene.biomechanics_training_fixed_object = next(
                    (
                        obj
                        for obj in rigid_collection.objects
                        if obj.type == "MESH" and base_object_name(obj.name) == "sacrum"
                    ),
                    None,
                )
        _update_transparency_overlay(scene, None)


def unregister_properties():
    _restore_transparency_overlay()
    del bpy.types.Scene.biomechanics_neural_model_status
    del bpy.types.Scene.biomechanics_neural_model_path
    del bpy.types.Scene.biomechanics_onnx_status
    del bpy.types.Scene.biomechanics_overlay_on_top
    del bpy.types.Scene.biomechanics_overlay_wireframe
    del bpy.types.Scene.biomechanics_overlay_opacity
    del bpy.types.Scene.biomechanics_transparency_mode
    del bpy.types.Scene.biomechanics_target_collection
    del bpy.types.Scene.biomechanics_source_collection
    del bpy.types.Scene.biomechanics_move_attached_bones
    del bpy.types.Scene.biomechanics_live_solve
    del bpy.types.Scene.biomechanics_rotation_limit
    del bpy.types.Scene.biomechanics_translation_limit_mm
    del bpy.types.Scene.biomechanics_neutral_matrix
    del bpy.types.Scene.biomechanics_pose_target
    del bpy.types.Scene.biomechanics_manipulated_object
    del bpy.types.Scene.biomechanics_training_summary
    del bpy.types.Scene.biomechanics_training_result_text
    del bpy.types.Scene.biomechanics_training_export_directory
    del bpy.types.Scene.biomechanics_training_seed
    del bpy.types.Scene.biomechanics_training_maximum_samples
    del bpy.types.Scene.biomechanics_training_proxy_faces
    del bpy.types.Scene.biomechanics_training_clearance_mm
    del bpy.types.Scene.biomechanics_training_rotation_steps
    del bpy.types.Scene.biomechanics_training_translation_steps
    del bpy.types.Scene.biomechanics_training_rotation
    del bpy.types.Scene.biomechanics_training_translation_mm
    del bpy.types.Scene.biomechanics_training_moving_object
    del bpy.types.Scene.biomechanics_training_fixed_object