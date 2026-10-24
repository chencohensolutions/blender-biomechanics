import bpy  # pyright: ignore[reportMissingImports]
import math
from .alignment import base_object_name


PREVIOUS_ALPHA_KEY = "_biomechanics_previous_alpha"
PREVIOUS_TRANSPARENCY_KEY = "_biomechanics_previous_transparency"
PREVIOUS_IN_FRONT_KEY = "_biomechanics_previous_in_front"
PREVIOUS_DISPLAY_TYPE_KEY = "_biomechanics_previous_display_type"
# Marks comparison meshes hidden while distance_heatmap draws them semi-transparent.
PREVIOUS_HIDDEN_KEY = "_biomechanics_previous_hidden"
_previous_viewport_color_modes = {}


class BIOMECHANICS_PG_neural_model_entry(bpy.types.PropertyGroup):
    """One published neural bundle, identified by its directed bone pair."""

    manifest_path: bpy.props.StringProperty(subtype="FILE_PATH")  # pyright: ignore[reportInvalidTypeForm]
    fixed_object_name: bpy.props.StringProperty()  # pyright: ignore[reportInvalidTypeForm]
    moving_object_name: bpy.props.StringProperty()  # pyright: ignore[reportInvalidTypeForm]


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
        if PREVIOUS_HIDDEN_KEY in obj:
            try:
                obj.hide_set(bool(obj[PREVIOUS_HIDDEN_KEY]))
            except RuntimeError:
                pass
            del obj[PREVIOUS_HIDDEN_KEY]
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


def _heatmap_active():
    # Imported lazily: distance_heatmap imports this module.
    from .distance_heatmap import is_heatmap_active

    return is_heatmap_active()


def refresh_transparency_overlay(scene, excluded_pair_name=None):
    _restore_transparency_overlay()
    # Solid-mode object transparency requires Object color, which would hide the
    # heatmap's Attribute colors. With the heatmap on, transparent comparison meshes
    # are hidden and distance_heatmap draws them semi-transparent itself.
    heatmap_overlay = not scene.biomechanics_overlay_wireframe and _heatmap_active()

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
        if heatmap_overlay:
            try:
                obj[PREVIOUS_HIDDEN_KEY] = obj.hide_get()
                obj.hide_set(True)
            except RuntimeError:
                obj.pop(PREVIOUS_HIDDEN_KEY, None)
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

    if scene.biomechanics_overlay_wireframe or heatmap_overlay:
        return

    for space in _view_3d_spaces():
        if space.shading.type != "SOLID":
            continue
        _previous_viewport_color_modes[space.as_pointer()] = space.shading.color_type
        space.shading.color_type = "OBJECT"


def _update_transparency_overlay(scene, context):
    selected_object = context.active_object if context is not None else bpy.context.active_object
    excluded_pair_name = None
    if selected_object is not None and selected_object.type == "MESH" and any(
        collection is not None and collection.objects.get(selected_object.name) is selected_object
        for collection in (scene.biomechanics_source_collection, scene.biomechanics_target_collection)
    ):
        excluded_pair_name = base_object_name(selected_object.name)

    refresh_transparency_overlay(scene, excluded_pair_name)


def _update_heatmap_scale(scene, context):
    from .distance_heatmap import update_heatmap_scale

    update_heatmap_scale(scene, context)


def _update_muscle_delta_scale(scene, context):
    from .muscle_strings import update_delta_scale

    update_delta_scale(scene, context)


def register_properties():
    bpy.types.Scene.biomechanics_heatmap_scale = bpy.props.EnumProperty(
        name="Heatmap Scale",
        description="Which maximum distance maps to red in the correspondence heatmap",
        items=(
            ("GLOBAL", "All Bones", "Red is the largest distance across all pairs, so bones are comparable"),
            ("PER_BONE", "Per Bone", "Red is each pair's own largest distance, showing the gradient within every bone"),
        ),
        default="GLOBAL",
        update=_update_heatmap_scale,
    )
    bpy.types.Scene.biomechanics_muscle_delta_unit = bpy.props.EnumProperty(
        name="Length Change Unit",
        description="How muscle string length change is measured for coloring",
        items=(
            ("PERCENT", "Percent", "Change relative to the Original length (strain), comparable across muscles"),
            ("MM", "mm", "Absolute change in millimeters"),
        ),
        default="PERCENT",
        update=_update_muscle_delta_scale,
    )
    bpy.types.Scene.biomechanics_muscle_delta_range = bpy.props.FloatProperty(
        name="Color Range",
        description="Length change that maps to full blue (shorter) or full red (longer)",
        default=10.0,
        min=0.1,
        soft_max=50.0,
        update=_update_muscle_delta_scale,
    )
    bpy.types.Scene.biomechanics_muscle_sheet_range = bpy.props.FloatProperty(
        name="Sheet Range %",
        description="Triangle area change in percent that maps to full blue (smaller) or full red (larger)",
        default=20.0,
        min=0.1,
        soft_max=100.0,
        update=_update_muscle_delta_scale,
    )
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
    bpy.types.Scene.biomechanics_training_seed = bpy.props.IntProperty(
        name="Sampling Seed",
        description="Deterministic seed recorded in the exported package for background dataset sampling",
        min=0,
        default=12345,
    )
    bpy.types.Scene.biomechanics_training_rotation_center = bpy.props.FloatVectorProperty(
        name="Training Rotation Center",
        description="World-space point used as the rotation center for generated training poses",
        size=3,
        subtype="TRANSLATION",
        precision=6,
    )
    bpy.types.Scene.biomechanics_neural_translation_domain_mm = bpy.props.FloatProperty(
        name="Translation Domain (mm)",
        description="Maximum sampled local translation from neutral on each axis",
        min=0.001,
        max=1000.0,
        default=10.0,
        precision=2,
    )
    bpy.types.Scene.biomechanics_neural_rotation_domain_degrees = bpy.props.FloatProperty(
        name="Rotation Limit (degrees)",
        description="Maximum sampled local rotation from neutral on each axis when domain limiting is enabled",
        min=0.001,
        max=180.0,
        default=10.0,
        precision=2,
    )
    bpy.types.Scene.biomechanics_neural_limit_rotation_domain = bpy.props.BoolProperty(
        name="Limit Rotation Domain",
        description="Restrict training to a smaller angular neighborhood; disabled maps the full 360-degree orientation space",
        default=False,
    )
    bpy.types.Scene.biomechanics_training_export_directory = bpy.props.StringProperty(
        name="Training Folder",
        description="Folder for terminal training inputs and results",
        subtype="DIR_PATH",
        default="//biomechanics_training",
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
    bpy.types.Scene.biomechanics_neural_models = bpy.props.CollectionProperty(
        type=BIOMECHANICS_PG_neural_model_entry,
    )
    bpy.types.Scene.biomechanics_neural_model_status = bpy.props.StringProperty(
        name="Neural Model Import Status",
        description="Result of the last neural model import/compatibility check",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_training_status = bpy.props.StringProperty(
        name="Neural Training Status",
        description="Current status of the external neural-training pipeline",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_training_pair = bpy.props.StringProperty(
        name="Neural Training Pair",
        description="Fixed and moving bone pair for the current or most recent neural-training run",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_training_status_path = bpy.props.StringProperty(
        name="Neural Training Status File",
        description="Atomic status file written by the external training pipeline",
        subtype="FILE_PATH",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_training_active = bpy.props.BoolProperty(
        name="Neural Training Active",
        description="Whether an external neural-training process is currently running",
        options={"HIDDEN"},
    )
    bpy.types.Scene.biomechanics_neural_training_sample_budget = bpy.props.IntProperty(
        name="Base Sample Budget",
        description="Total hierarchical samples before balancing and active learning",
        min=1000,
        max=100000,
        default=10000,
    )
    bpy.types.Scene.biomechanics_neural_training_active_learning = bpy.props.BoolProperty(
        name="Active Learning",
        description="Run bounded uncertain-pose labeling and retraining after the baseline model",
        default=False,
    )
    bpy.types.Scene.biomechanics_neural_training_active_rounds = bpy.props.IntProperty(
        name="Active-Learning Rounds",
        description="Maximum number of bounded active-learning rounds",
        min=1,
        max=50,
        default=5,
    )
    bpy.types.Scene.biomechanics_neural_training_max_new_samples = bpy.props.IntProperty(
        name="Active-Learning Samples",
        description="Maximum total new full-mesh labels requested by active learning",
        min=1,
        max=10000,
        default=300,
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
        name="Fallback Translation Limit (mm)",
        description="Maximum local translation per axis when no usable neural projection is available",
        min=0.0,
        default=5.0,
        precision=2,
    )
    bpy.types.Scene.biomechanics_rotation_limit = bpy.props.FloatProperty(
        name="Fallback Rotation Limit",
        description="Maximum total rotation when no usable neural projection is available",
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
        if not scene.is_property_set("biomechanics_training_rotation_center"):
            scene.biomechanics_training_rotation_center = scene.cursor.location
        if "biomechanics_rigid_transparent" in scene:
            del scene["biomechanics_rigid_transparent"]
        if scene.biomechanics_source_collection is None:
            scene.biomechanics_source_collection = bpy.data.collections.get("Collection")
        if scene.biomechanics_target_collection is None:
            scene.biomechanics_target_collection = bpy.data.collections.get("rigid")
        rigid_collection = scene.biomechanics_target_collection
        if rigid_collection is not None:
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
    del bpy.types.Scene.biomechanics_neural_training_sample_budget
    del bpy.types.Scene.biomechanics_neural_training_max_new_samples
    del bpy.types.Scene.biomechanics_neural_training_active_rounds
    del bpy.types.Scene.biomechanics_neural_training_active_learning
    del bpy.types.Scene.biomechanics_neural_training_active
    del bpy.types.Scene.biomechanics_neural_training_status_path
    del bpy.types.Scene.biomechanics_neural_training_pair
    del bpy.types.Scene.biomechanics_neural_training_status
    del bpy.types.Scene.biomechanics_neural_model_status
    del bpy.types.Scene.biomechanics_neural_models
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
    del bpy.types.Scene.biomechanics_training_export_directory
    del bpy.types.Scene.biomechanics_neural_limit_rotation_domain
    del bpy.types.Scene.biomechanics_neural_rotation_domain_degrees
    del bpy.types.Scene.biomechanics_neural_translation_domain_mm
    del bpy.types.Scene.biomechanics_training_rotation_center
    del bpy.types.Scene.biomechanics_training_seed
    del bpy.types.Scene.biomechanics_training_moving_object
    del bpy.types.Scene.biomechanics_training_fixed_object
    del bpy.types.Scene.biomechanics_heatmap_scale