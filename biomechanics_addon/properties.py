import bpy

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


def register_properties():
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
        _update_transparency_overlay(scene, None)


def unregister_properties():
    _restore_transparency_overlay()
    del bpy.types.Scene.biomechanics_overlay_on_top
    del bpy.types.Scene.biomechanics_overlay_wireframe
    del bpy.types.Scene.biomechanics_overlay_opacity
    del bpy.types.Scene.biomechanics_transparency_mode
    del bpy.types.Scene.biomechanics_target_collection
    del bpy.types.Scene.biomechanics_source_collection