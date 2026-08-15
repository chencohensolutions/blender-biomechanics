import bpy

from .operators import BIOMECHANICS_OT_align_to_source


class BIOMECHANICS_PT_main_panel(bpy.types.Panel):
    bl_label = "Biomechanics Simulator"
    bl_idname = "BIOMECHANICS_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Bone Movement Tools", icon="ARMATURE_DATA")
        layout.prop(
            context.scene,
            "biomechanics_source_collection",
            text="Original",
        )
        layout.prop(
            context.scene,
            "biomechanics_target_collection",
            text="Rigid",
        )
        layout.label(text="Comparison Display")
        layout.prop(
            context.scene,
            "biomechanics_transparency_mode",
            expand=True,
        )
        wireframe_row = layout.row()
        wireframe_row.enabled = context.scene.biomechanics_transparency_mode != "NONE"
        wireframe_row.prop(
            context.scene,
            "biomechanics_overlay_wireframe",
            text="Mesh Wireframe",
            toggle=True,
        )
        opacity_row = layout.row()
        opacity_row.enabled = (
            context.scene.biomechanics_transparency_mode != "NONE"
            and not context.scene.biomechanics_overlay_wireframe
        )
        opacity_row.prop(
            context.scene,
            "biomechanics_overlay_opacity",
            slider=True,
        )
        on_top_row = layout.row()
        on_top_row.enabled = context.scene.biomechanics_transparency_mode != "NONE"
        on_top_row.prop(
            context.scene,
            "biomechanics_overlay_on_top",
            text="Overlay On Top",
            toggle=True,
        )
        help_box = layout.box()
        help_box.label(text="1. Choose both collections", icon="INFO")
        help_box.label(text="2. Select a mesh in Original")
        help_box.label(text="3. Align it to its Rigid pair")
        layout.operator(
            BIOMECHANICS_OT_align_to_source.bl_idname,
            icon="CON_LOCLIKE",
        )