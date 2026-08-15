import bpy

from .onnx_compat import BIOMECHANICS_OT_check_onnx_compatibility
from .operators import (
    BIOMECHANICS_OT_align_to_source,
    BIOMECHANICS_OT_clear_neural_model,
    BIOMECHANICS_OT_export_training_package,
    BIOMECHANICS_OT_import_neural_model,
    BIOMECHANICS_OT_reset_interaction,
    BIOMECHANICS_OT_start_interaction,
    BIOMECHANICS_OT_start_point_interaction,
    BIOMECHANICS_OT_stop_interaction,
    BIOMECHANICS_OT_train_two_objects,
)
from .training import planned_sample_count


class BIOMECHANICS_PT_main_panel(bpy.types.Panel):
    bl_label = "Biomechanics Simulator"
    bl_idname = "BIOMECHANICS_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"

    def draw(self, _context):
        self.layout.label(text="Bone Movement Tools", icon="ARMATURE_DATA")


class BIOMECHANICS_PT_setup_panel(bpy.types.Panel):
    bl_label = "Setup & Alignment"
    bl_idname = "BIOMECHANICS_PT_setup_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname

    def draw(self, context):
        layout = self.layout
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
        layout.operator(
            BIOMECHANICS_OT_align_to_source.bl_idname,
            icon="CON_LOCLIKE",
        )


class BIOMECHANICS_PT_display_panel(bpy.types.Panel):
    bl_label = "Comparison Display"
    bl_idname = "BIOMECHANICS_PT_display_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
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


class BIOMECHANICS_PT_interaction_panel(bpy.types.Panel):
    bl_label = "Constrained Interaction"
    bl_idname = "BIOMECHANICS_PT_interaction_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "biomechanics_move_attached_bones")
        target = context.scene.biomechanics_pose_target
        if target is None:
            layout.label(text="Select one mesh in Rigid")
            layout.operator(
                BIOMECHANICS_OT_start_interaction.bl_idname,
                icon="EMPTY_ARROWS",
            )
            layout.label(text="Edit Mode: select vertices for point drag")
            layout.operator(
                BIOMECHANICS_OT_start_point_interaction.bl_idname,
                icon="PIVOT_CURSOR",
            )
        else:
            manipulated = context.scene.biomechanics_manipulated_object
            layout.label(
                text=manipulated.name if manipulated is not None else "Missing bone",
                icon="MESH_DATA",
            )
            layout.prop(context.scene, "biomechanics_live_solve")
            if target.get("biomechanics_point_target"):
                layout.label(text="Move point target with G", icon="PIVOT_CURSOR")
            else:
                layout.prop(context.scene, "biomechanics_translation_limit_mm")
                layout.prop(context.scene, "biomechanics_rotation_limit")
                layout.label(text="Move target with G; rotate with R", icon="INFO")
            row = layout.row(align=True)
            row.operator(BIOMECHANICS_OT_reset_interaction.bl_idname, icon="LOOP_BACK")
            row.operator(BIOMECHANICS_OT_stop_interaction.bl_idname, icon="CHECKMARK")


class BIOMECHANICS_PT_training_panel(bpy.types.Panel):
    bl_label = "Two-Bone Training"
    bl_idname = "BIOMECHANICS_PT_training_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "biomechanics_training_fixed_object")
        layout.prop(context.scene, "biomechanics_training_moving_object")
        layout.prop(context.scene, "biomechanics_training_translation_mm")
        layout.prop(context.scene, "biomechanics_training_rotation")
        row = layout.row(align=True)
        row.prop(context.scene, "biomechanics_training_translation_steps", text="Move Steps")
        row.prop(context.scene, "biomechanics_training_rotation_steps", text="Turn Steps")
        layout.prop(context.scene, "biomechanics_training_clearance_mm")
        layout.prop(context.scene, "biomechanics_training_proxy_faces")
        sample_count = planned_sample_count(
            context.scene.biomechanics_training_translation_steps,
            context.scene.biomechanics_training_rotation_steps,
            context.scene.biomechanics_training_translation_mm,
            context.scene.biomechanics_training_rotation,
        )
        layout.label(text=f"Planned samples: {sample_count:,}", icon="INFO")
        layout.prop(context.scene, "biomechanics_training_maximum_samples")
        layout.operator(BIOMECHANICS_OT_train_two_objects.bl_idname, icon="PLAY")
        layout.separator()
        layout.prop(context.scene, "biomechanics_training_export_directory")
        layout.prop(context.scene, "biomechanics_training_seed")
        layout.operator(
            BIOMECHANICS_OT_export_training_package.bl_idname,
            icon="EXPORT",
        )
        if context.scene.biomechanics_training_summary:
            layout.label(text=context.scene.biomechanics_training_summary, icon="INFO")
        layout.prop(
            context.scene,
            "biomechanics_training_result_text",
            text="Active Training Data",
            icon="TEXT",
        )


class BIOMECHANICS_PT_neural_model_panel(bpy.types.Panel):
    bl_label = "External Neural Model"
    bl_idname = "BIOMECHANICS_PT_neural_model_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.operator(
            BIOMECHANICS_OT_check_onnx_compatibility.bl_idname,
            icon="CHECKMARK",
        )
        if context.scene.biomechanics_onnx_status:
            layout.label(text=context.scene.biomechanics_onnx_status, icon="INFO")

        layout.separator()
        layout.operator(
            BIOMECHANICS_OT_import_neural_model.bl_idname,
            icon="IMPORT",
        )
        if context.scene.biomechanics_neural_model_path:
            layout.label(text=context.scene.biomechanics_neural_model_path, icon="FILE")
            layout.operator(
                BIOMECHANICS_OT_clear_neural_model.bl_idname,
                icon="X",
            )
        if context.scene.biomechanics_neural_model_status:
            layout.label(text=context.scene.biomechanics_neural_model_status, icon="INFO")