import bpy  # pyright: ignore[reportMissingImports]
import textwrap

from .onnx_compat import BIOMECHANICS_OT_check_onnx_compatibility
from .operators import (
    BIOMECHANICS_OT_align_to_source,
    BIOMECHANICS_OT_clear_neural_model,
    BIOMECHANICS_OT_center_cursor_on_trained_pose,
    BIOMECHANICS_OT_copy_interaction_info,
    BIOMECHANICS_OT_apply_training_center_to_cursor,
    BIOMECHANICS_OT_edit_training_center,
    BIOMECHANICS_OT_export_training_package,
    BIOMECHANICS_OT_import_neural_model,
    BIOMECHANICS_OT_start_neural_training,
    BIOMECHANICS_OT_validate_neural_pose,
    BIOMECHANICS_OT_reset_interaction,
    BIOMECHANICS_OT_start_interaction,
    BIOMECHANICS_OT_stop_interaction,
)


def _wrapped_labels(layout, text, icon=None, context=None):
    region_width = context.region.width if context is not None else 220
    ui_scale = context.preferences.system.ui_scale if context is not None else 1.0
    width = max(18, min(60, int((region_width - 32) / (7.0 * ui_scale))))
    lines = textwrap.wrap(text, width=width) or [""]
    for index, line in enumerate(lines):
        layout.label(text=line, icon=icon if index == 0 else "NONE")


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
        else:
            manipulated = context.scene.biomechanics_manipulated_object
            layout.label(
                text=manipulated.name if manipulated is not None else "Missing bone",
                icon="MESH_DATA",
            )
            layout.prop(context.scene, "biomechanics_live_solve")
            fallback = layout.column(align=True)
            fallback.prop(context.scene, "biomechanics_translation_limit_mm")
            fallback.prop(context.scene, "biomechanics_rotation_limit")
            matching_models = [
                entry
                for entry in context.scene.biomechanics_neural_models
                if manipulated is not None and entry.moving_object_name == manipulated.name
            ]
            if matching_models:
                _wrapped_labels(
                    layout,
                    "Neural model controls this bone's trained movement domain",
                    icon="INFO",
                    context=context,
                )
            else:
                _wrapped_labels(
                    layout,
                    f"No neural model trained to move {manipulated.name if manipulated else 'this bone'}",
                    icon="ERROR",
                    context=context,
                )
                layout.label(text="Using fallback scalar limits", icon="INFO")
            if context.scene.biomechanics_neural_model_status:
                _wrapped_labels(
                    layout,
                    context.scene.biomechanics_neural_model_status,
                    icon="INFO",
                    context=context,
                )
            _wrapped_labels(
                layout,
                "Move the selected bone or target with G; rotate with R",
                icon="INFO",
                context=context,
            )
            row = layout.row(align=True)
            row.operator(BIOMECHANICS_OT_reset_interaction.bl_idname, icon="LOOP_BACK")
            row.operator(BIOMECHANICS_OT_stop_interaction.bl_idname, icon="CHECKMARK")
            layout.operator(
                BIOMECHANICS_OT_center_cursor_on_trained_pose.bl_idname,
                icon="PIVOT_CURSOR",
            )
            layout.operator(
                BIOMECHANICS_OT_copy_interaction_info.bl_idname,
                text="Copy Panel Info",
                icon="COPY_ID",
            )


class BIOMECHANICS_PT_neural_dataset_panel(bpy.types.Panel):
    bl_label = "Geometric Dataset"
    bl_idname = "BIOMECHANICS_PT_neural_dataset_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.prop(scene, "biomechanics_training_fixed_object", text="Fixed Pair")
        layout.prop(scene, "biomechanics_training_moving_object", text="Moving Pair")
        fixed_object = scene.biomechanics_training_fixed_object
        moving_object = scene.biomechanics_training_moving_object
        if fixed_object is not None and moving_object is not None:
            layout.label(
                text=f"Selected training pair: {fixed_object.name} -> {moving_object.name}",
                icon="MESH_DATA",
            )
        layout.prop(scene, "biomechanics_neural_translation_domain_mm")
        layout.prop(scene, "biomechanics_neural_limit_rotation_domain")
        if scene.biomechanics_neural_limit_rotation_domain:
            layout.prop(scene, "biomechanics_neural_rotation_domain_degrees")
        else:
            layout.label(text="Rotation domain: full 360 degrees", icon="ORIENTATION_GIMBAL")
        layout.prop(scene, "biomechanics_neural_training_sample_budget")
        layout.label(text="Joint radius: 50 mm; target: 0.5 mm", icon="INFO")
        layout.label(text="Training Rotation Center", icon="PIVOT_CURSOR")
        layout.operator(BIOMECHANICS_OT_edit_training_center.bl_idname, icon="EMPTY_ARROWS")
        layout.operator(BIOMECHANICS_OT_apply_training_center_to_cursor.bl_idname, icon="PIVOT_CURSOR")
        layout.prop(scene, "biomechanics_training_export_directory")
        layout.prop(scene, "biomechanics_neural_training_active_learning")
        if scene.biomechanics_neural_training_active_learning:
            layout.prop(scene, "biomechanics_neural_training_active_rounds")
            layout.prop(scene, "biomechanics_neural_training_max_new_samples")
        layout.operator(
            BIOMECHANICS_OT_export_training_package.bl_idname,
            icon="EXPORT",
        )
        start_row = layout.row()
        start_row.enabled = not scene.biomechanics_neural_training_active
        start_row.operator(
            BIOMECHANICS_OT_start_neural_training.bl_idname,
            icon="PLAY",
        )
        if scene.biomechanics_neural_training_pair:
            layout.label(text=f"Last training run: {scene.biomechanics_neural_training_pair}", icon="TIME")
        if scene.biomechanics_neural_training_status:
            layout.label(text=scene.biomechanics_neural_training_status, icon="INFO")
        if scene.biomechanics_neural_training_status_path:
            layout.label(text=scene.biomechanics_neural_training_status_path, icon="FILE")


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
        layout.operator(
            BIOMECHANICS_OT_validate_neural_pose.bl_idname,
            icon="CHECKMARK",
        )
        for entry in context.scene.biomechanics_neural_models:
            layout.label(
                text=f"{entry.fixed_object_name} / {entry.moving_object_name}",
                icon="FILE",
            )
        if context.scene.biomechanics_neural_models:
            layout.operator(
                BIOMECHANICS_OT_clear_neural_model.bl_idname,
                icon="X",
            )
        if context.scene.biomechanics_neural_model_status:
            layout.label(text=context.scene.biomechanics_neural_model_status, icon="INFO")