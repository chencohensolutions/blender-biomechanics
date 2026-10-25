import bpy  # pyright: ignore[reportMissingImports]
import textwrap

from .correspondence import (
    MAX_RESULTS_SHOWN,
    BIOMECHANICS_OT_clear_correspondence_max,
    BIOMECHANICS_OT_find_max_correspondence_distance,
    BIOMECHANICS_OT_toggle_correspondence_hover,
    is_hover_active,
    max_distance_results,
)
from .distance_heatmap import BIOMECHANICS_OT_toggle_distance_heatmap, is_heatmap_active
from .atlas_import import BIOMECHANICS_OT_import_atlas_ligaments, BIOMECHANICS_OT_import_atlas_muscles
from .chain_interaction import (
    BIOMECHANICS_OT_export_chain_package,
    BIOMECHANICS_OT_precompute_chain_fields,
    BIOMECHANICS_OT_reset_rigid_to_original,
    BIOMECHANICS_OT_select_chain_target,
    BIOMECHANICS_OT_start_chain,
    BIOMECHANICS_OT_stop_chain,
    BIOMECHANICS_OT_toggle_force_colors,
    BIOMECHANICS_OT_toggle_move_bones,
    BIOMECHANICS_OT_toggle_posture_controller,
    chain_ready,
    fields_missing,
    is_active as is_chain_active,
    is_controller_active,
    is_force_colors_active,
    is_free_mode,
    session as chain_session,
)
from .muscle_sheets import BIOMECHANICS_OT_toggle_muscle_sheets, sheets_exist
from .muscle_strings import (
    BIOMECHANICS_OT_build_muscle_strings,
    BIOMECHANICS_OT_export_muscle_lengths,
    BIOMECHANICS_OT_remove_muscle_strings,
    BIOMECHANICS_OT_toggle_muscle_delta_colors,
    BIOMECHANICS_OT_toggle_muscle_hover,
    is_delta_active,
    is_hover_active as is_muscle_hover_active,
    string_collections,
)
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


class BIOMECHANICS_PT_correspondence_panel(bpy.types.Panel):
    bl_label = "Point Correspondence"
    bl_idname = "BIOMECHANICS_PT_correspondence_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname

    def draw(self, context):
        layout = self.layout
        hover_active = is_hover_active()
        layout.operator(
            BIOMECHANICS_OT_toggle_correspondence_hover.bl_idname,
            text="Hover Correspondence: On" if hover_active else "Hover Correspondence: Off",
            icon="RESTRICT_SELECT_OFF" if hover_active else "RESTRICT_SELECT_ON",
            depress=hover_active,
        )
        row = layout.row(align=True)
        row.operator(
            BIOMECHANICS_OT_find_max_correspondence_distance.bl_idname,
            icon="DRIVER_DISTANCE",
        )
        row.operator(
            BIOMECHANICS_OT_clear_correspondence_max.bl_idname,
            text="",
            icon="X",
        )
        heatmap_active = is_heatmap_active()
        layout.operator(
            BIOMECHANICS_OT_toggle_distance_heatmap.bl_idname,
            text="Distance Heatmap: On" if heatmap_active else "Distance Heatmap: Off",
            icon="COLOR",
            depress=heatmap_active,
        )
        scale_row = layout.row()
        scale_row.enabled = heatmap_active
        scale_row.prop(context.scene, "biomechanics_heatmap_scale", expand=True)
        results = max_distance_results()
        if results:
            column = layout.column(align=True)
            for index, result in enumerate(results[:MAX_RESULTS_SHOWN]):
                column.label(
                    text=f"{result['original']}: {result['distance_mm']:.3f} mm (v{result['vertex_index']})",
                    icon="ERROR" if index == 0 else "DOT",
                )


class BIOMECHANICS_PT_muscle_panel(bpy.types.Panel):
    bl_label = "Muscle Strings"
    bl_idname = "BIOMECHANICS_PT_muscle_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator(BIOMECHANICS_OT_build_muscle_strings.bl_idname, icon="CURVE_PATH").kind = "muscle"
        row.operator(BIOMECHANICS_OT_remove_muscle_strings.bl_idname, text="", icon="X")
        layout.operator(
            BIOMECHANICS_OT_build_muscle_strings.bl_idname, text="Build Ligaments & Discs", icon="CONSTRAINT"
        ).kind = "ligament"
        hover_active = is_muscle_hover_active()
        layout.operator(
            BIOMECHANICS_OT_toggle_muscle_hover.bl_idname,
            text="Hover Muscle Lengths: On" if hover_active else "Hover Muscle Lengths: Off",
            icon="RESTRICT_SELECT_OFF" if hover_active else "RESTRICT_SELECT_ON",
            depress=hover_active,
        )
        sheets_active = sheets_exist(context.scene)
        layout.operator(
            BIOMECHANICS_OT_toggle_muscle_sheets.bl_idname,
            text="Muscle Sheets: On" if sheets_active else "Muscle Sheets: Off",
            icon="MESH_GRID",
            depress=sheets_active,
        )
        delta_active = is_delta_active()
        layout.operator(
            BIOMECHANICS_OT_toggle_muscle_delta_colors.bl_idname,
            text="Length Change Colors: On" if delta_active else "Length Change Colors: Off",
            icon="COLOR",
            depress=delta_active,
        )
        scale_column = layout.column(align=True)
        scale_column.enabled = delta_active
        scale_column.row(align=True).prop(context.scene, "biomechanics_muscle_delta_unit", expand=True)
        scale_column.prop(context.scene, "biomechanics_muscle_delta_range")
        sheet_row = scale_column.row()
        sheet_row.enabled = sheets_active
        sheet_row.prop(context.scene, "biomechanics_muscle_sheet_range")
        atlas_row = layout.row(align=True)
        atlas_row.operator(BIOMECHANICS_OT_import_atlas_muscles.bl_idname, icon="IMPORT")
        atlas_row.operator(BIOMECHANICS_OT_import_atlas_ligaments.bl_idname, icon="IMPORT")
        layout.operator(BIOMECHANICS_OT_export_muscle_lengths.bl_idname, icon="EXPORT")
        for bones, kind, collection in sorted(string_collections(context.scene)):
            layout.label(text=f"{bones} {kind}s: {len(collection.objects)}", icon="DOT")


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


class BIOMECHANICS_PT_chain_panel(bpy.types.Panel):
    bl_label = "Bone Chain"
    bl_idname = "BIOMECHANICS_PT_chain_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Biomechanics"
    bl_parent_id = BIOMECHANICS_PT_main_panel.bl_idname

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        active = is_chain_active()
        free = is_free_mode()
        ready, reason = chain_ready(scene)
        layout.operator(
            BIOMECHANICS_OT_toggle_move_bones.bl_idname,
            text="Move Bones: On" if free else "Move Bones: Off",
            icon="VIEW_PAN" if free else "OBJECT_DATAMODE",
            depress=free,
        )
        if free:
            _wrapped_labels(layout, "Click any vertebra, press G (move) or R (rotate), drag, click to confirm.", "INFO", context)
        elif not ready:
            _wrapped_labels(layout, reason, "ERROR", context)
        settings = layout.column(align=True)
        settings.enabled = not active
        settings.prop(scene, "biomechanics_chain_base")
        settings.prop(scene, "biomechanics_chain_top")
        settings.prop(scene, "biomechanics_chain_move_ribs")
        settings.prop(scene, "biomechanics_chain_pin_top")
        settings.prop(scene, "biomechanics_chain_pelvis_hold")
        settings.prop(scene, "biomechanics_chain_rest_from_original")
        settings.prop(scene, "biomechanics_chain_hide_original_strings")
        gravity_row = layout.row(align=True)
        gravity_row.prop(scene, "biomechanics_chain_gravity", text="Gravity", toggle=True, icon="FORCE_FORCE")
        gravity_value = gravity_row.row()
        gravity_value.enabled = scene.biomechanics_chain_gravity
        gravity_value.prop(scene, "biomechanics_chain_gravity_mm", text="")
        muscles = layout.column(align=True)
        controller_on = is_controller_active()
        tone_row = muscles.row()
        tone_row.enabled = not controller_on
        tone_row.prop(scene, "biomechanics_muscle_tone")
        strength_row = muscles.row()
        strength_row.enabled = not active
        strength_row.prop(scene, "biomechanics_muscle_strength_mm")
        controller_row = muscles.row(align=True)
        controller_row.enabled = active
        controller_row.operator(
            BIOMECHANICS_OT_toggle_posture_controller.bl_idname,
            text="Posture Controller: On" if controller_on else "Posture Controller: Off",
            icon="ARMATURE_DATA",
            depress=controller_on,
        )
        muscles.prop(scene, "biomechanics_posture_policy_path", text="")
        follow_row = muscles.row()
        follow_row.enabled = active
        follow_row.prop(scene, "biomechanics_follow_training")
        force_on = is_force_colors_active()
        force_row = layout.row()
        force_row.enabled = active or force_on
        force_row.operator(
            BIOMECHANICS_OT_toggle_force_colors.bl_idname,
            text="Force Colors: On" if force_on else "Force Colors: Off",
            icon="COLOR",
            depress=force_on,
        )
        if force_on:
            _wrapped_labels(layout, "Red: muscle activation. Blue: force carried by ligament or disc.", "INFO", context)
        layout.operator(BIOMECHANICS_OT_reset_rigid_to_original.bl_idname, icon="FILE_REFRESH")
        missing = fields_missing(scene)
        row = layout.row()
        row.enabled = not active
        row.operator(
            BIOMECHANICS_OT_precompute_chain_fields.bl_idname,
            text=f"Precompute Collision Fields ({len(missing)} missing)" if missing else "Collision Fields Cached",
            icon="MESH_GRID",
        )
        if not active:
            if ready:
                layout.operator(BIOMECHANICS_OT_start_chain.bl_idname, text="Start with Target Arrows", icon="PLAY")
        else:
            if not free:
                layout.operator(BIOMECHANICS_OT_select_chain_target.bl_idname, icon="EMPTY_ARROWS")
                _wrapped_labels(
                    layout,
                    f"Move the target arrows or {chain_session().handle_name} itself with G / R.",
                    "INFO",
                    context,
                )
            row = layout.row(align=True)
            row.operator(BIOMECHANICS_OT_stop_chain.bl_idname, text="Stop, Keep Pose", icon="PAUSE").restore = False
            row.operator(BIOMECHANICS_OT_stop_chain.bl_idname, text="Stop, Reset", icon="LOOP_BACK").restore = True
            layout.operator(BIOMECHANICS_OT_export_chain_package.bl_idname, icon="EXPORT")
            status = chain_session().status(scene)
            column = layout.column(align=True)
            column.label(text=f"{status['bodies'] - 1} bones, {status['constraints']} constraints", icon="DOT")
            column.label(text=f"{status['muscles']} muscle strings in {status['muscle_groups']} groups", icon="DOT")
            column.label(text=f"Mean activation {status['activation_mean']:.2f}", icon="DOT")
            if status["policy_score"] is not None:
                column.label(
                    text=f"Policy generation {status['policy_generation']}, score {status['policy_score']:.2f}",
                    icon="DOT",
                )
            height = status["height_cm"]
            column.label(
                text=f"Top bone height {height:+.1f} cm" + ("  FAILED (below -5 cm)" if height <= -5.0 else ""),
                icon="ERROR" if height <= -5.0 else "DOT",
            )
            column.label(text=f"Penetration {status['penetration_mm']:.2f} mm", icon="DOT")
            column.label(text=f"Max ligament strain {status['strain_percent']:.1f}%", icon="DOT")
            column.label(text=f"Solve {status['solve_ms']:.0f} ms", icon="DOT")
        tuning = layout.column(align=True)
        tuning.prop(scene, "biomechanics_chain_iterations")
        tuning.prop(scene, "biomechanics_chain_substeps")
        tuning.prop(scene, "biomechanics_chain_handle_compliance")
        tuning.prop(scene, "biomechanics_chain_margin_mm")


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