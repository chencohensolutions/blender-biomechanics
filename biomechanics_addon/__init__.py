bl_info = {
    "name": "Biomechanics Simulator",
    "author": "Biomechanics Project",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar > Biomechanics",
    "description": "Tools for simulating biomechanical movement",
    "category": "3D View",
}

import bpy  # pyright: ignore[reportMissingImports]

from .correspondence import (
    BIOMECHANICS_OT_clear_correspondence_max,
    BIOMECHANICS_OT_find_max_correspondence_distance,
    BIOMECHANICS_OT_toggle_correspondence_hover,
    register_correspondence,
    unregister_correspondence,
)
from .distance_heatmap import BIOMECHANICS_OT_toggle_distance_heatmap, register_heatmap, unregister_heatmap
from .interaction import register_interaction, unregister_interaction
from .neural_constraints import clear_session_cache
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
    shutdown_neural_training,
)
from .panels import (
    BIOMECHANICS_PT_correspondence_panel,
    BIOMECHANICS_PT_display_panel,
    BIOMECHANICS_PT_interaction_panel,
    BIOMECHANICS_PT_neural_dataset_panel,
    BIOMECHANICS_PT_main_panel,
    BIOMECHANICS_PT_neural_model_panel,
    BIOMECHANICS_PT_setup_panel,
)
from .properties import BIOMECHANICS_PG_neural_model_entry, register_properties, unregister_properties


CLASSES = (
    BIOMECHANICS_PG_neural_model_entry,
    BIOMECHANICS_OT_export_training_package,
    BIOMECHANICS_OT_start_neural_training,
    BIOMECHANICS_OT_check_onnx_compatibility,
    BIOMECHANICS_OT_import_neural_model,
    BIOMECHANICS_OT_clear_neural_model,
    BIOMECHANICS_OT_center_cursor_on_trained_pose,
    BIOMECHANICS_OT_copy_interaction_info,
    BIOMECHANICS_OT_apply_training_center_to_cursor,
    BIOMECHANICS_OT_edit_training_center,
    BIOMECHANICS_OT_validate_neural_pose,
    BIOMECHANICS_OT_start_interaction,
    BIOMECHANICS_OT_reset_interaction,
    BIOMECHANICS_OT_stop_interaction,
    BIOMECHANICS_OT_align_to_source,
    BIOMECHANICS_OT_toggle_correspondence_hover,
    BIOMECHANICS_OT_find_max_correspondence_distance,
    BIOMECHANICS_OT_clear_correspondence_max,
    BIOMECHANICS_OT_toggle_distance_heatmap,
    BIOMECHANICS_PT_main_panel,
    BIOMECHANICS_PT_setup_panel,
    BIOMECHANICS_PT_display_panel,
    BIOMECHANICS_PT_correspondence_panel,
    BIOMECHANICS_PT_interaction_panel,
    BIOMECHANICS_PT_neural_dataset_panel,
    BIOMECHANICS_PT_neural_model_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    register_properties()
    register_interaction()
    register_correspondence()
    register_heatmap()


def unregister():
    shutdown_neural_training()
    unregister_heatmap()
    unregister_correspondence()
    unregister_interaction()
    unregister_properties()
    clear_session_cache()

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()