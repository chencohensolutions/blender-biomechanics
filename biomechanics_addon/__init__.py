bl_info = {
    "name": "Biomechanics Simulator",
    "author": "Biomechanics Project",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar > Biomechanics",
    "description": "Tools for simulating biomechanical movement",
    "category": "3D View",
}

import bpy

from .interaction import register_interaction, unregister_interaction
from .neural_constraints import clear_session_cache
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
from .panels import (
    BIOMECHANICS_PT_display_panel,
    BIOMECHANICS_PT_interaction_panel,
    BIOMECHANICS_PT_main_panel,
    BIOMECHANICS_PT_neural_model_panel,
    BIOMECHANICS_PT_setup_panel,
    BIOMECHANICS_PT_training_panel,
)
from .properties import register_properties, unregister_properties


CLASSES = (
    BIOMECHANICS_OT_train_two_objects,
    BIOMECHANICS_OT_export_training_package,
    BIOMECHANICS_OT_check_onnx_compatibility,
    BIOMECHANICS_OT_import_neural_model,
    BIOMECHANICS_OT_clear_neural_model,
    BIOMECHANICS_OT_start_interaction,
    BIOMECHANICS_OT_start_point_interaction,
    BIOMECHANICS_OT_reset_interaction,
    BIOMECHANICS_OT_stop_interaction,
    BIOMECHANICS_OT_align_to_source,
    BIOMECHANICS_PT_main_panel,
    BIOMECHANICS_PT_setup_panel,
    BIOMECHANICS_PT_display_panel,
    BIOMECHANICS_PT_interaction_panel,
    BIOMECHANICS_PT_training_panel,
    BIOMECHANICS_PT_neural_model_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    register_properties()
    register_interaction()


def unregister():
    unregister_interaction()
    unregister_properties()
    clear_session_cache()

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()