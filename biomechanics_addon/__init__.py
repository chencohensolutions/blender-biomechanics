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

from .operators import BIOMECHANICS_OT_align_to_source
from .panels import BIOMECHANICS_PT_main_panel
from .properties import register_properties, unregister_properties


CLASSES = (
    BIOMECHANICS_OT_align_to_source,
    BIOMECHANICS_PT_main_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    register_properties()


def unregister():
    unregister_properties()

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()