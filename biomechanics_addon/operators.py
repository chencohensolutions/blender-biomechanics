import bpy

from .alignment import base_object_name, mesh_alignment
from .properties import refresh_transparency_overlay


class BIOMECHANICS_OT_align_to_source(bpy.types.Operator):
    """Match the selected mesh by shape and move the original collection as one rigid group"""

    bl_idname = "biomechanics.align_to_source"
    bl_label = "Align to Source"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_collection = context.scene.biomechanics_source_collection
        target_collection = context.scene.biomechanics_target_collection
        selected_object = context.active_object

        if source_collection is None or target_collection is None:
            self.report({"ERROR"}, "Choose the original and rigid collections")
            return {"CANCELLED"}
        if source_collection == target_collection:
            self.report({"ERROR"}, "Original and rigid collections must be different")
            return {"CANCELLED"}
        if (
            selected_object is None
            or selected_object.type != "MESH"
            or source_collection.objects.get(selected_object.name) is not selected_object
        ):
            self.report({"ERROR"}, "Select one mesh object from the original collection")
            return {"CANCELLED"}

        pair_name = base_object_name(selected_object.name)
        matching_objects = [
            obj
            for obj in target_collection.objects
            if obj.type == "MESH" and base_object_name(obj.name) == pair_name
        ]
        if not matching_objects:
            self.report({"ERROR"}, f'No rigid pair found for "{selected_object.name}"')
            return {"CANCELLED"}
        if len(matching_objects) > 1:
            self.report({"ERROR"}, f'Multiple rigid pairs found for "{selected_object.name}"')
            return {"CANCELLED"}

        try:
            alignment = mesh_alignment(selected_object, matching_objects[0])
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        moved_count = 0
        for obj in source_collection.objects:
            if obj.type == "MESH":
                obj.matrix_world = alignment @ obj.matrix_world
                moved_count += 1

        refresh_transparency_overlay(context.scene, pair_name)

        self.report(
            {"INFO"},
            f'Aligned {moved_count} objects using "{selected_object.name}"',
        )
        return {"FINISHED"}