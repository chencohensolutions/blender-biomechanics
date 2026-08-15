import bpy
import bmesh
import json
import math
from pathlib import Path
from mathutils import Vector

from .alignment import base_object_name, mesh_alignment
from .interaction import (
    POINT_TARGET_NAME,
    TARGET_NAME,
    clear_trained_constraints,
    clear_attached_bones,
    prepare_attached_bones,
    prepare_trained_constraints,
    remove_pose_targets,
    set_neutral_matrix,
    solve_scene,
)
from . import neural_constraints
from .properties import refresh_transparency_overlay
from .schemas import SchemaError
from .training import build_geometry_package, planned_sample_count, train_two_objects


def _training_objects(scene):
    rigid_collection = scene.biomechanics_target_collection
    fixed_object = scene.biomechanics_training_fixed_object
    moving_object = scene.biomechanics_training_moving_object
    if rigid_collection is None:
        raise ValueError("Choose the rigid collection")
    if fixed_object is None or moving_object is None:
        raise ValueError("Choose fixed and moving bones")
    if fixed_object == moving_object:
        raise ValueError("Fixed and moving bones must be different")
    for obj in (fixed_object, moving_object):
        if obj.type != "MESH" or rigid_collection.objects.get(obj.name) is not obj:
            raise ValueError("Both training bones must be meshes in the rigid collection")
    return rigid_collection, fixed_object, moving_object


def _matrix_values(matrix):
    return [value for row in matrix for value in row]


def _validate_export_preconditions(scene, fixed_object, moving_object):
    errors = []
    if scene.biomechanics_pose_target is not None:
        errors.append(
            "Stop constrained interaction (Reset or Commit) before exporting; "
            "the transform may not be neutral"
        )
    for obj in (fixed_object, moving_object):
        # A near-zero axis (not just small uniform scale) collapses the transform.
        if min(abs(component) for component in obj.scale) < 1e-6:
            errors.append(f"{obj.name} has an unsupported near-zero scale axis")
    return errors


class BIOMECHANICS_OT_train_two_objects(bpy.types.Operator):
    """Sample collision and clearance for two rigid meshes across a coupled pose grid"""

    bl_idname = "biomechanics.train_two_objects"
    bl_label = "Train Two Rigid Bones"

    def execute(self, context):
        scene = context.scene
        try:
            rigid_collection, fixed_object, moving_object = _training_objects(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        source_matrices = {
            obj.name: obj.matrix_world.copy()
            for obj in rigid_collection.objects
            if obj.type == "MESH"
        }
        scene.biomechanics_live_solve = False
        scene.biomechanics_pose_target = None
        scene.biomechanics_manipulated_object = None
        clear_trained_constraints(scene)
        clear_attached_bones(scene)
        remove_pose_targets(scene)

        try:
            text, result = train_two_objects(
                context,
                fixed_object,
                moving_object,
                scene.biomechanics_training_translation_mm,
                math.degrees(scene.biomechanics_training_rotation),
                scene.biomechanics_training_translation_steps,
                scene.biomechanics_training_rotation_steps,
                scene.biomechanics_training_clearance_mm,
                scene.biomechanics_training_proxy_faces,
                maximum_samples=scene.biomechanics_training_maximum_samples,
                collection_name=rigid_collection.name,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            for object_name, source_matrix in source_matrices.items():
                obj = bpy.data.objects.get(object_name)
                if obj is not None:
                    obj.matrix_world = source_matrix
            context.view_layer.update()
        summary = result["summary"]
        scene.biomechanics_training_result_text = text
        scene.biomechanics_training_summary = (
            f'{summary["valid_samples"]} valid / {summary["total_samples"]} samples'
        )
        set_neutral_matrix(scene, moving_object.matrix_world)
        using_training, constraint_message = prepare_trained_constraints(scene, moving_object)
        report_level = {"INFO"} if using_training else {"WARNING"}
        self.report(
            report_level,
            f"Training complete: {scene.biomechanics_training_summary}. "
            f"{constraint_message}. Live Solve remains off",
        )
        return {"FINISHED"}


class BIOMECHANICS_OT_export_training_package(bpy.types.Operator):
    """Export two rigid meshes and settings for background Blender training"""

    bl_idname = "biomechanics.export_training_package"
    bl_label = "Export Terminal Training Package"

    def execute(self, context):
        scene = context.scene
        try:
            rigid_collection, fixed_object, moving_object = _training_objects(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        precondition_errors = _validate_export_preconditions(scene, fixed_object, moving_object)
        if precondition_errors:
            self.report({"ERROR"}, "; ".join(precondition_errors))
            return {"CANCELLED"}

        total_samples = planned_sample_count(
            scene.biomechanics_training_translation_steps,
            scene.biomechanics_training_rotation_steps,
            scene.biomechanics_training_translation_mm,
            scene.biomechanics_training_rotation,
        )
        if total_samples > scene.biomechanics_training_maximum_samples:
            self.report(
                {"ERROR"},
                f"Planned sample count {total_samples:,} exceeds the hard limit "
                f"of {scene.biomechanics_training_maximum_samples:,}",
            )
            return {"CANCELLED"}

        export_directory = Path(bpy.path.abspath(scene.biomechanics_training_export_directory))
        try:
            export_directory.mkdir(parents=True, exist_ok=True)
            probe_path = export_directory / ".biomechanics_write_check"
            probe_path.write_text("", encoding="utf-8")
            probe_path.unlink()
        except OSError as exc:
            self.report({"ERROR"}, f"Export folder is not writable: {exc}")
            return {"CANCELLED"}
        input_path = export_directory / "training_input.blend"
        config_path = export_directory / "training_config.json"
        command_path = export_directory / "RUN_COMMAND.txt"
        if input_path.exists():
            input_path.unlink()

        bpy.data.libraries.write(
            str(input_path),
            {fixed_object, moving_object},
            compress=True,
        )

        config = {
            "package_version": 1,
            "input_blend": input_path.name,
            "output_json": "training_result.json",
            "collection": rigid_collection.name,
            "scale_length": scene.unit_settings.scale_length or 1.0,
            "fixed_object": {
                "name": fixed_object.name,
                "matrix_world": _matrix_values(fixed_object.matrix_world),
            },
            "moving_object": {
                "name": moving_object.name,
                "matrix_world": _matrix_values(moving_object.matrix_world),
            },
            "settings": {
                "translation_limit_mm": scene.biomechanics_training_translation_mm,
                "rotation_limit_degrees": math.degrees(scene.biomechanics_training_rotation),
                "translation_steps": scene.biomechanics_training_translation_steps,
                "rotation_steps": scene.biomechanics_training_rotation_steps,
                "minimum_clearance_mm": scene.biomechanics_training_clearance_mm,
                "maximum_proxy_faces": scene.biomechanics_training_proxy_faces,
                "maximum_samples": scene.biomechanics_training_maximum_samples,
            },
            "planned_samples": total_samples,
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        try:
            package = build_geometry_package(
                context,
                fixed_object,
                moving_object,
                rigid_collection.name,
                scene.biomechanics_training_translation_mm,
                math.degrees(scene.biomechanics_training_rotation),
                scene.biomechanics_training_seed,
                scene.biomechanics_training_proxy_faces,
            )
        except SchemaError as exc:
            self.report({"ERROR"}, f"Package v2 validation failed: {exc}")
            return {"CANCELLED"}
        package_path = export_directory / "package.json"
        package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

        repository_root = Path(__file__).resolve().parents[1]
        runner_path = repository_root / "tools" / "run_training.ps1"
        command = (
            f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{runner_path}" '
            f'-PackagePath "{export_directory}"'
        )
        command_path.write_text(command + "\n", encoding="utf-8")
        scene.biomechanics_training_summary = f"Exported {total_samples:,} planned samples"
        self.report({"INFO"}, f'Training package exported to "{export_directory}"')
        return {"FINISHED"}


class BIOMECHANICS_OT_start_interaction(bpy.types.Operator):
    """Create a transform target for constrained movement of the selected rigid bone"""

    bl_idname = "biomechanics.start_interaction"
    bl_label = "Start Constrained Move"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        rigid_collection = scene.biomechanics_target_collection
        manipulated = context.active_object
        if rigid_collection is None:
            self.report({"ERROR"}, "Choose the rigid collection")
            return {"CANCELLED"}
        if (
            manipulated is None
            or manipulated.type != "MESH"
            or rigid_collection.objects.get(manipulated.name) is not manipulated
        ):
            self.report({"ERROR"}, "Select one mesh object from the rigid collection")
            return {"CANCELLED"}

        remove_pose_targets(scene)

        target = bpy.data.objects.new(TARGET_NAME, None)
        target.empty_display_type = "ARROWS"
        target.empty_display_size = max(0.01, min(0.1, manipulated.dimensions.length * 0.25))
        target.show_in_front = True
        target.matrix_world = manipulated.matrix_world
        target["biomechanics_pose_target"] = True
        scene.collection.objects.link(target)

        scene.biomechanics_manipulated_object = manipulated
        scene.biomechanics_pose_target = target
        set_neutral_matrix(scene, manipulated.matrix_world)
        follower_count = prepare_attached_bones(scene, manipulated)
        using_training, constraint_message = prepare_trained_constraints(scene, manipulated)

        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        report_level = {"INFO"} if using_training else {"WARNING"}
        follower_message = f"; {follower_count} attached follower" if follower_count else ""
        self.report(
            report_level,
            f"{constraint_message}{follower_message}. Move {TARGET_NAME} with G or R",
        )
        return {"FINISHED"}


class BIOMECHANICS_OT_start_point_interaction(bpy.types.Operator):
    """Drag selected mesh vertices through compatible trained valid poses"""

    bl_idname = "biomechanics.start_point_interaction"
    bl_label = "Drag Selected Bone Point"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        rigid_collection = scene.biomechanics_target_collection
        manipulated = context.active_object
        if (
            rigid_collection is None
            or manipulated is None
            or manipulated.type != "MESH"
            or rigid_collection.objects.get(manipulated.name) is not manipulated
        ):
            self.report({"ERROR"}, "Select one mesh object from the rigid collection")
            return {"CANCELLED"}
        if manipulated.mode != "EDIT":
            self.report({"ERROR"}, "Enter Edit Mode and select one or more vertices")
            return {"CANCELLED"}

        edit_mesh = bmesh.from_edit_mesh(manipulated.data)
        selected_vertices = [vertex for vertex in edit_mesh.verts if vertex.select]
        if not selected_vertices:
            self.report({"ERROR"}, "Select one or more vertices for the drag point")
            return {"CANCELLED"}
        local_point = sum(
            (vertex.co for vertex in selected_vertices),
            Vector((0.0, 0.0, 0.0)),
        ) / len(selected_vertices)

        using_training, constraint_message = prepare_trained_constraints(scene, manipulated)
        if not using_training:
            self.report({"ERROR"}, constraint_message)
            return {"CANCELLED"}

        bpy.ops.object.mode_set(mode="OBJECT")
        remove_pose_targets(scene)
        target = bpy.data.objects.new(POINT_TARGET_NAME, None)
        target.empty_display_type = "SPHERE"
        target.empty_display_size = max(0.002, min(0.02, manipulated.dimensions.length * 0.04))
        target.show_in_front = True
        target.location = manipulated.matrix_world @ local_point
        target["biomechanics_pose_target"] = True
        target["biomechanics_point_target"] = True
        target["biomechanics_local_point"] = tuple(local_point)
        scene.collection.objects.link(target)

        scene.biomechanics_manipulated_object = manipulated
        scene.biomechanics_pose_target = target
        set_neutral_matrix(scene, manipulated.matrix_world)
        prepare_attached_bones(scene, manipulated)
        scene.biomechanics_live_solve = True

        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        self.report(
            {"INFO"},
            f"Dragging {len(selected_vertices)}-vertex point with trained valid poses",
        )
        return {"FINISHED"}


class BIOMECHANICS_OT_reset_interaction(bpy.types.Operator):
    """Return the constrained bone and its target to the neutral pose"""

    bl_idname = "biomechanics.reset_interaction"
    bl_label = "Reset to Neutral"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.biomechanics_pose_target is not None and scene.biomechanics_manipulated_object is not None

    def execute(self, context):
        scene = context.scene
        from .interaction import neutral_matrix

        neutral = neutral_matrix(scene)
        target = scene.biomechanics_pose_target
        if target.get("biomechanics_point_target"):
            local_point = Vector(target["biomechanics_local_point"])
            target.location = neutral @ local_point
        else:
            target.matrix_world = neutral
        scene.biomechanics_manipulated_object.matrix_world = neutral
        solve_scene(scene)
        return {"FINISHED"}


class BIOMECHANICS_OT_stop_interaction(bpy.types.Operator):
    """Commit the current constrained pose and remove its transform target"""

    bl_idname = "biomechanics.stop_interaction"
    bl_label = "Commit and Stop"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.biomechanics_pose_target is not None

    def execute(self, context):
        scene = context.scene
        target = scene.biomechanics_pose_target
        manipulated = scene.biomechanics_manipulated_object
        scene.biomechanics_pose_target = None
        scene.biomechanics_manipulated_object = None
        clear_trained_constraints(scene)
        clear_attached_bones(scene)
        bpy.data.objects.remove(target, do_unlink=True)
        if manipulated is not None:
            manipulated.select_set(True)
            context.view_layer.objects.active = manipulated
        return {"FINISHED"}


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


def _portable_path(path):
    """`bpy.path.relpath` as a //-relative path when possible, falling back to the
    absolute path (e.g. the .blend file and the target path are on different
    Windows drives, which `bpy.path.relpath` cannot express and raises on)."""
    try:
        return bpy.path.relpath(str(path))
    except ValueError:
        return str(path)


class BIOMECHANICS_OT_import_neural_model(bpy.types.Operator):
    """Validate and activate an externally trained ONNX constraint model bundle"""

    bl_idname = "biomechanics.import_neural_model"
    bl_label = "Import Neural Model"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="model_manifest.json", options={"HIDDEN"})

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        manifest_path = Path(bpy.path.abspath(self.filepath))
        try:
            manifest, _onnx_path, reasons = neural_constraints.load_and_check_model(context, manifest_path)
        except neural_constraints.ModelLoadError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if reasons:
            # Never activate a stale/incompatible bundle: leave any previously active model untouched.
            scene.biomechanics_neural_model_status = "Incompatible: " + "; ".join(reasons)
            self.report({"ERROR"}, scene.biomechanics_neural_model_status)
            return {"CANCELLED"}

        scene.biomechanics_neural_model_path = _portable_path(manifest_path)
        scene.biomechanics_neural_model_status = (
            f"Active: {manifest['fixed_object']['name']}/{manifest['moving_object']['name']} "
            f"(range \u00b1{manifest['range']['translation_limit_mm']:g}mm/"
            f"\u00b1{manifest['range']['rotation_limit_degrees']:g}\u00b0, threshold {manifest['validity_threshold']:.3f})"
        )
        self.report({"INFO"}, scene.biomechanics_neural_model_status)
        return {"FINISHED"}


class BIOMECHANICS_OT_clear_neural_model(bpy.types.Operator):
    """Deactivate the currently imported neural constraint model"""

    bl_idname = "biomechanics.clear_neural_model"
    bl_label = "Clear Neural Model"

    def execute(self, context):
        scene = context.scene
        neural_constraints.clear_session_cache()
        scene.biomechanics_neural_model_path = ""
        scene.biomechanics_neural_model_status = ""
        self.report({"INFO"}, "Neural model cleared")
        return {"FINISHED"}