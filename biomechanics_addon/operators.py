import bpy  # pyright: ignore[reportMissingImports]
import json
import math
from pathlib import Path
from mathutils import Matrix, Vector  # pyright: ignore[reportMissingImports]

from .alignment import base_object_name, mesh_alignment
from .interaction import (
    TARGET_NAME,
    clear_attached_bones,
    clear_interaction_state,
    prepare_attached_bones,
    set_neutral_matrix,
    solve_scene,
    remove_pose_targets,
)
from . import neural_constraints
from . import neural_training
from .properties import refresh_transparency_overlay
from .schemas import SchemaError
from .neural_geometry import (
    _clearance,
    _full_mesh_geometry,
    _world_bvh,
    build_geometry_package,
)


NEURAL_MINIMUM_CLEARANCE_MM = 0.5
NEURAL_PROXY_MAX_FACES = 2000
TRAINING_CENTER_HANDLE_NAME = "Biomechanics Training Center"


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


class BIOMECHANICS_OT_copy_interaction_info(bpy.types.Operator):
    """Copy the visible constrained-interaction information to the clipboard"""

    bl_idname = "biomechanics.copy_interaction_info"
    bl_label = "Copy Panel Info"

    @classmethod
    def poll(cls, context):
        return context.scene.biomechanics_pose_target is not None

    def execute(self, context):
        scene = context.scene
        manipulated = scene.biomechanics_manipulated_object
        model_entries = [
            entry
            for entry in scene.biomechanics_neural_models
            if manipulated is not None and entry.moving_object_name == manipulated.name
        ]
        lines = [
            "Biomechanics - Constrained Interaction",
            f"Moving bone: {manipulated.name if manipulated is not None else 'Missing bone'}",
            f"Live solve: {'On' if scene.biomechanics_live_solve else 'Off'}",
            f"Move attached bones: {'On' if scene.biomechanics_move_attached_bones else 'Off'}",
            f"Translation limit: {scene.biomechanics_translation_limit_mm:g} mm",
            f"Rotation limit: {math.degrees(scene.biomechanics_rotation_limit):g} degrees",
        ]
        if model_entries:
            for entry in model_entries:
                lines.append(
                    f"Neural model: {entry.fixed_object_name} -> {entry.moving_object_name}"
                )
        else:
            lines.append("Neural model: none for this moving bone")
        if scene.biomechanics_neural_model_status:
            lines.append(f"Neural status: {scene.biomechanics_neural_model_status}")
        lines.extend(
            (
                "Controls: G moves; R rotates",
                "Reset to Neutral returns to the starting pose",
                "Commit and Stop keeps the constrained pose",
            )
        )
        context.window_manager.clipboard = "\n".join(lines)
        self.report({"INFO"}, "Constrained interaction info copied to clipboard")
        return {"FINISHED"}


class BIOMECHANICS_OT_center_cursor_on_trained_pose(bpy.types.Operator):
    """Place the 3D cursor at the active model's neutral moving-pose center."""

    bl_idname = "biomechanics.center_cursor_on_trained_pose"
    bl_label = "Center Cursor on Trained Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        manipulated = scene.biomechanics_manipulated_object
        return (
            scene.biomechanics_pose_target is not None
            and manipulated is not None
            and any(
                entry.moving_object_name == manipulated.name
                for entry in scene.biomechanics_neural_models
            )
        )

    def execute(self, context):
        scene = context.scene
        manipulated = scene.biomechanics_manipulated_object
        entry = next(
            (
                item
                for item in scene.biomechanics_neural_models
                if item.moving_object_name == manipulated.name
            ),
            None,
        )
        if entry is None:
            self.report({"ERROR"}, "No trained neural model controls this bone")
            return {"CANCELLED"}
        try:
            manifest = neural_constraints.load_manifest(bpy.path.abspath(entry.manifest_path))
        except neural_constraints.ModelLoadError as exc:
            self.report({"ERROR"}, f"Could not read trained pose center: {exc}")
            return {"CANCELLED"}

        neutral_matrix = _matrix_from_values(manifest["neutral_moving_matrix"])
        scene.cursor.location = neutral_matrix.translation
        self.report({"INFO"}, f"3D cursor centered on trained {manipulated.name} pose")
        return {"FINISHED"}


class BIOMECHANICS_OT_apply_training_center_to_cursor(bpy.types.Operator):
    """Place the 3D cursor at the assembled X/Y/Z training center."""

    bl_idname = "biomechanics.apply_training_center_to_cursor"
    bl_label = "Apply Training Center to Cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        handle = _training_center_handle(scene)
        if handle is not None:
            scene.biomechanics_training_rotation_center = handle.location
        scene.cursor.location = scene.biomechanics_training_rotation_center
        self.report({"INFO"}, "3D cursor set to the training rotation center")
        return {"FINISHED"}


def _training_center_handle(scene):
    handle = bpy.data.objects.get(TRAINING_CENTER_HANDLE_NAME)
    return handle if handle is not None and handle.get("biomechanics_training_center") else None


class BIOMECHANICS_OT_edit_training_center(bpy.types.Operator):
    """Create or select a draggable handle for the training rotation center."""

    bl_idname = "biomechanics.edit_training_center"
    bl_label = "Drag Training Center"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        handle = _training_center_handle(scene)
        if handle is None:
            handle = bpy.data.objects.new(TRAINING_CENTER_HANDLE_NAME, None)
            handle.empty_display_type = "SPHERE"
            handle.empty_display_size = 0.03
            handle.show_in_front = True
            handle["biomechanics_training_center"] = True
            scene.collection.objects.link(handle)
        handle.location = scene.biomechanics_training_rotation_center
        bpy.ops.object.select_all(action="DESELECT")
        handle.select_set(True)
        context.view_layer.objects.active = handle
        self.report({"INFO"}, "Drag the selected training center handle with G, then export training")
        return {"FINISHED"}


def _matrix_values(matrix):
    return [value for row in matrix for value in row]


def _matrix_from_values(values):
    return Matrix(tuple(tuple(values[index:index + 4]) for index in range(0, 16, 4)))


def _use_cursor_transform_pivot(scene, manifest):
    """Align viewport rotation controls with the pivot used to label this model."""
    scene.cursor.location = Vector(manifest["rotation_center_world"])
    scene.tool_settings.transform_pivot_point = "CURSOR"


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


_neural_training_timer_registered = False


def _neural_training_status_text(status):
    state = status.get("state", "unknown")
    stage = status.get("current_stage")
    error = status.get("error")
    if error:
        return f"{state}: {error}"
    if stage:
        return f"{state}: {stage}"
    return state


def _register_neural_model(scene, manifest_path, manifest):
    """Add or replace one model registry entry for the manifest's directed pair."""
    fixed_name = manifest["fixed_object"]["name"]
    moving_name = manifest["moving_object"]["name"]
    for entry in scene.biomechanics_neural_models:
        if entry.fixed_object_name == fixed_name and entry.moving_object_name == moving_name:
            entry.manifest_path = _portable_path(manifest_path)
            break
    else:
        entry = scene.biomechanics_neural_models.add()
        entry.fixed_object_name = fixed_name
        entry.moving_object_name = moving_name
        entry.manifest_path = _portable_path(manifest_path)
    # Preserve the former property as a latest-model pointer for old .blend files and UI state.
    scene.biomechanics_neural_model_path = entry.manifest_path


def _remove_neural_model(scene, fixed_name, moving_name):
    """Remove a registry entry that is no longer compatible with this scene."""
    for index in range(len(scene.biomechanics_neural_models) - 1, -1, -1):
        entry = scene.biomechanics_neural_models[index]
        if entry.fixed_object_name == fixed_name and entry.moving_object_name == moving_name:
            scene.biomechanics_neural_models.remove(index)


def _discover_neural_models(scene):
    """Restore all complete model bundles beneath the configured training root."""
    training_root = bpy.path.abspath(scene.biomechanics_training_export_directory)
    neural_training.migrate_legacy_model_bundle(training_root)
    valid_pairs = set()
    for manifest_path in neural_training.discover_model_manifests(training_root):
        try:
            manifest = neural_constraints.load_manifest(manifest_path)
            fixed_name = manifest["fixed_object"]["name"]
            moving_name = manifest["moving_object"]["name"]
            rigid_collection = scene.biomechanics_target_collection
            fixed_object = rigid_collection.objects.get(fixed_name) if rigid_collection else None
            moving_object = rigid_collection.objects.get(moving_name) if rigid_collection else None
            if fixed_object is None or moving_object is None:
                raise neural_constraints.ModelLoadError("trained bones are unavailable in Rigid")
            _manifest, _onnx_path, reasons = neural_constraints.load_and_check_model(
                bpy.context, manifest_path, fixed_object, moving_object
            )
        except neural_constraints.ModelLoadError:
            continue
        if reasons:
            pair = (fixed_name, moving_name)
            if pair not in valid_pairs:
                _remove_neural_model(scene, fixed_name, moving_name)
                scene.biomechanics_neural_model_status = (
                    f"Model {fixed_name}/{moving_name} unavailable: " + "; ".join(reasons)
                )
            continue
        _register_neural_model(scene, manifest_path, manifest)
        valid_pairs.add((fixed_name, moving_name))
        scene.biomechanics_neural_model_status = (
            f"Active trained model: {fixed_name}/{moving_name}"
        )


def _activate_completed_neural_model(context, scene, status):
    """Activate the model published by this scene's successful one-click run."""
    manifest_path = neural_training.completed_model_manifest(status)
    if manifest_path is None:
        return False
    try:
        manifest, _onnx_path, reasons = neural_constraints.load_and_check_model(context, manifest_path)
    except neural_constraints.ModelLoadError as exc:
        scene.biomechanics_neural_model_status = f"Training completed, but model unavailable: {exc}"
        return False
    if reasons:
        scene.biomechanics_neural_model_status = "Training completed, but model incompatible: " + "; ".join(reasons)
        return False
    _register_neural_model(scene, manifest_path, manifest)
    scene.biomechanics_neural_model_status = (
        f"Active trained model: {manifest['fixed_object']['name']}/{manifest['moving_object']['name']}"
    )
    return True


def _poll_neural_training_status():
    global _neural_training_timer_registered
    process_running = neural_training.process_is_running()
    for scene in bpy.data.scenes:
        status_path = scene.biomechanics_neural_training_status_path
        if not status_path:
            continue
        status = neural_training.read_status(bpy.path.abspath(status_path))
        scene.biomechanics_neural_training_active = process_running
        if process_running and status.get("state") not in {"running", "waiting"}:
            scene.biomechanics_neural_training_status = "running: launching"
        else:
            scene.biomechanics_neural_training_status = _neural_training_status_text(status)
        if not process_running and status.get("state") == "running":
            returncode = neural_training.process_returncode()
            scene.biomechanics_neural_training_status = (
                f"failed: orchestrator exited with code {returncode} before final status"
            )
        elif not process_running and status.get("state") == "completed" and bpy.context.scene is scene:
            if _activate_completed_neural_model(bpy.context, scene, status):
                scene.biomechanics_neural_training_status = "completed: model activated"
    if process_running:
        return 0.5
    _neural_training_timer_registered = False
    return None


def _ensure_neural_training_timer():
    global _neural_training_timer_registered
    if _neural_training_timer_registered:
        return
    bpy.app.timers.register(_poll_neural_training_status, first_interval=0.5)
    _neural_training_timer_registered = True


def shutdown_neural_training():
    """Forget external process state during reload without terminating or deleting artifacts."""
    global _neural_training_timer_registered
    if _neural_training_timer_registered and bpy.app.timers.is_registered(_poll_neural_training_status):
        bpy.app.timers.unregister(_poll_neural_training_status)
    _neural_training_timer_registered = False
    neural_training.forget_process()


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

        export_root = Path(bpy.path.abspath(scene.biomechanics_training_export_directory))
        export_directory = neural_training.training_package_directory(
            export_root,
            fixed_object.name,
            moving_object.name,
        )
        try:
            export_directory.mkdir(parents=True, exist_ok=True)
            probe_path = export_directory / ".biomechanics_write_check"
            probe_path.write_text("", encoding="utf-8")
            probe_path.unlink()
        except OSError as exc:
            self.report({"ERROR"}, f"Export folder is not writable: {exc}")
            return {"CANCELLED"}
        input_path = export_directory / "training_input.blend"
        command_path = export_directory / "RUN_COMMAND.txt"
        if input_path.exists():
            input_path.unlink()

        bpy.data.libraries.write(
            str(input_path),
            {fixed_object, moving_object},
            compress=True,
        )

        try:
            handle = _training_center_handle(scene)
            if handle is not None:
                scene.biomechanics_training_rotation_center = handle.location
                scene.cursor.location = handle.location
            rotation_domain_degrees = (
                scene.biomechanics_neural_rotation_domain_degrees
                if scene.biomechanics_neural_limit_rotation_domain
                else 180.0
            )
            package = build_geometry_package(
                context,
                fixed_object,
                moving_object,
                rigid_collection.name,
                scene.biomechanics_neural_translation_domain_mm,
                rotation_domain_degrees,
                scene.biomechanics_training_seed,
                NEURAL_PROXY_MAX_FACES,
                rotation_center_world=scene.biomechanics_training_rotation_center,
            )
        except SchemaError as exc:
            self.report({"ERROR"}, f"Package v2 validation failed: {exc}")
            return {"CANCELLED"}
        package_path = export_directory / "package.json"
        package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

        repository_root = Path(__file__).resolve().parents[1]
        orchestrator_path = repository_root / "tools" / "run_neural_training.ps1"
        generator_path = repository_root / "tools" / "generate_neural_dataset.py"
        trainer_path = repository_root / "tools" / "train_constraint_model.py"
        active_learning_path = repository_root / "tools" / "active_learning_loop.py"
        external_python = repository_root / ".venv" / "Scripts" / "python.exe"
        command = (
            "# One-click resumable neural training\n"
            f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{orchestrator_path}" '
            f'-PackagePath "{export_directory}"\n\n'
            "# External neural pipeline (run from the workspace .venv)\n"
            f'& "{bpy.app.binary_path}" --background --factory-startup --python "{generator_path}" '
            f'-- --package-dir "{export_directory}"\n'
            f'& "{external_python}" "{trainer_path}" --package-dir "{export_directory}"\n\n'
            "# Optional bounded active learning and final retraining\n"
            f'& "{external_python}" "{active_learning_path}" --package-dir "{export_directory}"'
        )
        command_path.write_text(command + "\n", encoding="utf-8")
        self.report({"INFO"}, f'Neural training package exported to "{export_directory}"')
        return {"FINISHED"}


class BIOMECHANICS_OT_start_neural_training(bpy.types.Operator):
    """Export the current pair and launch the resumable external neural pipeline."""

    bl_idname = "biomechanics.start_neural_training"
    bl_label = "Start One-Click Neural Training"

    def execute(self, context):
        scene = context.scene
        if neural_training.process_is_running() or scene.biomechanics_neural_training_active:
            self.report({"WARNING"}, "Neural training is already running")
            return {"CANCELLED"}
        try:
            _rigid_collection, fixed_object, moving_object = _training_objects(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        export_result = bpy.ops.biomechanics.export_training_package()
        if "FINISHED" not in export_result:
            return {"CANCELLED"}

        package_dir = neural_training.training_package_directory(
            bpy.path.abspath(scene.biomechanics_training_export_directory),
            fixed_object.name,
            moving_object.name,
        )
        try:
            package = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
            exported_pair = (
                package["fixed_object"]["name"],
                package["moving_object"]["name"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Could not verify exported neural training pair: {exc}")
            return {"CANCELLED"}
        expected_pair = (fixed_object.name, moving_object.name)
        if exported_pair != expected_pair:
            self.report(
                {"ERROR"},
                f"Exported pair is {exported_pair[0]} -> {exported_pair[1]}, expected "
                f"{expected_pair[0]} -> {expected_pair[1]}",
            )
            return {"CANCELLED"}
        status_path = neural_training.status_path_for(package_dir)
        command = neural_training.build_command(
            package_dir,
            module_file=__file__,
            sample_budget=scene.biomechanics_neural_training_sample_budget,
            active_learning=scene.biomechanics_neural_training_active_learning,
            active_rounds=scene.biomechanics_neural_training_active_rounds,
            max_total_new_samples=scene.biomechanics_neural_training_max_new_samples,
        )
        try:
            neural_training.launch(command, Path(__file__).resolve().parents[1])
        except (OSError, RuntimeError) as exc:
            self.report({"ERROR"}, f"Could not launch neural training: {exc}")
            return {"CANCELLED"}

        scene.biomechanics_neural_training_status_path = _portable_path(status_path)
        scene.biomechanics_neural_training_pair = f"{fixed_object.name} -> {moving_object.name}"
        scene.biomechanics_neural_training_status = "running: launching"
        scene.biomechanics_neural_training_active = True
        _ensure_neural_training_timer()
        self.report({"INFO"}, f"Neural training started: {neural_training.format_command(command)}")
        return {"FINISHED"}


class BIOMECHANICS_OT_start_interaction(bpy.types.Operator):
    """Create a transform target for constrained movement of the selected rigid bone"""

    bl_idname = "biomechanics.start_interaction"
    bl_label = "Start Constrained Move"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not context.scene.biomechanics_neural_training_active

    def execute(self, context):
        scene = context.scene
        if scene.biomechanics_neural_training_active:
            self.report({"WARNING"}, "Interaction is disabled while neural training is running")
            return {"CANCELLED"}
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

        if scene.biomechanics_neural_training_status_path:
            status = neural_training.read_status(
                bpy.path.abspath(scene.biomechanics_neural_training_status_path)
            )
            _activate_completed_neural_model(context, scene, status)

        _discover_neural_models(scene)

        if scene.biomechanics_neural_models and not any(
            entry.moving_object_name == manipulated.name for entry in scene.biomechanics_neural_models
        ):
            scene.biomechanics_neural_model_status = (
                f"No trained neural model moves {manipulated.name}; movement is unrestricted"
            )

        remove_pose_targets(scene)
        clear_interaction_state(scene)

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
        model_entry = next(
            (
                entry
                for entry in scene.biomechanics_neural_models
                if entry.moving_object_name == manipulated.name
            ),
            None,
        )
        if model_entry is not None:
            try:
                manifest = neural_constraints.load_manifest(bpy.path.abspath(model_entry.manifest_path))
                _use_cursor_transform_pivot(scene, manifest)
            except neural_constraints.ModelLoadError:
                pass
        follower_count = prepare_attached_bones(scene, manipulated)

        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        follower_message = f"; {follower_count} attached follower" if follower_count else ""
        self.report(
            {"INFO"},
            f"Neural control applies only when the active model was trained with this bone as moving"
            f"{follower_message}. "
            f"Move the selected bone or {TARGET_NAME} with G or R",
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
        clear_attached_bones(scene)
        clear_interaction_state(scene)
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
        if selected_object is None or selected_object.type != "MESH":
            self.report({"ERROR"}, "Select one mesh object from the original or rigid collection")
            return {"CANCELLED"}
        if source_collection.objects.get(selected_object.name) is not selected_object:
            if target_collection.objects.get(selected_object.name) is not selected_object:
                self.report({"ERROR"}, "Select one mesh object from the original or rigid collection")
                return {"CANCELLED"}
            # A Rigid selection stands for the same bone in Original.
            rigid_name = base_object_name(selected_object.name)
            original_matches = [
                obj
                for obj in source_collection.objects
                if obj.type == "MESH" and base_object_name(obj.name) == rigid_name
            ]
            if not original_matches:
                self.report({"ERROR"}, f'No original pair found for "{selected_object.name}"')
                return {"CANCELLED"}
            if len(original_matches) > 1:
                self.report({"ERROR"}, f'Multiple original pairs found for "{selected_object.name}"')
                return {"CANCELLED"}
            selected_object = original_matches[0]

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
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # pyright: ignore[reportInvalidTypeForm]
    filter_glob: bpy.props.StringProperty(default="model_manifest.json", options={"HIDDEN"})  # pyright: ignore[reportInvalidTypeForm]

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

        _register_neural_model(scene, manifest_path, manifest)
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
        scene.biomechanics_neural_models.clear()
        scene.biomechanics_neural_model_path = ""
        scene.biomechanics_neural_model_status = ""
        self.report({"INFO"}, "Neural model cleared")
        return {"FINISHED"}


class BIOMECHANICS_OT_validate_neural_pose(bpy.types.Operator):
    """Run an authoritative full-mesh check for the current fixed/moving pose."""

    bl_idname = "biomechanics.validate_neural_pose"
    bl_label = "Validate Current Pose"

    def execute(self, context):
        scene = context.scene
        fixed_object = scene.biomechanics_training_fixed_object
        moving_object = scene.biomechanics_training_moving_object
        if fixed_object is None or moving_object is None:
            self.report({"ERROR"}, "Choose fixed and moving bones before validation")
            return {"CANCELLED"}
        if fixed_object.type != "MESH" or moving_object.type != "MESH":
            self.report({"ERROR"}, "Fixed and moving bones must be meshes")
            return {"CANCELLED"}

        fixed_vertices, fixed_polygons = _full_mesh_geometry(context, fixed_object)
        moving_vertices, moving_polygons = _full_mesh_geometry(context, moving_object)
        fixed_world_vertices, fixed_bvh = _world_bvh(
            fixed_vertices, fixed_polygons, fixed_object.matrix_world
        )
        if scene.biomechanics_neural_model_path:
            try:
                neutral_moving = neural_constraints.load_manifest(
                    bpy.path.abspath(scene.biomechanics_neural_model_path)
                )["neutral_moving_matrix"]
            except neural_constraints.ModelLoadError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        else:
            neutral_moving = moving_object.matrix_world
        neutral_moving_matrix = (
            Matrix(tuple(tuple(neutral_moving[index : index + 4]) for index in range(0, 16, 4)))
            if isinstance(neutral_moving, list)
            else neutral_moving
        )
        neutral_vertices, neutral_bvh = _world_bvh(
            moving_vertices, moving_polygons, neutral_moving_matrix
        )
        baseline_overlap_pairs = len(fixed_bvh.overlap(neutral_bvh))
        current_world_vertices, current_bvh = _world_bvh(
            moving_vertices, moving_polygons, moving_object.matrix_world
        )
        overlap_pairs = len(fixed_bvh.overlap(current_bvh))
        clearance = 0.0
        if overlap_pairs == 0:
            clearance = _clearance(fixed_world_vertices, fixed_bvh, current_world_vertices, current_bvh)
        elif overlap_pairs > baseline_overlap_pairs:
            clearance = -_clearance(fixed_world_vertices, fixed_bvh, current_world_vertices, current_bvh)
        valid = overlap_pairs <= baseline_overlap_pairs and (overlap_pairs > 0 or clearance >= 0.0)
        scale = (scene.unit_settings.scale_length or 1.0) * 1000.0
        clearance_mm = clearance * scale
        scene.biomechanics_neural_model_status = (
            f"Exact full-mesh: {'valid' if valid else 'invalid'}, "
            f"clearance {clearance_mm:.2f} mm, overlap pairs {overlap_pairs}"
        )
        self.report({"INFO" if valid else "WARNING"}, scene.biomechanics_neural_model_status)
        return {"FINISHED"}