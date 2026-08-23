"""Pure helpers and process launch support for external neural training."""

import json
import os
import shutil
import subprocess
from pathlib import Path


_ORCHESTRATOR_NAME = "run_neural_training.ps1"
_process = None


def repository_root(module_file=None):
    """Return the repository root containing the add-on and tools directories."""
    source = Path(module_file or __file__).resolve()
    return source.parents[1]


def status_path_for(package_dir):
    return Path(package_dir).resolve() / "neural_training_status.json"


def training_package_directory(training_root, fixed_name, moving_name):
    """Return a stable child directory for one directed fixed/moving bone pair."""
    def safe_name(name):
        return "".join(character if character.isalnum() or character in " _-" else "_" for character in name)

    return Path(training_root).resolve() / f"{safe_name(fixed_name)}__{safe_name(moving_name)}"


def discover_model_manifests(training_root):
    """Find completed bundles in the legacy root and per-pair package directories."""
    root = Path(training_root).resolve()
    candidates = [root / "model" / "model_manifest.json"]
    if root.is_dir():
        candidates.extend(root.glob("*/model/model_manifest.json"))
    return [path.resolve() for path in candidates if path.is_file() and (path.parent / "constraint_model.onnx").is_file()]


def migrate_legacy_model_bundle(training_root):
    """Move one complete legacy root model into its directed pair package.

    The old ``<root>/model`` layout can hold only one trained pair. Never overwrite
    a pair-specific bundle; leaving the legacy bundle in place is safer when both
    exist and lets discovery retain the existing published artifact.
    """
    root = Path(training_root).resolve()
    legacy_model_dir = root / "model"
    legacy_manifest_path = legacy_model_dir / "model_manifest.json"
    if not model_bundle_is_complete(legacy_model_dir):
        return None
    try:
        manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        fixed_name = manifest["fixed_object"]["name"]
        moving_name = manifest["moving_object"]["name"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(fixed_name, str) or not isinstance(moving_name, str):
        return None

    destination_model_dir = training_package_directory(root, fixed_name, moving_name) / "model"
    if destination_model_dir.exists():
        return None
    destination_model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_model_dir), str(destination_model_dir))
    return destination_model_dir / "model_manifest.json"


def read_status(path):
    """Read an orchestrator status file without allowing malformed JSON to escape."""
    status_path = Path(path)
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {"state": "waiting", "error": f"Status file not found: {status_path}"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "unknown", "error": f"Could not read training status: {exc}"}
    if not isinstance(payload, dict):
        return {"state": "unknown", "error": "Training status must be a JSON object"}
    return payload


def write_json_atomic(path, payload):
    """Write JSON to a sibling temporary file and replace the destination atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def build_command(
    package_dir,
    *,
    module_file=None,
    powershell_exe="powershell.exe",
    blender_path=None,
    python_path=None,
    sample_budget=None,
    active_learning=False,
    active_rounds=None,
    max_total_new_samples=None,
):
    """Build the Windows command used by the Blender launch operator."""
    root = repository_root(module_file)
    script = root / "tools" / _ORCHESTRATOR_NAME
    command = [
        powershell_exe,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-PackagePath",
        str(Path(package_dir).resolve()),
    ]
    if blender_path:
        command.extend(("-BlenderPath", str(Path(blender_path).resolve())))
    if python_path:
        command.extend(("-PythonPath", str(Path(python_path).resolve())))
    if sample_budget is not None:
        command.extend(("-SampleBudget", str(sample_budget)))
    if active_learning:
        command.append("-ActiveLearning")
        if active_rounds is not None:
            command.extend(("-ActiveRounds", str(active_rounds)))
        if max_total_new_samples is not None:
            command.extend(("-MaxTotalNewSamples", str(max_total_new_samples)))
    return command


def format_command(command):
    """Return a readable Windows command line while preserving path boundaries."""
    formatted = []
    for value in command:
        text = str(value)
        if any(character.isspace() for character in text) or '"' in text:
            text = '"' + text.replace('"', '\\"') + '"'
        formatted.append(text)
    return " ".join(formatted)


def model_bundle_is_complete(model_dir):
    """A model is publishable only when both manifest and ONNX files exist."""
    directory = Path(model_dir)
    return (
        (directory / "model_manifest.json").is_file()
        and (directory / "constraint_model.onnx").is_file()
    )


def completed_model_manifest(status):
    """Return the published manifest after a successful training run, if present."""
    if status.get("state") != "completed" or status.get("exit_code") != 0:
        return None
    model_dir = status.get("model_dir")
    if not model_dir or not model_bundle_is_complete(model_dir):
        return None
    return Path(model_dir).resolve() / "model_manifest.json"


def launch(command, cwd):
    """Launch the external orchestrator in a visible console without blocking Blender."""
    global _process
    if process_is_running():
        raise RuntimeError("Neural training is already running")
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    _process = subprocess.Popen(
        [str(value) for value in command],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    return _process


def process_is_running():
    return _process is not None and _process.poll() is None


def process_returncode():
    return None if _process is None else _process.poll()


def forget_process():
    """Forget the child handle without terminating it or touching its artifacts."""
    global _process
    _process = None
