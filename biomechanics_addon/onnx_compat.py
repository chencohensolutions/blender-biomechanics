"""Blender-side ONNX Runtime compatibility diagnostics for neural constraints.

Reports whether the embedded Blender Python can import `onnxruntime` so the
UI can disable neural controls and fall back to format-v1 discrete/scalar
behavior when it cannot. Never raises; unavailability is a reported state.
"""

import platform
import sys

import bpy


def check_onnx_runtime():
    report = {
        "blender_version": bpy.app.version_string,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "onnxruntime_available": False,
        "onnxruntime_version": None,
        "execution_providers": [],
        "error": None,
    }
    try:
        import onnxruntime
    except Exception as exc:
        report["error"] = str(exc)
        return report

    report["onnxruntime_available"] = True
    report["onnxruntime_version"] = getattr(onnxruntime, "__version__", "unknown")
    try:
        report["execution_providers"] = list(onnxruntime.get_available_providers())
    except Exception as exc:
        report["error"] = f"imported but get_available_providers() failed: {exc}"
    return report


def user_modules_directory():
    """Per-user Blender module directory where onnxruntime should be installed."""
    return bpy.utils.user_resource("SCRIPTS", path="modules", create=True)


def format_status(report):
    if not report["onnxruntime_available"]:
        return f"ONNX Runtime unavailable ({report['error']}); neural controls disabled"
    providers = ", ".join(report["execution_providers"]) or "none reported"
    return f"ONNX Runtime {report['onnxruntime_version']} ready; providers: {providers}"


class BIOMECHANICS_OT_check_onnx_compatibility(bpy.types.Operator):
    """Report whether ONNX Runtime is importable inside Blender's embedded Python"""

    bl_idname = "biomechanics.check_onnx_compatibility"
    bl_label = "Check Neural Model Compatibility"

    def execute(self, context):
        report = check_onnx_runtime()
        context.scene.biomechanics_onnx_status = format_status(report)
        level = {"INFO"} if report["onnxruntime_available"] else {"WARNING"}
        self.report(level, context.scene.biomechanics_onnx_status)
        return {"FINISHED"}
