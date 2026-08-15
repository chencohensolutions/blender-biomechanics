"""Diagnose the external Python environment used for neural constraint training.

Run with the training environment's interpreter (not Blender's):
    python tools/check_neural_environment.py
"""

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _addon_import import load_addon_module  # noqa: E402 - path setup must run first

schemas = load_addon_module("schemas")


def _try_import(name):
    try:
        return __import__(name), None
    except Exception as exc:  # noqa: BLE001 - reporting only, never raise here
        return None, str(exc)


def _onnx_parity_smoke_test(torch, onnx, onnxruntime):
    import io

    import numpy as np

    model = torch.nn.Linear(6, 2)
    model.eval()
    sample = torch.randn(4, 6)
    with torch.no_grad():
        expected = model(sample).numpy()

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        sample,
        buffer,
        input_names=["pose"],
        output_names=["prediction"],
        dynamic_axes={"pose": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=17,
    )
    onnx_model = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(onnx_model)

    session = onnxruntime.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pose": sample.numpy().astype(np.float32)})[0]
    return float(np.max(np.abs(expected - actual)))


def main():
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    print(
        "schema versions: "
        f"package={schemas.PACKAGE_SCHEMA_VERSION} "
        f"dataset={schemas.DATASET_SCHEMA_VERSION} "
        f"model_manifest={schemas.MODEL_MANIFEST_SCHEMA_VERSION}"
    )

    modules = {}
    for name in ("numpy", "torch", "onnx", "onnxruntime"):
        module, error = _try_import(name)
        modules[name] = module
        if module is None:
            print(f"{name}: NOT AVAILABLE ({error})")
        else:
            print(f"{name}: {getattr(module, '__version__', 'unknown version')}")

    if modules["torch"] is not None:
        print(f"torch CUDA available: {modules['torch'].cuda.is_available()}")

    if all(modules[name] is not None for name in ("numpy", "torch", "onnx", "onnxruntime")):
        max_difference = _onnx_parity_smoke_test(modules["torch"], modules["onnx"], modules["onnxruntime"])
        print(f"PyTorch vs ONNX Runtime parity (max abs diff): {max_difference:.2e}")
        if max_difference > 1e-4:
            print("WARNING: parity difference exceeds 1e-4; investigate before training")
    else:
        print("Skipping ONNX export/parity smoke test: missing dependency")


if __name__ == "__main__":
    main()
