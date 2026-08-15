"""Run a packaged two-object training job in background Blender."""

import argparse
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from biomechanics_addon.training import train_two_objects


def parse_arguments():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(arguments)


def matrix_from_values(values):
    return Matrix(tuple(tuple(values[index:index + 4]) for index in range(0, 16, 4)))


def load_training_objects(config_path, config):
    input_path = config_path.parent / config["input_blend"]
    object_names = [config["fixed_object"]["name"], config["moving_object"]["name"]]
    with bpy.data.libraries.load(str(input_path), link=False) as (data_from, data_to):
        missing = [name for name in object_names if name not in data_from.objects]
        if missing:
            raise ValueError(f"Objects missing from training input: {', '.join(missing)}")
        data_to.objects = object_names

    loaded = {obj.name: obj for obj in data_to.objects if obj is not None}
    for obj in loaded.values():
        bpy.context.scene.collection.objects.link(obj)

    fixed_object = loaded[config["fixed_object"]["name"]]
    moving_object = loaded[config["moving_object"]["name"]]
    fixed_object.matrix_world = matrix_from_values(config["fixed_object"]["matrix_world"])
    moving_object.matrix_world = matrix_from_values(config["moving_object"]["matrix_world"])
    bpy.context.view_layer.update()
    return fixed_object, moving_object


def main():
    args = parse_arguments()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bpy.context.scene.unit_settings.scale_length = config["scale_length"]
    fixed_object, moving_object = load_training_objects(config_path, config)
    settings = config["settings"]
    started_at = time.perf_counter()
    last_reported_percent = -1

    def report_progress(completed, total, valid_count):
        nonlocal last_reported_percent
        percent = int(completed * 100 / total)
        if percent == last_reported_percent and completed != total:
            return
        last_reported_percent = percent
        elapsed = time.perf_counter() - started_at
        print(
            f"[training] {completed:,}/{total:,} ({percent:3d}%) | "
            f"valid {valid_count:,} | elapsed {elapsed:.1f}s",
            flush=True,
        )

    print(
        f"[training] fixed={fixed_object.name} moving={moving_object.name} "
        f"planned={config['planned_samples']:,}",
        flush=True,
    )
    _text, result = train_two_objects(
        bpy.context,
        fixed_object,
        moving_object,
        settings["translation_limit_mm"],
        settings["rotation_limit_degrees"],
        settings["translation_steps"],
        settings["rotation_steps"],
        settings["minimum_clearance_mm"],
        settings["maximum_proxy_faces"],
        maximum_samples=settings["maximum_samples"],
        progress_callback=report_progress,
        collection_name=config["collection"],
        write_text=False,
    )
    result["execution"] = {
        "mode": "background_blender",
        "elapsed_seconds": time.perf_counter() - started_at,
        "complete": True,
    }
    output_path = config_path.parent / config["output_json"]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"[training] complete: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[training] ERROR: {exc}", file=sys.stderr, flush=True)
        raise