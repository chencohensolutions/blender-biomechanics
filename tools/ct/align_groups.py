#!/usr/bin/env python
"""Bring the bones of a secondary acquisition group into the frame of the primary one.

A scan with several acquisitions (pelvis in one breath-hold, chest in another) places each
group in its own patient position.  Bones are never moved individually; instead the
vertebrae that both acquisitions cover are extracted from BOTH groups, one rigid transform
(no scale) is fitted from the secondary group's copies onto the primary group's copies, and
that single transform is applied to every bone that came from the secondary group.  The
primary group (first --match of the pipeline run) is the reference and is never touched.

  python tools/ct/align_groups.py RUN_DIR [--primary Delayed] [--blender] [--collection bones]

Outputs in RUN_DIR: align/<group>/ (the shared bones extracted from the other group),
meshes/unaligned/ (the original secondary-group meshes), rewritten meshes/<bone>.ply/.stl,
"aligned_transform" in each moved bone's JSON, and align/alignment.json with the transform,
the bones used and the residuals before and after.  Moved Blender objects carry
"ct_aligned_to" and "ct_align_rms_mm".
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402
import extract_bone  # noqa: E402
import fuse_planes  # noqa: E402
from pipeline import label_bounds, neighbours  # noqa: E402


def shared_bones(run, manifest, primary, secondary, margin=5.0):
    """Vertebrae whose guide label is non-empty and fully inside both groups' volumes."""
    out = []
    for bone in manifest["outputs"]:
        if not bone.startswith("vertebrae_"):
            continue
        ok = True
        for g in (primary, secondary):
            path = run / "totalseg" / g / f"{bone}.nii.gz"
            if not path.exists():
                ok = False
                break
            b = label_bounds(path)
            if b is None:
                ok = False
                break
            lo, hi, _ = b
            glo, ghi = (np.array(v) for v in manifest["groups"][g]["bounds_lps"])
            if not (np.all(lo - margin >= glo) and np.all(hi + margin <= ghi)):
                ok = False
                break
        if ok:
            out.append(bone)
    return out


def extract_from_group(run, manifest, bone, group, out_dir, atlas_labels=None):
    g = manifest["groups"][group]
    fused = out_dir / "fused" / f"{bone}_fused.nrrd"
    if not fused.exists():
        fuse_planes.main(["--inputs", *g["volumes"], "--roi-from-mask", str(run / "totalseg" / group / f"{bone}.nii.gz"),
                          "--pad", "20", "--out", str(fused)])
    guides = [str(run / "totalseg" / group / f"{c}.nii.gz") for c in [bone, *neighbours(bone)]
              if (run / "totalseg" / group / f"{c}.nii.gz").exists()]
    args = ["--fused", str(fused), "--target", bone, "--labels", *guides, "--out-dir", str(out_dir / "meshes")]
    if atlas_labels:
        args += ["--atlas-labels", str(atlas_labels)]
    extract_bone.main(args)
    return out_dir / "meshes" / f"{bone}.ply"


def rigid_fit(src_meshes, dst_meshes, samples=6000, iterations=60):
    """One rigid transform (metres) mapping the source meshes onto the destination meshes."""
    src_pts = np.vstack([trimesh.sample.sample_surface(m, samples)[0] for m in src_meshes])
    dst = trimesh.util.concatenate(dst_meshes)
    matrix, transformed, cost = trimesh.registration.icp(src_pts, dst, initial=np.eye(4), threshold=1e-7,
                                                         max_iterations=iterations, scale=False)
    return matrix, cost


def mean_distance(points, mesh):
    """Mean point-to-surface distance using vertices (rtree-free approximation)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(mesh.vertices).query(points)
    return float(d.mean()), float(np.percentile(d, 95))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--primary", default=None, help="reference group (default: the run's first group)")
    ap.add_argument("--atlas-labels", default=None, help="fitted-atlas labels for the shared-bone extraction")
    ap.add_argument("--blender", action="store_true", help="re-import the moved bones into the running Blender")
    ap.add_argument("--collection", default="bones")
    args = ap.parse_args(argv)

    run = Path(args.run)
    manifest = json.loads((run / "run.json").read_text())
    groups = list(manifest["groups"])
    primary = args.primary or groups[0]
    report = {"primary": primary, "groups": {}}
    t0 = time.time()
    for secondary in [g for g in groups if g != primary]:
        bones = shared_bones(run, manifest, primary, secondary)
        print(f"group {secondary} -> {primary}: shared vertebrae {bones}")
        if len(bones) < 1:
            print("!! no shared bone, cannot align this group")
            continue
        src, dst = [], []
        for bone in bones:
            own = manifest["outputs"][bone]["group"]
            other = secondary if own == primary else primary
            path = extract_from_group(run, manifest, bone, other, run / "align" / other, args.atlas_labels)
            own_mesh = trimesh.load(manifest["outputs"][bone]["ply"])
            other_mesh = trimesh.load(path)
            if own == primary:
                src.append(other_mesh); dst.append(own_mesh)  # secondary copy -> primary copy
            else:
                src.append(own_mesh); dst.append(other_mesh)
        pts = np.vstack([trimesh.sample.sample_surface(m, 6000)[0] for m in src])
        dst_all = trimesh.util.concatenate(dst)
        before = mean_distance(pts, dst_all)
        matrix, cost = rigid_fit(src, dst)
        after = mean_distance(trimesh.transform_points(pts, matrix), dst_all)
        angle = np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1) / 2, -1, 1)))
        shift = matrix[:3, 3] * 1000
        print(f"  rigid fit: rotation {angle:.2f} deg, translation {np.round(shift, 2).tolist()} mm, "
              f"mean surface distance {before[0]*1000:.2f} -> {after[0]*1000:.2f} mm (p95 {after[1]*1000:.2f})")
        moved = []
        unaligned_dir = run / "meshes" / "unaligned"
        unaligned_dir.mkdir(exist_ok=True)
        for bone, files in manifest["outputs"].items():
            if files["group"] != secondary:
                continue
            ply = Path(files["ply"])
            if ply.exists() and not (unaligned_dir / ply.name).exists():
                m = trimesh.load(ply)
                m.export(unaligned_dir / ply.name)
                m.apply_transform(matrix)
                m.export(ply)
                m.export(ply.with_suffix(".stl"))
                info = json.loads(Path(files["json"]).read_text())
                info["aligned_transform"] = matrix.tolist()
                info["aligned_to"] = primary
                Path(files["json"]).write_text(json.dumps(info, indent=1))
                moved.append(bone)
        report["groups"][secondary] = {"shared_bones": bones, "matrix_m": matrix.tolist(), "rotation_deg": float(angle),
                                       "translation_mm": shift.tolist(), "mean_mm_before": before[0] * 1000,
                                       "mean_mm_after": after[0] * 1000, "p95_mm_after": after[1] * 1000,
                                       "moved": moved}
        print(f"  moved {len(moved)} bones of group {secondary}; originals in {unaligned_dir}")
        if args.blender:
            for bone in moved:
                files = manifest["outputs"][bone]
                try:
                    blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                        "path": files["ply"], "name": bone, "collection": args.collection, "decimate": 0.0,
                        "props": {"ct_group": secondary, "ct_aligned_to": primary,
                                  "ct_align_rms_mm": round(after[0] * 1000, 2),
                                  "ct_run": manifest.get("run_tag", "")}})
                except OSError as exc:
                    print(f"!! Blender not reachable ({exc})")
                    break
    (run / "align").mkdir(exist_ok=True)
    (run / "align" / "alignment.json").write_text(json.dumps(report, indent=1))
    print(f"done in {time.time() - t0:.0f}s; report {run / 'align' / 'alignment.json'}")


if __name__ == "__main__":
    main()
