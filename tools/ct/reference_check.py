#!/usr/bin/env python3
"""Compare CT bone meshes against an already-fitted atlas reference, without refitting.

register_atlas.py fits the BodyParts3D reference to a set of CT meshes and writes
<out>/meshes/<bone>_atlas.ply.  This tool measures how far other meshes of the same bones
(e.g. a new segmentation method) sit from those SAME fitted references, so two methods can be
compared against one fixed yardstick:

  mean_mm / p95_mm   distance from points on the CT mesh to its own reference surface
  far_frac           fraction of the CT surface more than --far-mm from its own reference
  misassigned_frac   fraction of the CT surface that is far from its own reference AND within
                     --near-mm of a neighbouring bone's reference (= bone given to the wrong object)
  missing_frac       fraction of the REFERENCE surface more than --far-mm from the CT mesh
                     (bone the segmentation did not capture, e.g. hollow bodies, missing processes)

The reference never shapes the bones; it is a yardstick only.

Usage:
  python tools/ct/reference_check.py --atlas-meshes RUN/atlas_bp3d_check2/meshes \
      --meshes RUN/meshes RUN/nni/meshes --labels old nni --bones T6 T7 T8 ... [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402


def load(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    m.apply_scale(1000.0)  # metres -> mm
    return m


def surface_distance(points, mesh):
    q = trimesh.proximity.ProximityQuery(mesh)
    return np.abs(q.signed_distance(points)) if mesh.is_watertight else np.linalg.norm(
        points - q.on_surface(points)[0], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas-meshes", required=True, help="folder with <bone>_atlas.ply (register_atlas output)")
    ap.add_argument("--meshes", nargs="+", required=True, help="one or more mesh folders to compare")
    ap.add_argument("--labels", nargs="+", help="short names for the mesh folders")
    ap.add_argument("--bones", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--far-mm", type=float, default=4.0)
    ap.add_argument("--near-mm", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write reference_check.json here")
    args = ap.parse_args(argv)
    labels = args.labels or [Path(m).parent.name for m in args.meshes]
    atlas_dir = Path(args.atlas_meshes)
    bones = [pipeline.canonical(b) for b in args.bones]
    refs = {}

    def ref(bone):
        if bone not in refs:
            p = atlas_dir / f"{bone}_atlas.ply"
            refs[bone] = load(p) if p.exists() else None
        return refs[bone]

    rng = np.random.default_rng(args.seed)
    results = {}
    print(f"{'bone':<16} {'set':<6} {'mean':>6} {'p95':>6} {'far%':>6} {'misas%':>7} {'missing%':>9} {'cm3':>7}")
    for bone in bones:
        r = ref(bone)
        if r is None:
            print(f"{bone:<16} no reference mesh")
            continue
        ref_pts, _ = trimesh.sample.sample_surface(r, args.samples, seed=args.seed)
        for folder, label in zip(args.meshes, labels):
            p = Path(folder) / f"{bone}.ply"
            if not p.exists():
                print(f"{bone:<16} {label:<6} missing")
                continue
            m = load(p)
            pts, _ = trimesh.sample.sample_surface(m, args.samples, seed=args.seed)
            d_own = surface_distance(pts, r)
            far = d_own > args.far_mm
            misas = np.zeros(len(pts), dtype=bool)
            for nb in pipeline.neighbours(bone):
                rn = ref(nb)
                if rn is None or not far.any():
                    continue
                d_nb = surface_distance(pts[far], rn)
                idx = np.where(far)[0][d_nb < args.near_mm]
                misas[idx] = True
            d_ref = surface_distance(ref_pts, m)
            vol = float(m.volume) / 1000.0 if m.is_watertight else float("nan")
            row = {"mean_mm": float(d_own.mean()), "p95_mm": float(np.percentile(d_own, 95)),
                   "far_frac": float(far.mean()), "misassigned_frac": float(misas.mean()),
                   "missing_frac": float((d_ref > args.far_mm).mean()), "volume_cm3": vol,
                   "watertight": bool(m.is_watertight)}
            results.setdefault(bone, {})[label] = row
            print(f"{bone:<16} {label:<6} {row['mean_mm']:6.2f} {row['p95_mm']:6.2f} {100 * row['far_frac']:6.1f} "
                  f"{100 * row['misassigned_frac']:7.1f} {100 * row['missing_frac']:9.1f} {vol:7.1f}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reference_check.json").write_text(json.dumps(
            {"atlas_meshes": str(atlas_dir), "meshes": dict(zip(labels, map(str, args.meshes))),
             "far_mm": args.far_mm, "near_mm": args.near_mm, "bones": results}, indent=1))
        print("wrote", out / "reference_check.json")


if __name__ == "__main__":
    main()
