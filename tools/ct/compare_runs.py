#!/usr/bin/env python
"""Compare the bone meshes of two CT-to-mesh runs (two scans, or two parameter settings).

A run directory holds `<run>/meshes/<bone>.ply` (+ .stl, .json) as written by
extract_bone.py: binary PLY in metres, Blender/LPS axes (x = patient left, y = posterior,
z = superior), scanner origin kept.  Meshes of the same scan therefore share one frame;
meshes of two different scans share a frame only if the scans were registered.

For every bone present in both runs (or the --bones subset) mesh A and mesh B are loaded,
converted to millimetres, and compared:

  volume_cm3      from <bone>.json if present, else trimesh volume (watertight meshes only);
                  dVol% uses trimesh volumes for both when A and B got their volumes from
                  different sources (the json value can differ slightly from the mesh's)
  size_mm         axis-aligned extent along x / y / z (LPS)
  centroid_mm     volume centroid (surface-area centroid if not watertight), scanner frame
  centroid_shift  B centroid - A centroid in mm (vector and norm): how far the bone sits
                  from where it was, in scanner coordinates
  unaligned       distance from ~20000 points sampled on B's surface to A's surface, no
                  alignment: mean / median / p95 / max in mm.  Mixes "the bone moved"
                  (patient positioning, different scan) with "the bone changed shape"
  aligned         the same statistics after a rigid ICP fit of B onto A (start: centroid
                  match).  Positioning is removed, so what remains is shape change
                  (growth, remodelling, osteophytes, a different iso-level or segmentation)
  icp             rotation angle (deg) of the fit, and the motion of B's centroid under it
                  (mm); the full 4x4 matrix (B mm -> A mm) is in compare.json

Same mesh in both runs: everything ~0.  A rigid move: large unaligned, ~0 aligned, ICP
angle/motion = the move.  A shape change: aligned stays large (p95/max show where).
ICP is plain point-to-point (no outlier trimming): a large local change (fracture, a
missing process) can bias the fit; then trust the unaligned numbers in a registered frame.

Outputs (default --out <RUN_B>/compare_vs_<basename RUN_A>/):
  <bone>_diff.ply   mesh B in Blender metres, unaligned (drops into the same Blender scene),
                    per-vertex colour = unaligned distance to A on a fixed scale:
                    blue 0 mm -> white 1 mm -> red >= 3 mm (same scale for every run)
  compare.json      all numbers per bone, plus the colour scale

Surface distance uses trimesh.proximity.closest_point when `rtree` is installed; without
it, a scipy cKDTree on A's vertices proposes candidates and the exact point-to-triangle
distance is taken over the triangles around the nearest vertices (method noted per bone).

  python tools/ct/compare_runs.py RUN_A RUN_B [--bones sacrum hip_left] [--out DIR] [--no-align]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

M_TO_MM = 1000.0
MM_TO_BLENDER = 0.001
COLOR_SCALE = {"blue_mm": 0.0, "white_mm": 1.0, "red_mm": 3.0,
               "meaning": "unaligned distance from each vertex of B to the surface of A"}


def load_mesh_mm(ply):
    mesh = trimesh.load(str(ply), force="mesh", process=False)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * M_TO_MM
    return mesh


def load_info(ply):
    path = ply.with_suffix(".json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


class SurfaceDistance:
    """Unsigned distance from points to a mesh surface, with or without rtree."""

    def __init__(self, mesh, k=4, chunk=20000):
        self.mesh, self.k, self.chunk = mesh, k, chunk
        self.method = None
        self._tree = None

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64)
        if self.method in (None, "trimesh.proximity.closest_point"):
            try:
                _, dist, _ = trimesh.proximity.closest_point(self.mesh, points)
                self.method = "trimesh.proximity.closest_point"
                return np.asarray(dist)
            except Exception as exc:  # rtree missing -> ImportError / wrapped exception
                self.method = f"cKDTree + local point-to-triangle (exact surface query failed: {exc})"
        return self._kdtree_distance(points)

    def _kdtree_distance(self, points):
        if self._tree is None:
            self._tree = cKDTree(self.mesh.vertices)
            vf = np.asarray(self.mesh.vertex_faces)  # (V, maxdeg), padded with -1
            self._vf = vf
        tris = self.mesh.triangles
        out = np.empty(len(points))
        for start in range(0, len(points), self.chunk):
            p = points[start:start + self.chunk]
            _, idx = self._tree.query(p, k=self.k)
            cand = self._vf[idx].reshape(len(p), -1)  # (n, k*maxdeg)
            valid = cand >= 0
            cand = np.where(valid, cand, 0)
            pp = np.repeat(p, cand.shape[1], axis=0)
            cp = trimesh.triangles.closest_point(tris[cand.ravel()], pp)
            d = np.linalg.norm(cp - pp, axis=1).reshape(cand.shape)
            d[~valid] = np.inf
            out[start:start + len(p)] = d.min(axis=1)
        return out


def stats(d):
    return {"mean_mm": float(np.mean(d)), "median_mm": float(np.median(d)),
            "p95_mm": float(np.percentile(d, 95)), "max_mm": float(np.max(d))}


def centroid(mesh):
    if mesh.is_watertight and mesh.volume > 0:
        return np.asarray(mesh.center_mass, dtype=float), "volume"
    return np.asarray(mesh.triangles_center.T @ mesh.area_faces / mesh.area, dtype=float), "surface"


def volume_cm3(mesh, info):
    if info.get("volume_cm3") is not None:
        return float(info["volume_cm3"]), "json"
    if mesh.is_watertight:
        return float(abs(mesh.volume)) / 1000.0, "trimesh"
    return None, "not watertight"


def distance_colors(d):
    """blue (0 mm) -> white (1 mm) -> red (>= 3 mm), RGBA uint8."""
    lo, mid, hi = COLOR_SCALE["blue_mm"], COLOR_SCALE["white_mm"], COLOR_SCALE["red_mm"]
    blue, white, red = np.array([0, 60, 255.0]), np.array([255.0, 255, 255]), np.array([230, 0, 0.0])
    t1 = np.clip((d - lo) / (mid - lo), 0, 1)[:, None]
    t2 = np.clip((d - mid) / (hi - mid), 0, 1)[:, None]
    rgb = np.where(d[:, None] <= mid, blue + (white - blue) * t1, white + (red - white) * t2)
    rgba = np.empty((len(d), 4), dtype=np.uint8)
    rgba[:, :3] = np.round(rgb).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def rotation_angle_deg(matrix):
    c = (np.trace(matrix[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def compare_bone(name, ply_a, ply_b, out_dir, n_samples, align, seed):
    rng_state = np.random.default_rng(seed)
    a, b = load_mesh_mm(ply_a), load_mesh_mm(ply_b)
    info_a, info_b = load_info(ply_a), load_info(ply_b)
    res = {"bone": name, "a": str(ply_a), "b": str(ply_b)}

    for key, mesh, info in (("A", a, info_a), ("B", b, info_b)):
        vol, vsrc = volume_cm3(mesh, info)
        c, csrc = centroid(mesh)
        res[key] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                    "volume_cm3": vol, "volume_source": vsrc,
                    "size_mm": (mesh.bounds[1] - mesh.bounds[0]).tolist(),
                    "centroid_mm": c.tolist(), "centroid_kind": csrc}
    ca, cb = np.array(res["A"]["centroid_mm"]), np.array(res["B"]["centroid_mm"])
    shift = cb - ca
    res["centroid_shift_mm"] = shift.tolist()
    res["centroid_shift_norm_mm"] = float(np.linalg.norm(shift))
    va, vb = res["A"]["volume_cm3"], res["B"]["volume_cm3"]
    res["volume_change_source"] = res["A"]["volume_source"]
    if res["A"]["volume_source"] != res["B"]["volume_source"] and a.is_watertight and b.is_watertight:
        # json and trimesh volumes differ slightly; compare like with like
        va, vb = abs(a.volume) / 1000.0, abs(b.volume) / 1000.0
        res["volume_change_source"] = "trimesh (sources differ)"
    res["volume_change_cm3"] = (vb - va) if va is not None and vb is not None else None
    res["volume_change_pct"] = (100.0 * (vb - va) / va) if va and vb is not None else None

    # trimesh.sample uses the global numpy RNG unless given a seed
    samples, _ = trimesh.sample.sample_surface(b, n_samples, seed=int(rng_state.integers(2**31)))
    dist_a = SurfaceDistance(a)
    res["unaligned"] = stats(dist_a(samples))
    res["distance_method"] = dist_a.method

    if align:
        initial = np.eye(4)
        initial[:3, 3] = ca - cb  # centroid match, B -> A
        target, _ = trimesh.sample.sample_surface(a, max(100000, n_samples * 5),
                                                  seed=int(rng_state.integers(2**31)))
        target = np.vstack([a.vertices, target])
        matrix, moved, _ = trimesh.registration.icp(
            samples, target, initial=initial, threshold=1e-6, max_iterations=80,
            reflection=False, scale=False)
        moved_centroid = trimesh.transform_points(cb[None], matrix)[0]
        res["aligned"] = stats(dist_a(moved))
        res["icp"] = {"rotation_deg": rotation_angle_deg(matrix),
                      "centroid_motion_mm": (moved_centroid - cb).tolist(),
                      "centroid_motion_norm_mm": float(np.linalg.norm(moved_centroid - cb)),
                      "matrix_b_to_a_mm": matrix.tolist()}

    # per-vertex colours on B (unaligned), written back in Blender metres
    vdist = dist_a(b.vertices)
    res["vertex_distance_mm"] = stats(vdist)
    diff = trimesh.Trimesh(vertices=b.vertices * MM_TO_BLENDER, faces=b.faces, process=False)
    diff.visual.vertex_colors = distance_colors(vdist)
    diff_path = out_dir / f"{name}_diff.ply"
    diff.export(str(diff_path))
    res["diff_ply"] = str(diff_path)
    return res


def fmt(v, spec="7.2f"):
    return "    n/a" if v is None else format(v, spec)


def print_table(results, align):
    print()
    print(f"{'bone':<14} {'vol A':>7} {'vol B':>7} {'dVol%':>7}  {'centroid shift A->B (x,y,z) mm':>32} {'|shift|':>7}")
    for r in results:
        s = r["centroid_shift_mm"]
        vec = f"({s[0]:+7.2f},{s[1]:+7.2f},{s[2]:+7.2f})"
        print(f"{r['bone']:<14} {fmt(r['A']['volume_cm3'])} {fmt(r['B']['volume_cm3'])} "
              f"{fmt(r['volume_change_pct'], '+7.2f')}  {vec:>32} {r['centroid_shift_norm_mm']:7.2f}")
    print()
    print(f"{'bone':<14} {'size A x/y/z mm':>24}   {'size B x/y/z mm':>24}")
    for r in results:
        sa = "/".join(f"{v:.1f}" for v in r["A"]["size_mm"])
        sb = "/".join(f"{v:.1f}" for v in r["B"]["size_mm"])
        print(f"{r['bone']:<14} {sa:>24}   {sb:>24}")
    print()
    head = f"{'bone':<14} {'surface B->A mm':<10} {'mean':>7} {'median':>7} {'p95':>7} {'max':>7}"
    if align:
        head += f"   {'ICP rot deg':>11} {'ICP move mm':>11}"
    print(head)
    for r in results:
        rows = [("unaligned", r["unaligned"])] + ([("aligned", r["aligned"])] if align else [])
        for i, (label, st) in enumerate(rows):
            line = (f"{r['bone'] if i == 0 else '':<14} {label:<15} {st['mean_mm']:7.3f} "
                    f"{st['median_mm']:7.3f} {st['p95_mm']:7.3f} {st['max_mm']:7.3f}")
            if align and label == "aligned":
                line += f"   {r['icp']['rotation_deg']:11.3f} {r['icp']['centroid_motion_norm_mm']:11.3f}"
            print(line)
    methods = sorted({r["distance_method"] for r in results})
    print()
    for m in methods:
        print(f"distance method: {m}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="reference run directory (contains meshes/)")
    ap.add_argument("run_b", help="run to compare against A (contains meshes/)")
    ap.add_argument("--bones", nargs="+", default=None, help="bone names (default: all in both runs)")
    ap.add_argument("--out", default=None, help="output dir (default <RUN_B>/compare_vs_<basename RUN_A>)")
    ap.add_argument("--no-align", action="store_true", help="skip the rigid ICP alignment")
    ap.add_argument("--samples", type=int, default=20000, help="surface samples on B")
    ap.add_argument("--seed", type=int, default=0, help="random seed for surface sampling")
    args = ap.parse_args(argv)

    run_a, run_b = Path(args.run_a).expanduser().resolve(), Path(args.run_b).expanduser().resolve()
    mesh_a, mesh_b = run_a / "meshes", run_b / "meshes"
    for d in (mesh_a, mesh_b):
        if not d.is_dir():
            sys.exit(f"no meshes/ folder in {d.parent}")
    common = sorted({p.stem for p in mesh_a.glob("*.ply")} & {p.stem for p in mesh_b.glob("*.ply")})
    common = [n for n in common if not n.endswith("_diff")]
    bones = args.bones or common
    missing = [n for n in bones if n not in common]
    if missing:
        sys.exit(f"not in both runs: {missing}; common bones: {common}")
    if not bones:
        sys.exit("no bone .ply present in both runs")

    out_dir = Path(args.out).expanduser() if args.out else run_b / f"compare_vs_{run_a.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    align = not args.no_align

    results = []
    for name in bones:
        print(f"comparing {name} ...", flush=True)
        results.append(compare_bone(name, mesh_a / f"{name}.ply", mesh_b / f"{name}.ply",
                                    out_dir, args.samples, align, args.seed))

    print_table(results, align)
    report = {"run_a": str(run_a), "run_b": str(run_b), "aligned": align,
              "samples": args.samples, "seed": args.seed, "units": "mm, cm3; LPS scanner frame",
              "color_scale": COLOR_SCALE, "bones": results}
    (out_dir / "compare.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_dir / 'compare.json'} and {len(results)} *_diff.ply")


if __name__ == "__main__":
    main()
