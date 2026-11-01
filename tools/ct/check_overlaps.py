#!/usr/bin/env python
"""Second-pass QA: find where independently extracted neighbouring bones overlap.

extract_bone.py splits each bone from its neighbours in its own run, so two neighbouring
bones (vertebra and rib, sacrum and hip) are split twice, once from each side.  When the two
runs disagree, a region (a rib head, an SI-joint rim) is claimed by both bones, or by neither.
This tool checks every neighbouring pair of a run directory:

  <run>/meshes/<bone>.ply           binary PLY, metres, Blender = scanner LPS mm * 0.001
  <run>/meshes/<bone>.json          volume_cm3, bbox_lps_mm_min/max, basins_file, basin_ids, target_id
  <run>/meshes/<bone>_basins.nrrd   uint8 labels on the bone's cropped grid: target_id = the final
                                    solid bone, other ids = neighbours' basins as seen from this run

Voxel check (needs basin_ids in both JSONs; old runs without them are skipped with a message):
  overlap_vox / mm3   voxels in both final masks (B resampled onto A's grid, nearest neighbour)
  overlap_pct         overlap as % of the smaller bone's final mask
  A_says_B_not_B      voxels A's run put in B's basin that B's run did not keep as final bone
  B_says_A_not_A      the same the other way; together: where the two runs disagree on the split
Mesh check (20000 surface points sampled on the smaller mesh, tested against the other mesh):
  inside / inside_pct sampled points of the smaller mesh lying inside the other mesh
  pen_mm              max penetration: deepest point of either surface inside the other mesh
  gap_mm              min surface-to-surface distance when the meshes do not intersect
                      (joint-space width; from the samples plus the vertices near the other bone)
Flags: OK = overlap < 20 mm3 and pen < 0.5 mm; BAD = overlap > 200 mm3 or pen > 1.5 mm; else CHECK.
Fix for a BAD pair: rerun extract_bone.py for both bones with a smaller --band-mm, or hand-correct.

Outputs (default --out <RUN_DIR>/qa/):
  overlaps.json               every number per pair, plus the methods used
  overlap_<A>__<B>.ply        CHECK/BAD pairs: overlap voxel centres as a point cloud (metres,
                              Blender frame; drop it into the same scene); without voxel data or
                              overlap, the penetrating mesh sample points are written instead

Pairs: by default (A, B) with B in pipeline.neighbours(A) or A in neighbours(B); --all-pairs
takes every pair whose JSON bounding boxes, grown by 2 mm, intersect.

Inside test: trimesh `contains` (ray tests) needs rtree; without it the sign of the nearest
triangle's normal is used (cKDTree on the vertices proposes candidate triangles, the exact
point-to-triangle distance picks the nearest; the method is printed and stored).

  python tools/ct/check_overlaps.py RUN_DIR [--bones sacrum hip_left] [--all-pairs] [--out DIR]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import canonical, neighbours  # noqa: E402

M_TO_MM = 1000.0
MM_TO_BLENDER = 0.001
OK_MM3, BAD_MM3 = 20.0, 200.0
OK_PEN_MM, BAD_PEN_MM = 0.5, 1.5
BBOX_GROW_MM = 2.0
NEAR_MM = 10.0  # vertices this close to the other bone's bbox join the gap search


# ---------------------------------------------------------------- loading

def load_info(meshes, name):
    path = meshes / f"{name}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def load_mesh_mm(ply):
    mesh = trimesh.load(str(ply), force="mesh", process=False)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * M_TO_MM
    return mesh


def bbox_mm(info, mesh_path):
    if "bbox_lps_mm_min" in info and "bbox_lps_mm_max" in info:
        return np.array(info["bbox_lps_mm_min"], float), np.array(info["bbox_lps_mm_max"], float)
    mesh = load_mesh_mm(mesh_path)
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def label_volume(meshes, name, info):
    """(image, array, target_id, {label name: id}) or a string saying why it is unusable."""
    if not info.get("basin_ids") or info.get("target_id") is None:
        return f"{name}: no basin_ids/target_id in {name}.json (old run), voxel check skipped"
    fname = info.get("basins_file") or f"{name}_basins.nrrd"
    path = meshes / fname
    if not path.exists():
        return f"{name}: {fname} missing, voxel check skipped"
    img = sitk.ReadImage(str(path))
    ids = {label: int(i) for i, label in info["basin_ids"].items()}
    return img, sitk.GetArrayFromImage(img), int(info["target_id"]), ids


def resample_onto(img, ref):
    res = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(res)


def voxel_mm3(img):
    return float(np.prod(img.GetSpacing()))


def voxel_centres_mm(img, mask):
    """Physical LPS mm of the True voxels of an array (z, y, x) on img's grid."""
    zyx = np.argwhere(mask).astype(float)
    ijk = zyx[:, ::-1]
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return np.array(img.GetOrigin()) + (ijk * np.array(img.GetSpacing())) @ direction.T


# ---------------------------------------------------------------- mesh queries

class SignedSurface:
    """Distance from points to a mesh surface and whether they lie inside it."""

    exact = False  # ray-based `contains` is exact but minutes per pair on 800k-face meshes

    def __init__(self, mesh, k=4, chunk=20000):
        self.mesh, self.k, self.chunk = mesh, k, chunk
        self.method = None if SignedSurface.exact else "cKDTree + local point-to-triangle, sign of nearest face normal (fast)"
        self._tree = None

    def __call__(self, points):
        """(unsigned distance mm, inside bool) per point."""
        points = np.asarray(points, dtype=np.float64)
        if not SignedSurface.exact:
            # distances from the fast local method; the inside flag from exact ray tests, which are
            # cheap for a few thousand points (the nearest-normal sign test misfires on smooth,
            # interlocking surfaces) but not for the tens of thousands of near vertices
            dist, inside = self._local(points)
            if len(points) > 8000:
                return dist, inside
            try:
                inside = np.asarray(self.mesh.contains(points), bool)
                self.method = "cKDTree + local point-to-triangle distances, ray-test inside (rtree)"
            except Exception as exc:
                self.method = f"cKDTree + local point-to-triangle, sign of nearest face normal (ray test failed: {exc})"
            return dist, inside
        if self.method in (None, "trimesh contains + closest_point (rtree)"):
            try:
                inside = self.mesh.contains(points)
                _, dist, _ = trimesh.proximity.closest_point(self.mesh, points)
                self.method = "trimesh contains + closest_point (rtree)"
                return np.asarray(dist), np.asarray(inside, bool)
            except Exception as exc:  # rtree missing
                self.method = ("cKDTree + local point-to-triangle, sign of nearest face normal "
                               f"(trimesh contains failed: {type(exc).__name__}: {exc})")
        return self._local(points)

    def _local(self, points):
        if self._tree is None:
            self._tree = cKDTree(self.mesh.vertices)
            self._vf = np.asarray(self.mesh.vertex_faces)  # (V, maxdeg), padded with -1
            self._tris = self.mesh.triangles
            self._normals = self.mesh.face_normals
        dist = np.empty(len(points))
        inside = np.zeros(len(points), bool)
        for start in range(0, len(points), self.chunk):
            p = points[start:start + self.chunk]
            _, idx = self._tree.query(p, k=min(self.k, len(self.mesh.vertices)))
            idx = idx.reshape(len(p), -1)
            cand = self._vf[idx].reshape(len(p), -1)
            valid = cand >= 0
            cand = np.where(valid, cand, 0)
            pp = np.repeat(p, cand.shape[1], axis=0)
            cp = trimesh.triangles.closest_point(self._tris[cand.ravel()], pp)
            vec = (pp - cp).reshape(len(p), cand.shape[1], 3)
            d = np.linalg.norm(vec, axis=2)
            d[~valid] = np.inf
            dmin = d.min(axis=1)
            # ties (closest point on a shared edge/vertex): the face whose normal is most
            # aligned with the offset decides the sign
            dot = np.einsum("nkj,nkj->nk", vec, self._normals[cand]) / np.maximum(d, 1e-12)
            tie = d <= dmin[:, None] + 1e-6
            pick = np.argmax(np.where(tie, np.abs(dot), -1.0), axis=1)
            sign = dot[np.arange(len(p)), pick]
            dist[start:start + len(p)] = dmin
            inside[start:start + len(p)] = (sign < 0) & (dmin > 1e-6)
        return dist, inside


def near_vertices(mesh, lo, hi, margin):
    v = mesh.vertices
    keep = np.all((v >= lo - margin) & (v <= hi + margin), axis=1)
    return v[keep]


def mesh_check(mesh_a, mesh_b, small, n_samples, seed):
    """small = 'A' or 'B': which mesh is sampled for the inside count."""
    s_mesh, o_mesh = (mesh_a, mesh_b) if small == "A" else (mesh_b, mesh_a)
    rng = np.random.default_rng(seed)
    samples, _ = trimesh.sample.sample_surface(s_mesh, n_samples, seed=int(rng.integers(2**31)))
    rev, _ = trimesh.sample.sample_surface(o_mesh, n_samples, seed=int(rng.integers(2**31)))
    q_other, q_small = SignedSurface(o_mesh), SignedSurface(s_mesh)

    d_s, in_s = q_other(samples)  # smaller surface vs other mesh
    d_r, in_r = q_small(rev)      # other surface vs smaller mesh (penetration both ways)
    # vertices near the other bone sharpen the gap / depth beyond the random samples
    v_s = near_vertices(s_mesh, *o_mesh.bounds, NEAR_MM)
    v_o = near_vertices(o_mesh, *s_mesh.bounds, NEAR_MM)
    d_vs, in_vs = q_other(v_s) if len(v_s) else (np.empty(0), np.empty(0, bool))
    d_vo, in_vo = q_small(v_o) if len(v_o) else (np.empty(0), np.empty(0, bool))

    pen_parts = [d[i] for d, i in ((d_s, in_s), (d_r, in_r), (d_vs, in_vs), (d_vo, in_vo)) if i.any()]
    penetration = float(max(p.max() for p in pen_parts)) if pen_parts else 0.0
    intersect = bool(pen_parts)
    out_parts = [d[~i] for d, i in ((d_s, in_s), (d_r, in_r), (d_vs, in_vs), (d_vo, in_vo)) if (~i).any()]
    gap = None if intersect or not out_parts else float(min(p.min() for p in out_parts))
    pen_points = np.vstack([samples[in_s], rev[in_r]]) if intersect else np.empty((0, 3))
    return {
        "sampled_mesh": small, "samples": int(n_samples),
        "inside": int(in_s.sum()), "inside_pct": float(100.0 * in_s.mean()),
        "reverse_inside": int(in_r.sum()), "reverse_inside_pct": float(100.0 * in_r.mean()),
        "near_vertices_tested": int(len(v_s) + len(v_o)),
        "intersect": intersect, "penetration_mm": penetration, "min_gap_mm": gap,
        "inside_method": q_other.method,
    }, pen_points


# ---------------------------------------------------------------- voxel check

def voxel_check(name_a, name_b, lv_a, lv_b, target_a, target_b):
    img_a, arr_a, tid_a, ids_a = lv_a
    img_b, arr_b, tid_b, ids_b = lv_b
    final_a, final_b = arr_a == tid_a, arr_b == tid_b
    b_on_a = resample_onto(img_b, img_a)  # B's labels on A's grid (0 outside B's crop)
    a_on_b = resample_onto(img_a, img_b)
    overlap = final_a & (b_on_a == tid_b)
    n_a, n_b = int(final_a.sum()), int(final_b.sum())
    vox_a, vox_b = voxel_mm3(img_a), voxel_mm3(img_b)
    smaller_mm3 = min(n_a * vox_a, n_b * vox_b)
    res = {"A_final_vox": n_a, "B_final_vox": n_b, "voxel_mm3": vox_a,
           "overlap_vox": int(overlap.sum()), "overlap_mm3": float(overlap.sum() * vox_a),
           "overlap_pct_of_smaller": float(100.0 * overlap.sum() * vox_a / smaller_mm3) if smaller_mm3 else None}
    # disagreement: one run gave the voxel to the other bone's basin, the other run did not keep it
    id_b_in_a, id_a_in_b = ids_a.get(target_b), ids_b.get(target_a)
    if id_b_in_a is not None:
        a_says_b = arr_a == id_b_in_a
        dis = a_says_b & (b_on_a != tid_b)
        res.update({"A_says_B_vox": int(a_says_b.sum()), "A_says_B_not_B_vox": int(dis.sum()),
                    "A_says_B_not_B_mm3": float(dis.sum() * vox_a)})
    else:
        res["A_says_B_not_B_vox"] = None
        res["note_A"] = f"{target_b} is not a basin in {name_a}'s run"
    if id_a_in_b is not None:
        b_says_a = arr_b == id_a_in_b
        dis = b_says_a & (a_on_b != tid_a)
        res.update({"B_says_A_vox": int(b_says_a.sum()), "B_says_A_not_A_vox": int(dis.sum()),
                    "B_says_A_not_A_mm3": float(dis.sum() * vox_b)})
    else:
        res["B_says_A_not_A_vox"] = None
        res["note_B"] = f"{target_a} is not a basin in {name_b}'s run"
    return res, voxel_centres_mm(img_a, overlap)


# ---------------------------------------------------------------- pairs / report

def flag(overlap_mm3, pen_mm):
    ov = overlap_mm3 or 0.0
    if ov > BAD_MM3 or pen_mm > BAD_PEN_MM:
        return "BAD"
    if ov < OK_MM3 and pen_mm < OK_PEN_MM:
        return "OK"
    return "CHECK"


def safe_neighbours(bone):
    try:
        return set(neighbours(bone))
    except (ValueError, IndexError):
        return set()


def choose_pairs(bones, meshes, infos, all_pairs):
    pairs = []
    boxes = {b: bbox_mm(infos[b], meshes / f"{b}.ply") for b in bones} if all_pairs else {}
    for a, b in itertools.combinations(bones, 2):
        if all_pairs:
            (lo_a, hi_a), (lo_b, hi_b) = boxes[a], boxes[b]
            if np.all(lo_a - BBOX_GROW_MM <= hi_b + BBOX_GROW_MM) and np.all(lo_b - BBOX_GROW_MM <= hi_a + BBOX_GROW_MM):
                pairs.append((a, b))
        elif b in safe_neighbours(a) or a in safe_neighbours(b):
            pairs.append((a, b))
    return pairs


def fmt(v, spec="8.1f", none="     n/a"):
    return none if v is None else format(v, spec)


def print_table(results):
    print()
    print(f"{'flag':<5} {'A':<16} {'B':<16} {'ovl vox':>8} {'ovl mm3':>8} {'ovl %':>6} "
          f"{'A>B !B':>7} {'B>A !A':>7}   {'inside':>6} {'in %':>5} {'pen mm':>6} {'gap mm':>6}")
    for r in results:
        v, m = r.get("voxel") or {}, r["mesh"]
        print(f"{r['flag']:<5} {r['A']:<16} {r['B']:<16} "
              f"{fmt(v.get('overlap_vox'), '8d')} {fmt(v.get('overlap_mm3'))} "
              f"{fmt(v.get('overlap_pct_of_smaller'), '6.2f', '   n/a')} "
              f"{fmt(v.get('A_says_B_not_B_vox'), '7d', '    n/a')} {fmt(v.get('B_says_A_not_A_vox'), '7d', '    n/a')}   "
              f"{m['inside']:6d} {m['inside_pct']:5.1f} {m['penetration_mm']:6.2f} "
              f"{fmt(m['min_gap_mm'], '6.2f', '     -')}")
    print("\novl = voxels in both final masks; A>B !B = A's run put it in B's basin, B's run did not keep it;")
    print("inside = samples of the smaller mesh inside the other; pen = max penetration; gap = min distance")
    for method in sorted({r["mesh"]["inside_method"] for r in results}):
        print(f"inside test: {method}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run directory (contains meshes/)")
    ap.add_argument("--bones", nargs="+", default=None, help="restrict to these bones (default: all present)")
    ap.add_argument("--all-pairs", action="store_true",
                    help=f"every pair whose bboxes (grown {BBOX_GROW_MM:g} mm) intersect, not only neighbours")
    ap.add_argument("--out", default=None, help="output dir (default <RUN_DIR>/qa)")
    ap.add_argument("--exact", action="store_true",
                    help="use trimesh ray-based inside tests (needs rtree; exact but very slow on large meshes); "
                         "default is the KD-tree + local-triangle method, which matched brute force in tests")
    ap.add_argument("--samples", type=int, default=4000, help="surface samples on the smaller mesh")
    ap.add_argument("--seed", type=int, default=0, help="random seed for surface sampling")
    args = ap.parse_args(argv)
    SignedSurface.exact = bool(args.exact)

    run = Path(args.run_dir).expanduser().resolve()
    meshes = run / "meshes"
    if not meshes.is_dir():
        sys.exit(f"no meshes/ folder in {run}")
    present = sorted(p.stem for p in meshes.glob("*.ply") if not p.stem.endswith("_diff"))
    bones = [canonical(b) for b in args.bones] if args.bones else present
    missing = [b for b in bones if b not in present]
    if missing:
        sys.exit(f"not in {meshes}: {missing}; present: {present}")
    infos = {b: load_info(meshes, b) for b in bones}
    pairs = choose_pairs(bones, meshes, infos, args.all_pairs)
    if not pairs:
        sys.exit(f"no {'touching' if args.all_pairs else 'neighbouring'} pairs among {bones}")

    out_dir = Path(args.out).expanduser() if args.out else run / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    for b in sorted({x for p in pairs for x in p}):
        labels[b] = label_volume(meshes, b, infos[b])
        if isinstance(labels[b], str):
            print(labels[b])
    mesh_cache = {}

    def mesh(name):
        if name not in mesh_cache:
            mesh_cache[name] = load_mesh_mm(meshes / f"{name}.ply")
        return mesh_cache[name]

    results = []
    for a, b in pairs:
        print(f"checking {a} / {b} ...", flush=True)
        ta, tb = infos[a].get("target", a), infos[b].get("target", b)
        res = {"A": a, "B": b}
        points, points_kind = np.empty((0, 3)), None
        if not isinstance(labels[a], str) and not isinstance(labels[b], str):
            res["voxel"], points = voxel_check(a, b, labels[a], labels[b], ta, tb)
            points_kind = "overlap voxel centres"
        else:
            res["voxel"] = None
            res["voxel_skipped"] = "; ".join(x for x in (labels[a], labels[b]) if isinstance(x, str))
        ma, mb = mesh(a), mesh(b)
        vol_a = infos[a].get("volume_cm3") or abs(ma.volume) / 1000.0
        vol_b = infos[b].get("volume_cm3") or abs(mb.volume) / 1000.0
        res["mesh"], pen_points = mesh_check(ma, mb, "A" if vol_a <= vol_b else "B", args.samples, args.seed)
        res["mesh"]["sampled_bone"] = a if vol_a <= vol_b else b
        overlap = res["voxel"]["overlap_mm3"] if res["voxel"] else None
        res["flag"] = flag(overlap, res["mesh"]["penetration_mm"])
        if res["flag"] != "OK":
            if not len(points) and len(pen_points):
                points, points_kind = pen_points, "penetrating mesh sample points"
            if len(points):
                pc = trimesh.PointCloud(points * MM_TO_BLENDER, colors=np.tile([255, 0, 0, 255], (len(points), 1)))
                path = out_dir / f"overlap_{a}__{b}.ply"
                pc.export(str(path))
                res["overlap_ply"], res["overlap_ply_content"] = str(path), points_kind
        results.append(res)

    def sort_key(r):
        return (-(r["voxel"]["overlap_mm3"] if r["voxel"] else -1.0), -r["mesh"]["penetration_mm"])

    results.sort(key=sort_key)
    print_table(results)
    report = {"run": str(run), "pairs_mode": "all-pairs (bbox)" if args.all_pairs else "neighbours",
              "samples": args.samples, "seed": args.seed,
              "units": "mm, mm3; LPS scanner frame (PLY outputs in Blender metres)",
              "thresholds": {"ok_overlap_mm3": OK_MM3, "bad_overlap_mm3": BAD_MM3,
                             "ok_penetration_mm": OK_PEN_MM, "bad_penetration_mm": BAD_PEN_MM},
              "pairs": results}
    (out_dir / "overlaps.json").write_text(json.dumps(report, indent=1))
    n_ply = sum(1 for r in results if r.get("overlap_ply"))
    print(f"\nwrote {out_dir / 'overlaps.json'} and {n_ply} overlap_*.ply")


if __name__ == "__main__":
    main()
