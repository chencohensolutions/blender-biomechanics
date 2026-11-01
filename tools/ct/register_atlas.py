#!/usr/bin/env python
"""Fit the Z-Anatomy skeleton atlas onto the CT bone meshes of a run (global similarity, then
a per-bone similarity ICP), write every fitted atlas bone as a mesh and as a label volume (an
anatomical prior for splitting touching bones, extract_bone.py --atlas-labels), and optionally
import the globally fitted atlas into the running Blender for naming and QA.

  python tools/ct/register_atlas.py RUN_DIR [--meshes-dir DIR] [--atlas-json PATH]
      [--atlas-meshes DIR] [--fbx PATH] [--out DIR] [--voxel-mm 0.9] [--no-refine]
      [--atlas-mapping PATH.json] [--blender] [--collection atlas] [--min-bones 4]
      [--no-cartilage] [--reuse-fit]

Inputs
  --meshes-dir/<ts>.ply     CT bone mesh (extract_bone.py; default RUN_DIR/meshes): metres,
                            x = patient left, y = posterior, z = superior, scanner origin kept
  --meshes-dir/<ts>.json    size_mm, volume_cm3, bbox_lps_mm_min/max
  --atlas-json              atlas_bones.json: [{name, vertices, bbox_min, bbox_max, centroid}]
                            in metres, atlas frame (Blender axes after a default FBX import:
                            +x = patient left, -y = anterior, +z = up).  Both frames use the
                            same axis conventions, so the fitted rotation should be small.
  --atlas-meshes            one PLY per atlas bone + index.json {atlas name: file}, metres, atlas
                            frame (export_atlas_meshes.py, run in headless Blender; default
                            <atlas-json dir>/meshes).  Missing -> the per-bone step is skipped.
  --atlas-mapping           JSON {ts_name: atlas name | [atlas part names]} replacing the built-in
                            Z-Anatomy map (TS_TO_ATLAS / TS_COMPOSITE) for another atlas, e.g.
                            BodyParts3D (index_bp3d.py).  A string (or a one-element list) is a
                            1:1 bone used in the global fit; a longer list is a composite.  TS
                            classes absent from the file have no atlas bone.

Fit
  Every CT mesh whose TotalSegmentator class maps 1:1 to an atlas bone (TS_TO_ATLAS) gives
  one centroid pair: atlas JSON centroid -> CT trimesh centroid (volume centroid for a
  watertight mesh).  Umeyama's least-squares similarity (rotation, uniform scale,
  translation) maps atlas centroids onto CT centroids.  With fewer than --min-bones pairs
  the fit falls back to translation only (mean offset) and warns that rotation and scale
  are unresolved.  A near-collinear set of centroids (e.g. vertebrae only) leaves the roll
  about that line undetermined: a warning is printed.
  Caveat: the atlas centroid is the JSON value (the vertex mean of the atlas mesh), the CT
  centroid is a volume centroid; the two definitions differ by a few mm per bone.

  Composite TS classes (sternum, costal_cartilages, skull) cover several atlas objects
  (TS_COMPOSITE); they are not used in the fit (no comparable centroid), only in the
  size table, with the union of the parts' bounding boxes.

Per-bone refinement (every CT mesh with an atlas bone, composites as the union of their parts)
  The atlas bone, placed by the global transform, is fitted to the CT mesh by a similarity ICP
  in mm (ICP_SAMPLES surface samples on each side, ICP_ITERS iterations, the first
  ICP_RIGID_ITERS without scale).  trimesh.registration.icp is one-directional and with
  scale=True it collapses (atlas -> CT shrinks the atlas onto part of the bone, CT -> atlas
  inflates it), so each iteration pairs atlas -> nearest CT *and* CT -> nearest atlas sample
  and solves one trimesh.registration.procrustes over both, trimmed at ICP_TRIM x median
  distance.  Atlas samples outside the CT bounding box (+ ICP_BOX_MARGIN_MM) take no part in
  atlas -> CT pairs: a CT bone cut by the scan field of view (femur, humerus, C6) is matched
  only where it was scanned.
  Residual = untrimmed symmetric surface distance on dense samples (CT -> fitted atlas, and
  fitted atlas inside the CT box -> CT); mean and 95th percentile in mm.  A fit whose ICP
  scale (relative to the global fit) is outside ICP_SCALE_RANGE or whose mean residual
  exceeds MAX_RESIDUAL_MM is retried as a rigid ICP at the global scale (mode "rigid": a bone
  cut by the field of view seldom determines its scale); if that is rejected too the bone
  falls back to the global transform (mode "global").  Both are flagged.  "partial" flags a
  CT bone covering less than PARTIAL_FRAC of the fitted atlas surface (cut by the field of
  view or under-segmented): its fit rests on the scanned part only.

Outputs (default --out RUN_DIR/atlas/)
  atlas_fit.json   4x4 matrix (row-major, metres, atlas -> CT), scale, rotation angle,
                   matched pairs, residuals (mm), size table, unmatched names on both sides;
                   "bones": per-bone transform (4x4 metres, atlas -> CT), scale (total and
                   ICP), residuals, flag, coverage, label volume and output paths
  meshes/<ts>_atlas.ply   fitted atlas bone, metres, Blender/LPS frame (same as the CT mesh)
  labels/<ts>.nii.gz      binary uint8 label of the fitted atlas bone, axis-aligned --voxel-mm
                   grid in LPS mm (identity direction, origin = first voxel centre): surface
                   voxels of mesh.voxelized(), filled by VoxelGrid.fill(); scipy
                   binary_fill_holes after a one-voxel closing where that fails or leaves a
                   non-watertight atlas mesh hollow
  stdout           RMS / per-bone residuals, scale, rotation angle, and the size table:
                   CT size_mm vs atlas bbox size * scale (atlas axes; meaningful while the
                   rotation is small).  Ratios off by more than SIZE_FLAG flag a bone that
                   may be mislabelled (e.g. a rib assigned to the wrong level); per-bone
                   table (scale, mean / p95 residual, flag) sorted by residual

Costal cartilages (--cartilage, default on; --no-cartilage to skip)
  TotalSegmentator's costal_cartilages class is nearly empty on calcified cartilage (it is
  labelled as rib), so each cartilage instead follows its rib: for every fitted rib_{side}_{n}
  whose atlas has a costal cartilage part for (side, n) (find_costal_cartilages: name
  patterns, works for Z-Anatomy "Costal cartilage of first rib.l" and BodyParts3D "left first
  costal cartilage"; unsided parts are ignored), the rib's final per-bone 4x4 is applied to
  the cartilage mesh -> meshes/costal_cartilage_{side}_{n}_atlas.ply and
  labels/costal_cartilage_{side}_{n}.nii.gz (same voxelization / LPS mm conventions), listed
  in atlas_fit.json "cartilages" with the rib they follow.  Not fitted to the CT: where the
  rib ICP absorbed a calcified cartilage the rib (and so its cartilage) may be stretched.
--reuse-fit keeps the global and per-bone fits stored in <out>/atlas_fit.json (no refit; the
  CT meshes are not read) and only adds the cartilages, updating that JSON in place.

--blender sends a script through blender_bridge.run_code that imports the FBX, keeps only
the real bone meshes (BONE_SUFFIX_RE / numeric-suffix filter), moves them into --collection,
applies obj.matrix_world = M @ obj.matrix_world, tags each with "atlas_source", hides the
collection in the viewport, and saves the .blend if it has a path.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from trimesh.registration import procrustes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DRIVE = Path("/Users/hencohen/Google Drive/My Drive/spine")
DEFAULT_ATLAS_JSON = DRIVE / "segmentation/atlas/atlas_bones.json"
DEFAULT_FBX = DRIVE / "zanatomy/SkeletalSystem100.fbx"
ATLAS_SOURCE = "Z-Anatomy SkeletalSystem100.fbx (CC BY-SA 4.0)"
BONE_SUFFIX_RE = r"\.(i|j|el|er|ol|or|t|[eo]\d*[lr])$"   # Z-Anatomy label/helper objects
NUMERIC_SUFFIX_RE = r"\.\d{3,}$"                           # Blender duplicate names (.001)
SIZE_FLAG = 0.25        # flag a bone when any CT/atlas size ratio is off by more than 25 %
COLLINEAR_FLAG = 0.05   # 2nd/1st singular value of the centred atlas centroids
M_TO_MM = 1000.0
ICP_SAMPLES = 5000            # surface samples per side for the ICP
ICP_ITERS = 50
ICP_RIGID_ITERS = 10          # first iterations without scale (scale-ICP collapses from a poor start)
ICP_TRIM = 3.0                # drop pairs farther than ICP_TRIM x median (min ICP_TRIM_MIN_MM)
ICP_TRIM_MIN_MM = 2.0
ICP_BOX_MARGIN_MM = 5.0       # atlas samples outside the CT bbox + margin: no atlas -> CT pair
ICP_SCALE_RANGE = (0.7, 1.4)  # ICP scale (relative to the global fit) outside -> fall back
MAX_RESIDUAL_MM = 6.0         # mean symmetric residual above -> fall back to the global fit
RES_SAMPLES = 50000           # dense surface samples for the residual
PARTIAL_FRAC = 0.5            # fitted atlas surface within 3 mm of the CT below -> "partial"
VOXEL_MM = 0.9

# ---------------------------------------------------------------- TS class -> atlas name
SIDES = {"left": "l", "right": "r"}   # TS "left" = patient left = atlas ".l" (+x)
ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth",
            "Ninth", "Tenth", "Eleventh", "Twelfth"]


def _build_mapping():
    m = {"sacrum": "Sacrum", "vertebrae_C1": "Atlas (C1)", "vertebrae_C2": "Axis (C2)"}
    m.update({f"vertebrae_C{i}": f"Vertebra C{i}" for i in range(3, 8)})
    m.update({f"vertebrae_T{i}": f"Vertebra T{i}" for i in range(1, 13)})
    m.update({f"vertebrae_L{i}": f"Vertebra L{i}" for i in range(1, 6)})
    for side, s in SIDES.items():
        m[f"hip_{side}"] = f"Hip bone.{s}"
        m[f"femur_{side}"] = f"Femur.{s}"
        m[f"clavicula_{side}"] = f"Clavicle.{s}"
        m[f"scapula_{side}"] = f"Scapula.{s}"
        m[f"humerus_{side}"] = f"Humerus.{s}"
        for i, word in enumerate(ORDINALS, 1):
            m[f"rib_{side}_{i}"] = f"{word} rib.{s}"
    return m


TS_TO_ATLAS = _build_mapping()

_PAIRED_SKULL = ["Temporal bone", "Parietal bone", "Maxilla", "Zygomatic bone", "Nasal bone",
                 "Lacrimal bone", "Palatine bone", "Inferior nasal concha bone"]
TS_COMPOSITE = {
    "sternum": ["Manubrium of sternum", "Body of sternum", "Xiphoid process"],
    "costal_cartilages": [f"Costal cartilage of {w.lower()} rib.{s}"
                          for w in ORDINALS[:10] for s in ("l", "r")],
    "skull": (["Frontal bone", "Occipital bone", "Sphenoid bone", "Ethmoid bone", "Vomer",
               "Mandible"] + [f"{b}.{s}" for b in _PAIRED_SKULL for s in ("l", "r")]),
}

TS_CLASSES = (["sacrum"] + [f"vertebrae_C{i}" for i in range(1, 8)]
              + [f"vertebrae_T{i}" for i in range(1, 13)] + [f"vertebrae_L{i}" for i in range(1, 6)]
              + [f"{b}_{s}" for b in ("hip", "femur") for s in SIDES]
              + [f"rib_{s}_{i}" for s in SIDES for i in range(1, 13)]
              + ["sternum"] + [f"{b}_{s}" for b in ("clavicula", "scapula", "humerus") for s in SIDES]
              + ["skull", "costal_cartilages"])


def apply_mapping(path):
    """Replace TS_TO_ATLAS / TS_COMPOSITE (in place) by the map in a JSON file
    {ts_name: atlas name or [atlas part names]}.  Returns (n 1:1, n composite)."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"--atlas-mapping {path}: expected a JSON object {{ts_name: atlas name or list}}")
    one, comp = {}, {}
    for ts, target in raw.items():
        if isinstance(target, str):
            one[ts] = target
        elif isinstance(target, list) and target and all(isinstance(t, str) for t in target):
            if len(target) == 1:
                one[ts] = target[0]
            else:
                comp[ts] = list(target)
        else:
            raise SystemExit(f"--atlas-mapping {path}: {ts!r} -> {target!r} is not a name or a list of names")
    TS_TO_ATLAS.clear()
    TS_TO_ATLAS.update(one)
    TS_COMPOSITE.clear()
    TS_COMPOSITE.update(comp)
    return len(one), len(comp)


def check_mapping(atlas_names):
    """Return (missing_ts, bad_targets): TS classes with no 1:1 atlas name, and mapped
    atlas names (incl. composite parts) that do not exist in the atlas JSON."""
    names = set(atlas_names)
    missing = [ts for ts in TS_CLASSES if ts not in TS_TO_ATLAS]
    bad = sorted({a for a in TS_TO_ATLAS.values() if a not in names}
                 | {a for parts in TS_COMPOSITE.values() for a in parts if a not in names})
    return missing, bad


# ---------------------------------------------------------------- fitting
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity dst ~ s * R @ src + t (Umeyama 1991).  Returns (s, R, t)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def to_matrix(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def rotation_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def rotation_axis(R):
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 1e-12 else [0.0, 0.0, 0.0]


def fit(atlas_pts, ct_pts, min_bones=4):
    """Similarity fit atlas -> CT; translation-only below min_bones.  Returns a dict."""
    atlas_pts, ct_pts = np.asarray(atlas_pts, float), np.asarray(ct_pts, float)
    warnings = []
    n = len(atlas_pts)
    if n == 0:
        raise ValueError("no matched bones")
    if n >= max(min_bones, 3):
        mode = "similarity"
        s, R, t = umeyama(atlas_pts, ct_pts)
        sv = np.linalg.svd(atlas_pts - atlas_pts.mean(0), compute_uv=False)
        if sv[1] < COLLINEAR_FLAG * sv[0]:
            warnings.append(f"matched atlas centroids are nearly collinear (sv ratio {sv[1] / sv[0]:.3f}): "
                            "rotation about their common line is poorly determined")
    else:
        mode = "translation"
        s, R = 1.0, np.eye(3)
        t = ct_pts.mean(0) - atlas_pts.mean(0)
        warnings.append(f"only {n} matched bone(s) (< --min-bones {min_bones}): translation-only fit, "
                        "rotation and scale are UNRESOLVED (scale=1, rotation=identity)")
    pred = (s * (R @ atlas_pts.T)).T + t
    res = np.linalg.norm(pred - ct_pts, axis=1) * M_TO_MM
    return {"mode": mode, "scale": s, "R": R, "t": t, "matrix": to_matrix(s, R, t),
            "rotation_deg": rotation_angle_deg(R), "rotation_axis": rotation_axis(R),
            "residuals_mm": res, "rms_mm": float(np.sqrt((res ** 2).mean())), "warnings": warnings}


# ---------------------------------------------------------------- I/O
def load_atlas(path):
    return {e["name"]: e for e in json.loads(Path(path).read_text())}


def ct_bone_info(mesh_dir, ts):
    ply = mesh_dir / f"{ts}.ply"
    mesh = trimesh.load(ply, force="mesh", process=False)
    meta_path = mesh_dir / f"{ts}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    size = meta.get("size_mm") or (mesh.extents * M_TO_MM).tolist()
    return {"centroid_m": np.asarray(mesh.centroid, float), "size_mm": [float(v) for v in size],
            "volume_cm3": meta.get("volume_cm3"), "watertight": bool(mesh.is_watertight)}


def atlas_bbox_size_m(atlas, names):
    lo = np.min([atlas[n]["bbox_min"] for n in names], axis=0)
    hi = np.max([atlas[n]["bbox_max"] for n in names], axis=0)
    return hi - lo


def size_table(atlas, ct, scale):
    rows = []
    for ts, info in ct.items():
        parts = [TS_TO_ATLAS[ts]] if ts in TS_TO_ATLAS else TS_COMPOSITE.get(ts)
        if not parts or any(p not in atlas for p in parts):
            continue
        a_mm = atlas_bbox_size_m(atlas, parts) * scale * M_TO_MM
        c_mm = np.asarray(info["size_mm"])
        ratio = c_mm / np.maximum(a_mm, 1e-9)
        rows.append({"ts": ts, "atlas": parts[0] if len(parts) == 1 else f"{len(parts)} parts",
                     "ct_size_mm": c_mm.round(2).tolist(), "atlas_size_mm": a_mm.round(2).tolist(),
                     "ratio": ratio.round(3).tolist(),
                     "flag": bool(np.any(np.abs(ratio - 1.0) > SIZE_FLAG))})
    return rows


# ---------------------------------------------------------------- per-bone refinement
def load_atlas_mesh_index(mesh_dir):
    """{atlas name: PLY path} from export_atlas_meshes.py's index.json, or None."""
    index = Path(mesh_dir) / "index.json"
    if not index.exists():
        return None
    return {name: Path(mesh_dir) / f for name, f in json.loads(index.read_text()).items()}


def atlas_parts_mesh(index, parts):
    """Union (concatenation) of the atlas parts' meshes, metres, atlas frame."""
    meshes = [trimesh.load(index[n], force="mesh", process=False) for n in parts]
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def m_to_mm_matrix(M):
    out = np.array(M, float)
    out[:3, 3] *= M_TO_MM
    return out


def mm_to_m_matrix(M):
    out = np.array(M, float)
    out[:3, 3] /= M_TO_MM
    return out


def matrix_scale(M):
    return float(np.cbrt(abs(np.linalg.det(np.asarray(M)[:3, :3]))))


def in_box(pts, lo, hi):
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def similarity_icp(a, c, initial, iters=ICP_ITERS, rigid_iters=ICP_RIGID_ITERS):
    """Symmetric trimmed similarity ICP (mm): returns M with c ~ M @ a.  Each iteration pairs
    a -> nearest c (only a inside the c bbox + margin) and c -> nearest a, drops pairs beyond
    ICP_TRIM x median, and solves one procrustes (no reflection; scale after rigid_iters)."""
    lo, hi = c.min(0) - ICP_BOX_MARGIN_MM, c.max(0) + ICP_BOX_MARGIN_MM
    ctree = cKDTree(c)
    M = np.array(initial, float)
    for it in range(iters):
        ap = trimesh.transform_points(a, M)
        ins = in_box(ap, lo, hi)
        d1, i1 = ctree.query(ap[ins], workers=-1)
        d2, i2 = cKDTree(ap).query(c, workers=-1)
        src = np.vstack([ap[ins], ap[i2]])
        dst = np.vstack([c[i1], c])
        d = np.concatenate([d1, d2])
        keep = d <= max(ICP_TRIM * np.median(d), ICP_TRIM_MIN_MM)
        step = procrustes(src[keep], dst[keep], reflection=False, scale=it >= rigid_iters,
                          return_cost=False)
        M = step @ M
        if it >= rigid_iters and np.abs(step - np.eye(4)).max() < 1e-7 and np.abs(step[:3, 3]).max() < 1e-4:
            break
    return M


def surface_residuals(a_dense, c_dense, c_tree, lo, hi):
    """Untrimmed symmetric distances (mm): CT -> atlas, and atlas inside the CT box -> CT.
    Returns (all distances, fraction of atlas samples within 3 mm of the CT)."""
    d_ca = cKDTree(a_dense, balanced_tree=False).query(c_dense, workers=-1)[0]
    d_ac_all = c_tree.query(a_dense, workers=-1)[0]
    d_ac = d_ac_all[in_box(a_dense, lo, hi)]
    return np.concatenate([d_ca, d_ac]), float((d_ac_all <= 3.0).mean())


def refine_bone(atlas_m, ct_m, G_m, seed=0):
    """Per-bone similarity ICP of an atlas mesh (metres, atlas frame) onto a CT mesh (metres),
    starting from the global transform G_m.  Returns a dict (transform in metres)."""
    A = atlas_m.copy()
    A.apply_scale(M_TO_MM)
    C = ct_m.copy()
    C.apply_scale(M_TO_MM)
    G = m_to_mm_matrix(G_m)
    a = trimesh.sample.sample_surface(A, ICP_SAMPLES, seed=seed)[0]
    c = trimesh.sample.sample_surface(C, ICP_SAMPLES, seed=seed + 1)[0]
    a_dense = trimesh.sample.sample_surface(A, RES_SAMPLES, seed=seed + 2)[0]
    c_dense = trimesh.sample.sample_surface(C, RES_SAMPLES, seed=seed + 3)[0]
    c_tree = cKDTree(c_dense, balanced_tree=False)
    lo, hi = C.bounds[0] - ICP_BOX_MARGIN_MM, C.bounds[1] + ICP_BOX_MARGIN_MM

    def stats(M):
        d, cover = surface_residuals(trimesh.transform_points(a_dense, M), c_dense, c_tree, lo, hi)
        return float(d.mean()), float(np.percentile(d, 95)), cover

    def reject(M, mean):
        s_icp = matrix_scale(M @ np.linalg.inv(G))
        if not ICP_SCALE_RANGE[0] <= s_icp <= ICP_SCALE_RANGE[1]:
            return f"ICP scale {s_icp:.3f}"
        if mean > MAX_RESIDUAL_MM:
            return f"residual {mean:.1f} mm"
        return None

    g_mean, g_p95, g_cover = stats(G)
    M = similarity_icp(a, c, G)
    icp_scale = matrix_scale(M @ np.linalg.inv(G))
    mean, p95, cover = stats(M)
    icp = {"icp_scale": icp_scale, "icp_residual_mean_mm": mean, "icp_residual_p95_mm": p95}
    mode, flag = "similarity", None
    why = reject(M, mean)
    if why:
        # a bone cut by the field of view rarely fixes the scale: retry rigid at the global scale
        Mr = similarity_icp(a, c, G, rigid_iters=ICP_ITERS)
        r_mean, r_p95, r_cover = stats(Mr)
        if reject(Mr, r_mean) is None:
            M, mean, p95, cover, mode = Mr, r_mean, r_p95, r_cover, "rigid"
            flag = f"rigid (similarity rejected: {why})"
        else:
            M, mean, p95, cover, mode = G, g_mean, g_p95, g_cover, "global"
            flag = f"fallback: {why}; rigid residual {r_mean:.1f} mm"
    partial = cover < PARTIAL_FRAC
    if partial:
        flag = f"{flag}; partial" if flag else "partial"
    R = M[:3, :3] / matrix_scale(M)
    Rg = G[:3, :3] / matrix_scale(G)
    return {"matrix_mm": M, "matrix_atlas_to_ct": mm_to_m_matrix(M), "scale": matrix_scale(M),
            **icp, "rotation_vs_global_deg": rotation_angle_deg(R @ Rg.T),
            "residual_mean_mm": mean, "residual_p95_mm": p95,
            "global_residual_mean_mm": g_mean, "global_residual_p95_mm": g_p95,
            "atlas_coverage": cover, "mode": mode, "fallback": mode == "global", "partial": partial,
            "flag": flag}


def voxelize_label(mesh_mm, pitch):
    """Binary label of a mesh in mm on an axis-aligned grid.  Returns (sitk image, method)."""
    vg = mesh_mm.voxelized(pitch)
    T = np.asarray(vg.transform)
    if np.abs(T[:3, :3] - np.diag(np.diag(T[:3, :3]))).max() > 1e-9:
        raise RuntimeError("voxel grid is not axis-aligned")
    surface = np.pad(np.asarray(vg.matrix, bool), 1)
    method = "trimesh fill"
    try:
        filled = np.pad(np.asarray(vg.copy().fill().matrix, bool), 1)
    except Exception as exc:  # noqa: BLE001 - any failure of trimesh's filler -> scipy below
        filled, method = None, f"scipy fill_holes (trimesh fill failed: {type(exc).__name__})"
    if filled is None or not mesh_mm.is_watertight:
        # a gap in a non-watertight surface lets binary_fill_holes leak: seal one voxel first
        sealed = ndi.binary_fill_holes(ndi.binary_closing(surface, structure=ndi.generate_binary_structure(3, 1)))
        if filled is None:
            filled = sealed | surface
        elif sealed.sum() > 1.05 * filled.sum():
            filled, method = sealed | surface, "scipy closing + fill_holes (trimesh fill left it hollow)"
    # trimesh's subdivide_to_size can emit a stray unreferenced vertex -> an isolated voxel far
    # from the bone: keep only components that hold a voxel of a real mesh vertex
    comp, n = ndi.label(filled, structure=np.ones((3, 3, 3)))
    if n > 1:
        used = mesh_mm.vertices[np.unique(mesh_mm.faces)]
        ijk = np.clip(vg.points_to_indices(used) + 1, 0, np.array(filled.shape) - 1)
        keep = np.unique(comp[tuple(ijk.T)])
        filled = np.isin(comp, keep[keep > 0])
    arr = filled.astype(np.uint8).transpose(2, 1, 0)       # [x, y, z] -> sitk [z, y, x]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(pitch)] * 3)
    img.SetOrigin((T[:3, 3] - pitch).tolist())               # one voxel of padding
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img, method


def label_stats(img):
    arr = sitk.GetArrayFromImage(img)
    zyx = np.argwhere(arr > 0)
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    lo = org + zyx.min(0)[::-1] * sp - sp / 2
    hi = org + zyx.max(0)[::-1] * sp + sp / 2
    return {"voxels": int(len(zyx)), "volume_cm3": float(len(zyx) * np.prod(sp) / 1000.0),
            "bbox_lps_mm_min": lo.tolist(), "bbox_lps_mm_max": hi.tolist(), "size": list(img.GetSize())}


def write_fitted(atlas_m, M_mm, mesh_path, label_path, pitch):
    """Place an atlas mesh (metres, atlas frame) by M_mm (mm, atlas*1000 -> CT mm), write it as
    a PLY in metres and as a binary label.  Returns (fitted mesh in mm, label stats, fill method,
    label bbox vs mesh bbox in mm)."""
    fitted_mm = atlas_m.copy()
    fitted_mm.apply_scale(M_TO_MM)
    fitted_mm.apply_transform(M_mm)
    fitted_m = fitted_mm.copy()
    fitted_m.apply_scale(1.0 / M_TO_MM)
    fitted_m.export(mesh_path)
    img, method = voxelize_label(fitted_mm, pitch)
    sitk.WriteImage(img, str(label_path), useCompression=True)
    lab = label_stats(img)
    used = fitted_mm.vertices[np.unique(fitted_mm.faces)]   # ignore loose atlas vertices
    bbox_err = max(np.abs(np.array(lab["bbox_lps_mm_min"]) - used.min(0)).max(),
                   np.abs(np.array(lab["bbox_lps_mm_max"]) - used.max(0)).max())
    return fitted_mm, lab, method, float(bbox_err)


def refine_all(ct_mesh_dir, ts_names, atlas, index, G_m, out_dir, pitch):
    """Fit, write mesh + label for every CT bone with atlas parts.  Returns {ts: result}."""
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    results = {}
    for ts in ts_names:
        parts = [TS_TO_ATLAS[ts]] if ts in TS_TO_ATLAS else TS_COMPOSITE.get(ts)
        if not parts:
            continue
        missing = [n for n in parts if n not in index]
        if missing:
            print(f"  {ts}: atlas mesh missing for {missing}, skipped")
            continue
        t0 = time.time()
        atlas_m = atlas_parts_mesh(index, parts)
        ct_m = trimesh.load(ct_mesh_dir / f"{ts}.ply", force="mesh", process=False)
        r = refine_bone(atlas_m, ct_m, G_m)
        mesh_path = out_dir / "meshes" / f"{ts}_atlas.ply"
        label_path = out_dir / "labels" / f"{ts}.nii.gz"
        _, lab, method, bbox_err = write_fitted(atlas_m, r["matrix_mm"], mesh_path, label_path, pitch)
        results[ts] = {
            "atlas": parts, "matrix_atlas_to_ct": r["matrix_atlas_to_ct"].tolist(),
            "scale": r["scale"], "icp_scale": r["icp_scale"],
            "rotation_vs_global_deg": r["rotation_vs_global_deg"],
            "residual_mean_mm": r["residual_mean_mm"], "residual_p95_mm": r["residual_p95_mm"],
            "global_residual_mean_mm": r["global_residual_mean_mm"],
            "global_residual_p95_mm": r["global_residual_p95_mm"],
            "icp_residual_mean_mm": r["icp_residual_mean_mm"], "atlas_coverage": r["atlas_coverage"],
            "mode": r["mode"], "fallback": r["fallback"], "partial": r["partial"], "flag": r["flag"],
            "atlas_watertight": bool(atlas_m.is_watertight), "mesh": str(mesh_path),
            "label": str(label_path), "label_fill": method, "label_voxel_mm": pitch,
            "label_volume_cm3": lab["volume_cm3"], "label_size": lab["size"],
            "label_bbox_lps_mm_min": lab["bbox_lps_mm_min"], "label_bbox_lps_mm_max": lab["bbox_lps_mm_max"],
            "label_bbox_vs_mesh_mm": float(bbox_err), "seconds": round(time.time() - t0, 2),
        }
        print(f"  {ts:<18} scale {r['scale']:.3f} (icp {r['icp_scale']:.3f})  residual "
              f"{r['residual_mean_mm']:.2f}/{r['residual_p95_mm']:.2f} mm (global {r['global_residual_mean_mm']:.1f})  "
              f"label {lab['volume_cm3']:.1f} cm3  {r['flag'] or ''}")
    return results


def print_bone_table(bones):
    print(f"\n{'ts bone':<18} {'scale':>6} {'icp s':>6} {'mean mm':>8} {'p95 mm':>7} {'global':>7} "
          f"{'cover':>6} {'label cm3':>9}  flag")
    for ts, b in sorted(bones.items(), key=lambda kv: kv[1]["residual_mean_mm"]):
        print(f"{ts:<18} {b['scale']:>6.3f} {b['icp_scale']:>6.3f} {b['residual_mean_mm']:>8.2f} "
              f"{b['residual_p95_mm']:>7.2f} {b['global_residual_mean_mm']:>7.1f} {b['atlas_coverage']:>6.2f} "
              f"{b['label_volume_cm3']:>9.1f}  {b['flag'] or ''}")
    n_fb = sum(b["fallback"] for b in bones.values())
    print(f"{len(bones)} bones: {n_fb} fell back to the global transform, "
          f"{sum(b['mode'] == 'rigid' for b in bones.values())} rigid retry, "
          f"{sum(b['partial'] for b in bones.values())} partial")


# ---------------------------------------------------------------- costal cartilages
_ORDINAL_NUM = {w.lower(): i for i, w in enumerate(ORDINALS, 1)}
_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINAL_NUM) + r")\b")
_DIGIT_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\b")
_SIDE_SUFFIX_RE = re.compile(r"\.([lr])$")
_SIDE_WORD_RE = re.compile(r"^(left|right)\b")


def find_costal_cartilages(names):
    """{(side, n): atlas name} for the sided costal cartilage parts among atlas names; side is
    "left"/"right" (patient side, as in the TS rib names), n the rib number.  A name matches when
    it contains "costal cartilage", exactly one rib number (ordinal word first..twelfth or a
    number, e.g. "3rd") and a side (".l"/".r" suffix or a "left"/"right" prefix).  Unsided or
    ambiguous names (BodyParts3D "first costal cartilage", "costal cartilage") are ignored; a
    second name for the same (side, n) is reported and the first (sorted) kept."""
    found, dupes = {}, []
    for name in sorted(names):
        low = name.lower().strip()
        if "costal cartilage" not in low:
            continue
        m = _SIDE_SUFFIX_RE.search(low)
        if m:
            side = "left" if m.group(1) == "l" else "right"
            low = low[:m.start()]
        else:
            m = _SIDE_WORD_RE.search(low)
            if not m:
                continue
            side = m.group(1)
        nums = {_ORDINAL_NUM[w] for w in _ORDINAL_RE.findall(low)} | {int(d) for d in _DIGIT_RE.findall(low)}
        if len(nums) != 1:
            continue
        n = nums.pop()
        if not 1 <= n <= 12:
            continue
        if (side, n) in found:
            dupes.append(name)
            continue
        found[(side, n)] = name
    if dupes:
        print("WARNING: extra costal cartilage names ignored (same side and rib):", dupes)
    return found


def fit_cartilages(bones, atlas_names, index, out_dir, pitch):
    """Place every atlas costal cartilage with its rib's final per-bone transform (bones[rib]
    ["matrix_atlas_to_ct"]) and write mesh + label.  Returns ({name: result}, found list)."""
    found = find_costal_cartilages(atlas_names)
    for side in SIDES:
        ns = sorted(n for s, n in found if s == side)
        print(f"  atlas costal cartilages, {side}: {ns or 'none'}")
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    results = {}
    for (side, n), part in sorted(found.items()):
        rib, name = f"rib_{side}_{n}", f"costal_cartilage_{side}_{n}"
        b = bones.get(rib)
        if b is None:
            print(f"  {name}: {rib} has no per-bone fit, skipped")
            continue
        if index is None or part not in index:
            print(f"  {name}: atlas mesh for {part!r} missing, skipped")
            continue
        t0 = time.time()
        atlas_m = trimesh.load(index[part], force="mesh", process=False)
        M_m = np.asarray(b["matrix_atlas_to_ct"], float)
        mesh_path = out_dir / "meshes" / f"{name}_atlas.ply"
        label_path = out_dir / "labels" / f"{name}.nii.gz"
        fitted_mm, lab, method, bbox_err = write_fitted(atlas_m, m_to_mm_matrix(M_m), mesh_path, label_path, pitch)
        results[name] = {
            "atlas": [part], "follows": rib, "matrix_atlas_to_ct": M_m.tolist(), "scale": matrix_scale(M_m),
            "rib_mode": b.get("mode"), "rib_flag": b.get("flag"),
            "rib_residual_mean_mm": b.get("residual_mean_mm"),
            "atlas_watertight": bool(atlas_m.is_watertight), "mesh": str(mesh_path),
            "label": str(label_path), "label_fill": method, "label_voxel_mm": pitch,
            "label_volume_cm3": lab["volume_cm3"], "label_size": lab["size"],
            "label_bbox_lps_mm_min": lab["bbox_lps_mm_min"], "label_bbox_lps_mm_max": lab["bbox_lps_mm_max"],
            "label_bbox_vs_mesh_mm": bbox_err, "centroid_lps_mm": fitted_mm.vertices.mean(0).tolist(),
            "seconds": round(time.time() - t0, 2),
        }
        print(f"  {name:<24} <- {rib:<13} {part!r:<36} label {lab['volume_cm3']:.1f} cm3"
              f"{'  (rib ' + b['flag'] + ')' if b.get('flag') else ''}")
    return results, [{"side": s, "rib": n, "atlas": p} for (s, n), p in sorted(found.items())]


# ---------------------------------------------------------------- Blender
BLENDER_IMPORT = r'''
import bpy, re
from mathutils import Matrix
fbx, col_name, rows, source = %(fbx)r, %(collection)r, %(matrix)r, %(source)r
M = Matrix(rows)
label_re, dup_re = re.compile(%(label_re)r), re.compile(%(dup_re)r)
scene = bpy.context.scene
col = bpy.data.collections.get(col_name)
if col is None:
    col = bpy.data.collections.new(col_name)
if col.name not in [c.name for c in scene.collection.children]:
    scene.collection.children.link(col)
# replace an earlier atlas import so names do not collide (.001)
stale = [o for o in col.objects if "atlas_source" in o]
for o in stale:
    bpy.data.objects.remove(o, do_unlink=True)
before = set(bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=fbx)
new = [o for o in bpy.data.objects if o not in before]
keep = [o for o in new if o.type == "MESH" and not label_re.search(o.name) and not dup_re.search(o.name)]
keep_set = set(keep)
clashes = [o.name for o in new if o.type == "MESH" and dup_re.search(o.name)
           and dup_re.sub("", o.name) in {b.name for b in before}]
# keep world placement when parents (empties, armatures) are deleted
for o in keep:
    if o.parent is not None:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
for o in new:
    if o not in keep_set:
        bpy.data.objects.remove(o, do_unlink=True)
for o in keep:
    for c in list(o.users_collection):
        c.objects.unlink(o)
    col.objects.link(o)
    o.matrix_world = M @ o.matrix_world
    o["atlas_source"] = source
for block in list(bpy.data.meshes):
    if block.users == 0:
        bpy.data.meshes.remove(block)
col.hide_viewport = True
if bpy.data.filepath:
    bpy.ops.wm.save_mainfile()
print("imported atlas bones:", len(keep), "of", len(new), "objects; replaced", len(stale),
      "; collection", col.name, "(hidden)")
if clashes:
    print("WARNING: dropped", len(clashes), "bones renamed .NNN because the scene already had that name:",
          clashes[:10])
'''


def blender_script(matrix, fbx, collection):
    return BLENDER_IMPORT % {"fbx": str(fbx), "collection": collection,
                             "matrix": [[float(v) for v in row] for row in np.asarray(matrix)],
                             "source": ATLAS_SOURCE, "label_re": BONE_SUFFIX_RE, "dup_re": NUMERIC_SUFFIX_RE}


# ---------------------------------------------------------------- main
CARTILAGES_LAYOUT = ("costal cartilage placed by the per-bone transform of the rib it follows (not fitted "
                     "to the CT); matrix_atlas_to_ct = that rib's 4x4, metres, p_ct = M @ p_atlas")


def run_reuse(run_dir, atlas_json, out, atlas_meshes, voxel_mm, cartilage):
    """--reuse-fit: keep the stored fits in <out>/atlas_fit.json, add the cartilages only."""
    t_start = time.time()
    out_dir = Path(out) if out else Path(run_dir) / "atlas"
    fit_path = out_dir / "atlas_fit.json"
    if not fit_path.exists():
        raise SystemExit(f"--reuse-fit: no {fit_path}")
    result = json.loads(fit_path.read_text())
    bones = result.get("bones")
    if not bones:
        raise SystemExit(f"--reuse-fit: {fit_path} has no per-bone fits ('bones')")
    if Path(result.get("atlas_json", atlas_json)) != Path(atlas_json):
        print(f"WARNING: --atlas-json {atlas_json} differs from the stored fit's {result['atlas_json']}")
    atlas_meshes = Path(atlas_meshes) if atlas_meshes else Path(result.get("atlas_meshes") or Path(atlas_json).parent / "meshes")
    print(f"reusing {len(bones)} per-bone fits from {fit_path}")
    if not cartilage:
        print("--no-cartilage: nothing to do")
        return result
    index = load_atlas_mesh_index(atlas_meshes)
    if index is None:
        raise SystemExit(f"no {atlas_meshes / 'index.json'}: cannot place the cartilages")
    print(f"\ncostal cartilages from {atlas_json}")
    carts, found = fit_cartilages(bones, list(load_atlas(atlas_json)), index, out_dir, voxel_mm)
    result.update({"cartilages_layout": CARTILAGES_LAYOUT, "cartilage_parts_found": found, "cartilages": carts})
    fit_path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {len(carts)} cartilage meshes/labels; updated {fit_path}  ({time.time() - t_start:.1f} s)")
    return result


def run(run_dir, atlas_json=DEFAULT_ATLAS_JSON, out=None, min_bones=4, meshes_dir=None,
        atlas_meshes=None, refine=True, voxel_mm=VOXEL_MM, atlas_mapping=None, cartilage=True,
        reuse_fit=False):
    if reuse_fit:
        return run_reuse(run_dir, atlas_json, out, atlas_meshes, voxel_mm, cartilage)
    t_start = time.time()
    run_dir = Path(run_dir)
    if atlas_mapping:
        n1, nc = apply_mapping(atlas_mapping)
        print(f"atlas mapping {atlas_mapping}: {n1} 1:1 bones, {nc} composites (built-in map replaced)")
    mesh_dir = Path(meshes_dir) if meshes_dir else run_dir / "meshes"
    atlas = load_atlas(atlas_json)
    missing, bad = check_mapping(atlas)
    if missing:
        print("TS classes without a 1:1 atlas bone (not used in the fit):", ", ".join(missing))
    if bad:
        print("WARNING: mapped atlas names missing from the atlas JSON:", ", ".join(bad))

    present = sorted(p.stem for p in mesh_dir.glob("*.ply"))
    ct = {ts: ct_bone_info(mesh_dir, ts) for ts in present
          if (ts in TS_TO_ATLAS and TS_TO_ATLAS[ts] in atlas) or ts in TS_COMPOSITE}
    matched = [ts for ts in present if ts in ct and ts in TS_TO_ATLAS]
    unmatched_ct = [ts for ts in present if ts not in matched]
    if not matched:
        raise SystemExit(f"no CT mesh in {mesh_dir} maps to an atlas bone (found: {present})")

    a_pts = np.array([atlas[TS_TO_ATLAS[ts]]["centroid"] for ts in matched])
    c_pts = np.array([ct[ts]["centroid_m"] for ts in matched])
    f = fit(a_pts, c_pts, min_bones)
    for w in f["warnings"]:
        print("WARNING:", w)

    print(f"\nfit: {f['mode']}  bones {len(matched)}  RMS {f['rms_mm']:.2f} mm  scale {f['scale']:.5f}  "
          f"rotation {f['rotation_deg']:.3f} deg about {np.round(f['rotation_axis'], 3).tolist()}  "
          f"translation {np.round(f['t'] * M_TO_MM, 1).tolist()} mm")
    print(f"\n{'ts bone':<18} {'atlas bone':<18} {'residual mm':>11}")
    for ts, r in zip(matched, f["residuals_mm"]):
        print(f"{ts:<18} {TS_TO_ATLAS[ts]:<18} {r:>11.2f}")

    sizes = size_table(atlas, ct, f["scale"])
    print(f"\n{'ts bone':<18} {'CT size x/y/z mm':>22} {'atlas*s x/y/z mm':>22} {'ratio x/y/z':>20}")
    for row in sizes:
        fmt = lambda v, d: "/".join(f"{x:.{d}f}" for x in v)  # noqa: E731
        print(f"{row['ts']:<18} {fmt(row['ct_size_mm'], 1):>22} {fmt(row['atlas_size_mm'], 1):>22} "
              f"{fmt(row['ratio'], 2):>20}{'  <-- check' if row['flag'] else ''}")
    if f["mode"] == "translation":
        print("(sizes use scale 1: the atlas scale is unresolved)")

    out_dir = Path(out) if out else run_dir / "atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    bones = {}
    atlas_meshes = Path(atlas_meshes) if atlas_meshes else Path(atlas_json).parent / "meshes"
    if refine:
        index = load_atlas_mesh_index(atlas_meshes)
        if index is None:
            print(f"\nWARNING: no {atlas_meshes / 'index.json'} (export_atlas_meshes.py): per-bone fit skipped")
        else:
            print(f"\nper-bone similarity ICP ({len(ct)} bones, labels at {voxel_mm} mm)")
            bones = refine_all(mesh_dir, sorted(ct), atlas, index, f["matrix"], out_dir, voxel_mm)
            print_bone_table(bones)
    carts, found = {}, []
    if bones and cartilage:
        print(f"\ncostal cartilages (placed by their rib's per-bone transform)")
        carts, found = fit_cartilages(bones, list(atlas), index, out_dir, voxel_mm)

    used_atlas = {TS_TO_ATLAS[ts] for ts in matched}
    result = {
        "run_dir": str(run_dir), "meshes_dir": str(mesh_dir), "atlas_json": str(atlas_json),
        "atlas_mapping": str(atlas_mapping) if atlas_mapping else "built-in (Z-Anatomy)",
        "mode": f["mode"],
        "matrix_atlas_to_ct": f["matrix"].tolist(), "matrix_layout": "row-major 4x4, metres, p_ct = M @ p_atlas",
        "scale": f["scale"], "rotation_deg": f["rotation_deg"], "rotation_axis": f["rotation_axis"],
        "translation_m": f["t"].tolist(), "rms_mm": f["rms_mm"], "warnings": f["warnings"],
        "pairs": [{"ts": ts, "atlas": TS_TO_ATLAS[ts], "atlas_centroid_m": a.tolist(),
                   "ct_centroid_m": c.tolist(), "residual_mm": float(r),
                   "ct_watertight": ct[ts]["watertight"]}
                  for ts, a, c, r in zip(matched, a_pts, c_pts, f["residuals_mm"])],
        "size_table": sizes,
        "unmatched_ct": unmatched_ct,
        "unmatched_atlas": sorted(n for n in atlas if n not in used_atlas),
        "ts_without_atlas_bone": missing,
    }
    if bones:
        result.update({
            "atlas_meshes": str(atlas_meshes),
            "bones_layout": "matrix_atlas_to_ct: row-major 4x4, metres, p_ct = M @ p_atlas (per bone); "
                            "scale = total, icp_scale = relative to the global fit; residuals mm",
            "bones": bones,
            "fallback_bones": sorted(ts for ts, b in bones.items() if b["fallback"]),
            "partial_bones": sorted(ts for ts, b in bones.items() if b["partial"]),
        })
    if cartilage and bones:
        result.update({"cartilages_layout": CARTILAGES_LAYOUT, "cartilage_parts_found": found,
                       "cartilages": carts})
    out_path = out_dir / "atlas_fit.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out_path}  (unmatched CT: {unmatched_ct or 'none'}; "
          f"unmatched atlas: {len(result['unmatched_atlas'])} bones)")
    if bones:
        print(f"wrote {len(bones)} fitted meshes to {out_dir / 'meshes'} and labels to {out_dir / 'labels'}"
              + (f" (+ {len(carts)} costal cartilages)" if carts else ""))
    print(f"done in {time.time() - t_start:.1f} s")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run directory containing meshes/<ts_name>.ply")
    ap.add_argument("--meshes-dir", default=None, help="CT meshes <ts_name>.ply (default RUN_DIR/meshes)")
    ap.add_argument("--atlas-json", default=str(DEFAULT_ATLAS_JSON))
    ap.add_argument("--atlas-meshes", default=None,
                    help="per-bone atlas PLYs + index.json (default <atlas-json dir>/meshes)")
    ap.add_argument("--voxel-mm", type=float, default=VOXEL_MM, help="label voxel size")
    ap.add_argument("--no-refine", action="store_true", help="global fit only: no per-bone meshes/labels")
    ap.add_argument("--atlas-mapping", default=None,
                    help="JSON {ts_name: atlas name or [part names]} replacing the built-in Z-Anatomy map")
    ap.add_argument("--fbx", default=str(DEFAULT_FBX))
    ap.add_argument("--out", default=None, help="output dir (default RUN_DIR/atlas)")
    ap.add_argument("--blender", action="store_true", help="import the fitted atlas into the running Blender")
    ap.add_argument("--collection", default="atlas")
    ap.add_argument("--min-bones", type=int, default=4, help="fewer matched bones -> translation-only fit")
    ap.add_argument("--cartilage", action=argparse.BooleanOptionalAction, default=True,
                    help="place each atlas costal cartilage with its rib's per-bone transform (default on)")
    ap.add_argument("--reuse-fit", action="store_true",
                    help="reuse the fits stored in <out>/atlas_fit.json (no refit), only add the cartilages")
    args = ap.parse_args(argv)
    if args.reuse_fit and args.no_refine:
        ap.error("--reuse-fit and --no-refine are exclusive")
    result = run(args.run_dir, args.atlas_json, args.out, args.min_bones, args.meshes_dir,
                 args.atlas_meshes, not args.no_refine, args.voxel_mm, args.atlas_mapping,
                 args.cartilage, args.reuse_fit)
    if args.blender:
        if args.atlas_mapping:
            print("WARNING: --blender imports the Z-Anatomy FBX (--fbx); the fit used another atlas mapping")
        import blender_bridge
        print(blender_bridge.run_code(blender_script(result["matrix_atlas_to_ct"], args.fbx, args.collection)))


if __name__ == "__main__":
    main()
