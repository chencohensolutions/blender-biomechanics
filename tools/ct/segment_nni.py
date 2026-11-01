#!/usr/bin/env python3
"""Learned bone segmentation at native CT resolution with nnInteractive.

Replaces the threshold + watershed + heuristics path of extract_bone.py.  Per bone:

  1. pick the acquisition group whose ORIGINAL axial series contains the bone,
  2. crop around the bone's TotalSegmentator (or fitted-atlas) label.  With --planes fused (default)
     the crop is an isotropic volume fused from the axial, coronal and sagittal series of the group
     (fuse_series.py: the console built the reformats from the thin raw reconstruction, so each
     series is sharp in two directions; the fusion takes every direction from the series that
     resolve it).  --planes axial uses the axial slices alone,
  3. give nnInteractive the coarse label as the initial segmentation, then positive point prompts
     deep inside the bone and negative point prompts inside each neighbouring bone's label,
  4. keep the largest connected component of the refined mask.

Then all bones of a group are resolved jointly (a contested voxel goes to the bone whose
uncontested core is nearer), meshed (Gaussian-smoothed mask iso-surface, Taubin smoothing),
written as PLY/STL in Blender metres (= LPS mm * 0.001, scanner origin kept) and optionally
imported into Blender.  For every touching pair the HU at the interface is recorded: a gap drawn
through tissue denser than --measured-max-hu is INFERRED by the model, not measured.

Model: nnInteractive v1.0 (DKFZ, CC BY-NC-SA 4.0 weights; code Apache-2.0),
weights from Hugging Face `nnInteractive/nnInteractive` (see --model-dir).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # adaptive_avg_pool3d is missing on MPS

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402
import fuse_series  # noqa: E402
import pipeline  # noqa: E402

MM_TO_BLENDER = 0.001
DEFAULT_MODEL = Path.home() / ".cache" / "nnInteractive" / "nnInteractive_v1.0"


# ----------------------------------------------------------------------------- groups / labels
def load_groups(run):
    """Acquisition groups from run.json: ORIGINAL axial volume + TotalSegmentator folder each."""
    manifest = json.loads((run / "run.json").read_text())
    groups = []
    for name, g in manifest["groups"].items():
        if "+" in name:  # e.g. Lung+CHEST+ABD: the user excluded the 3 mm chest series
            continue
        axial = [Path(v) for v in g["volumes"] if "_axial" in Path(v).name]
        if not axial:
            continue
        # finest ORIGINAL axial series of the group
        axial.sort(key=lambda p: sitk.ReadImage(str(p)).GetSpacing()[2])
        vol = axial[0]
        nii = vol.with_suffix("").with_suffix(".nii.gz") if vol.suffix == ".nrrd" else vol
        groups.append({"name": name, "volume": nii if nii.exists() else vol, "totalseg": run / "totalseg" / name,
                       "bounds": [np.array(b) for b in g["bounds_lps"]], "volumes": [vol],
                       "series": fuse_series.group_series(run, name)})
    order = {"Lung": 0, "Delayed": 1}
    groups.sort(key=lambda g: order.get(g["name"], 9))
    return groups


class Volume:
    """HU volume of one acquisition group on a regular LPS-aligned grid, with its guide labels.

    planes="axial": the finest ORIGINAL axial series at its native voxels, loaded once.
    planes="fused": an isotropic grid (fused_mm) over the same extent; HU is produced per crop by
    fuse_series.fuse_crop from the axial + coronal + sagittal series (cached, a few crops at a time).
    """

    def __init__(self, group, atlas_labels=None, planes="fused", fused_mm=0.6):
        self.group = group
        self.planes = planes
        axial = sitk.ReadImage(str(group["volume"]))
        assert np.allclose(np.array(axial.GetDirection()), np.eye(3).ravel(), atol=1e-3), "axial series expected"
        self.origin = np.array(axial.GetOrigin())  # LPS of voxel (0,0,0)
        if planes == "axial":
            self.hu = sitk.GetArrayFromImage(axial).astype(np.float32)  # (z, y, x)
            self.spacing = np.array(axial.GetSpacing())[::-1]  # (z, y, x) mm
            self.shape = self.hu.shape
            self.series = [fuse_series.Series(group["volume"])]
        else:
            self.hu = None
            self.series = group["series"]
            extent = (np.array(axial.GetSize()) - 1) * np.array(axial.GetSpacing())  # xyz mm
            self.spacing = np.array([fused_mm] * 3)
            self.shape = tuple(int(np.floor(e / fused_mm)) + 1 for e in extent[::-1])
        self.series_names = "+".join(s.name for s in self.series)
        self.atlas_labels = atlas_labels
        self._labels = {}
        self._crops = {}

    MAX_CACHED = 4  # full-grid bool labels are ~230-290 MB each
    MAX_CROPS = 3

    @property
    def ref(self):
        """A SimpleITK image on this grid (no meaningful pixels) to resample other images onto."""
        if not hasattr(self, "_ref"):
            size = [int(self.shape[2]), int(self.shape[1]), int(self.shape[0])]
            self._ref = sitk.Image(size, sitk.sitkUInt8)
            self._ref.SetSpacing(self.spacing[::-1].tolist())
            self._ref.SetOrigin(self.origin.tolist())
        return self._ref

    def hu_crop(self, sl):
        """HU in the (z, y, x) slice tuple `sl` of this grid."""
        if self.hu is not None:
            return self.hu[sl]
        sl = tuple(slice(max(s.start or 0, 0), min(s.stop if s.stop is not None else n, n))
                   for s, n in zip(sl, self.shape))
        key = tuple((s.start, s.stop) for s in sl)
        if key not in self._crops:
            origin = self.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * self.spacing[::-1]
            shape = tuple(s.stop - s.start for s in sl)
            t = time.time()
            self._crops[key] = fuse_series.fuse_crop(self.series, origin, float(self.spacing[0]), shape)
            while len(self._crops) > self.MAX_CROPS:
                self._crops.pop(next(iter(self._crops)))
        return self._crops[key]

    def label(self, bone):
        if bone not in self._labels:
            path = pipeline.label_path(bone, self.group, self.atlas_labels)
            if path is None:
                self._labels[bone] = None
            else:
                img = sitk.ReadImage(str(path))
                if img.GetSize() != self.ref.GetSize() or not np.allclose(img.GetSpacing(), self.ref.GetSpacing()):
                    img = sitk.Resample(img, self.ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                self._labels[bone] = sitk.GetArrayFromImage(img) > 0
            while len(self._labels) > self.MAX_CACHED:
                self._labels.pop(next(iter(self._labels)))
        return self._labels[bone]


def crop_box(mask, margin_mm, spacing, shape):
    zs, ys, xs = np.where(mask)
    m = np.ceil(margin_mm / spacing).astype(int)
    return tuple(slice(max(int(a.min()) - int(mm), 0), min(int(a.max()) + int(mm) + 1, int(n)))
                 for a, mm, n in zip((zs, ys, xs), m, shape))


def interior_points(mask, spacing, n, min_sep_mm):
    """Up to n voxels deep inside `mask`, greedily separated by min_sep_mm (distance-transform maxima)."""
    if not mask.any():
        return []
    dt = ndi.distance_transform_edt(mask, sampling=spacing)
    pts = []
    work = dt.copy()
    for _ in range(n):
        p = np.unravel_index(int(work.argmax()), work.shape)
        if work[p] <= 0:
            break
        pts.append(tuple(int(i) for i in p))
        zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        d2 = (((zz - p[0]) * spacing[0]) ** 2 + ((yy - p[1]) * spacing[1]) ** 2 + ((xx - p[2]) * spacing[2]) ** 2)
        work[d2 < min_sep_mm ** 2] = 0
    return pts


def largest_component(mask):
    comp, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, comp, index=np.arange(1, n + 1))
    return comp == (int(np.argmax(sizes)) + 1)


# ----------------------------------------------------------------------------- nnInteractive
class Segmenter:
    def __init__(self, model_dir, device):
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        self.torch = torch
        self.session = nnInteractiveInferenceSession(device=torch.device(device), use_torch_compile=False,
                                                     verbose=False, torch_n_threads=8, do_autozoom=True)
        self.session.initialize_from_trained_model_folder(str(model_dir))

    def run(self, hu_crop, init_seg, pos_points, neg_points, log=print):
        ses = self.session
        ses.set_image(hu_crop[None].astype(np.float32))
        out = np.zeros(hu_crop.shape, dtype=np.uint8)
        ses.set_target_buffer(out)
        t = time.time()
        ses.add_initial_seg_interaction(init_seg.astype(np.uint8), run_prediction=True)
        log(f"    initial label refined in {time.time() - t:.1f} s ({int(init_seg.sum())} -> {int(out.sum())} vox)")
        for p in pos_points:
            t = time.time()
            ses.add_point_interaction(p, include_interaction=True)
            log(f"    + point {p}: {time.time() - t:.1f} s -> {int(out.sum())} vox")
        for name, p in neg_points:
            t = time.time()
            ses.add_point_interaction(p, include_interaction=False)
            log(f"    - point {p} in {name}: {time.time() - t:.1f} s -> {int(out.sum())} vox")
        return out > 0


def segment_bone(bone, vol, seg, args, out_dir, log=print):
    """Refine one bone; writes <bone>.nii.gz (crop) + <bone>.json.  Returns the json dict."""
    lab = vol.label(bone)
    if lab is None or not lab.any():
        raise RuntimeError(f"no guide label for {bone} in group {vol.group['name']}")
    sl = crop_box(lab, args.margin_mm, vol.spacing, vol.shape)
    t_hu = time.time()
    hu = vol.hu_crop(sl)
    if vol.hu is None:
        log(f"  {bone}: fused {hu.shape} crop from {vol.series_names} in {time.time() - t_hu:.0f} s")
    init = lab[sl]
    pos = interior_points(init, vol.spacing, args.pos_points, args.point_sep_mm)
    neg = []
    near = None
    for nb in pipeline.neighbours(bone):
        nl = vol.label(nb)
        if nl is None:
            continue
        nl = nl[sl]
        if not nl.any():
            continue
        # the part of the neighbour nearest to this bone matters most: restrict to within 40 mm of it
        if near is None:
            near = ndi.distance_transform_edt(~init, sampling=vol.spacing) < 40.0
        cand = nl & near
        pts = interior_points(cand if cand.any() else nl, vol.spacing, args.neg_points, args.point_sep_mm)
        neg += [(nb, p) for p in pts]
    log(f"  {bone}: crop {hu.shape}, {len(pos)} positive, {len(neg)} negative prompts")
    t0 = time.time()
    mask = seg.run(hu, init, pos, neg, log=log)
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    elapsed = time.time() - t0
    vox_mm3 = float(np.prod(vol.spacing))
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(vol.spacing[::-1].tolist())
    img.SetOrigin((vol.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * vol.spacing[::-1]).tolist())
    sitk.WriteImage(img, str(out_dir / f"{bone}.nii.gz"), useCompression=True)
    info = {"bone": bone, "group": vol.group["name"], "series": vol.series_names, "planes": vol.planes,
            "spacing_mm_xyz": vol.spacing[::-1].tolist(), "crop_zyx": [[s.start, s.stop] for s in sl],
            "guide_voxels": int(init.sum()), "voxels": int(mask.sum()), "volume_cm3": float(mask.sum() * vox_mm3 / 1000),
            "pos_points": pos, "neg_points": neg, "seconds": round(elapsed, 1),
            "hu_median": float(np.median(hu[mask])) if mask.any() else None,
            "frac_below_100hu": float((hu[mask] < 100).mean()) if mask.any() else None,
            "model": "nnInteractive_v1.0"}
    (out_dir / f"{bone}.json").write_text(json.dumps(info, indent=1))
    log(f"  {bone}: {info['volume_cm3']:.1f} cm^3, median {info['hu_median']:.0f} HU, {elapsed:.0f} s")
    return info


# ----------------------------------------------------------------------------- resolve + mesh
def load_mask(path, vol):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    start = np.round((np.array(img.GetOrigin()) - vol.origin) / vol.spacing[::-1]).astype(int)[::-1]  # z,y,x
    sl = tuple(slice(int(s), int(s) + int(n)) for s, n in zip(start, arr.shape))
    return arr, sl


def resolve_group(bones, vol, masks_dir, log=print):
    """Full-grid label volume for all `bones` of one group; contested voxels go to the nearer core."""
    labels = np.zeros(vol.shape, dtype=np.int16)
    claims = np.zeros(vol.shape, dtype=np.int8)
    parts = {}
    for i, b in enumerate(bones, start=1):
        p = masks_dir / f"{b}.nii.gz"
        if not p.exists():
            continue
        arr, sl = load_mask(p, vol)
        parts[i] = (b, arr, sl)
        claims[sl][arr] += 1
    contested_total = 0
    for i, (b, arr, sl) in parts.items():
        own = arr & (claims[sl] == 1)
        labels[sl][own] = i
    contested = claims > 1
    if contested.any():
        best_d = np.full(vol.shape, np.inf, dtype=np.float32)
        best_i = np.zeros(vol.shape, dtype=np.int16)
        for i, (b, arr, sl) in parts.items():
            mine = arr & contested[sl]
            if not mine.any():
                continue
            # work only around this bone's contested voxels (EDT on the full grid would be far too slow)
            zs, ys, xs = np.where(mine)
            box = tuple(slice(max(s.start + int(a.min()) - 4, 0), min(s.start + int(a.max()) + 5, n))
                        for s, a, n in zip(sl, (zs, ys, xs), vol.shape))
            core = labels[box] == i  # uncontested part of this bone
            if not core.any():
                continue
            d = ndi.distance_transform_edt(~core, sampling=vol.spacing)
            full = np.zeros(labels[box].shape, dtype=bool)
            inter = tuple(slice(max(b_.start, s.start), min(b_.stop, s.stop)) for b_, s in zip(box, sl))
            full[tuple(slice(i.start - b_.start, i.stop - b_.start) for i, b_ in zip(inter, box))] = \
                arr[tuple(slice(i.start - s.start, i.stop - s.start) for i, s in zip(inter, sl))]
            win = contested[box] & full & (d < best_d[box])
            bd, bi = best_d[box], best_i[box]
            bd[win] = d[win]
            bi[win] = i
            best_d[box], best_i[box] = bd, bi
        labels[contested] = best_i[contested]
        contested_total = int(contested.sum())
        del best_d, best_i
    log(f"  group {vol.group['name']}: {len(parts)} bones, {contested_total} contested voxels resolved")
    return labels, {i: b for i, (b, _, _) in parts.items()}


def carve_across_groups(resolved, log=print):
    """Neighbouring bones segmented on different series were resolved separately and may overlap:
    where a bone of the finer series exists, the coarser series' neighbour gives way."""
    for ia in range(len(resolved)):
        for ib in range(len(resolved)):
            if ia == ib:
                continue
            gf, vf, lf, idf, _ = resolved[ia]
            gc, vc, lc, idc, _ = resolved[ib]
            if vf.spacing.max() > vc.spacing.max():
                continue  # only carve the coarser one
            if vf.spacing.max() == vc.spacing.max() and ia > ib:
                continue
            for i, bf in idf.items():
                targets = [j for j, bc in idc.items() if bc in pipeline.neighbours(bf) or bf in pipeline.neighbours(bc)]
                if not targets:
                    continue
                m = lf == i
                if not m.any():
                    continue
                zs, ys, xs = np.where(m)
                sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in (zs, ys, xs))
                crop = sitk.GetImageFromArray(m[sl].astype(np.uint8))
                crop.SetSpacing(vf.spacing[::-1].tolist())
                crop.SetOrigin((vf.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * vf.spacing[::-1]).tolist())
                on_coarse = sitk.GetArrayFromImage(
                    sitk.Resample(crop, vc.ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
                for j in targets:
                    hit = on_coarse & (lc == j)
                    if hit.any():
                        lc[hit] = 0
                        log(f"  cross-series: {bf} ({gf['name']}) takes {int(hit.sum())} voxels from "
                            f"{idc[j]} ({gc['name']})")


def contact_report(labels, ids, vol, measured_max_hu):
    """For each touching pair: interface voxel count and HU there (dense = inferred boundary)."""
    report = {}
    struct = ndi.generate_binary_structure(3, 1)
    for i, b in ids.items():
        m = labels == i
        if not m.any():
            continue
        zs, ys, xs = np.where(m)
        box = tuple(slice(max(int(a.min()) - 2, 0), int(a.max()) + 3) for a in (zs, ys, xs))
        sub = labels[box]
        ring = ndi.binary_dilation(sub == i, struct) & (sub != i) & (sub > 0)
        if not ring.any():
            continue
        hu = vol.hu_crop(box)
        for j in np.unique(sub[ring]):
            sel = ring & (sub == j)
            vals = hu[sel]
            key = f"{b}:{ids[int(j)]}"
            report[key] = {"interface_voxels": int(sel.sum()),
                           "interface_mm2": float(sel.sum() * vol.spacing[1] * vol.spacing[2]),
                           "hu_median": float(np.median(vals)),
                           "frac_dense": float((vals > measured_max_hu).mean()),
                           "boundary": "inferred" if np.median(vals) > measured_max_hu else "measured"}
    return report


def mesh_bone(bone, idx, labels, vol, out_dir, smooth_mm, taubin_iters, extra, smooth_min_vox=0.0):
    import trimesh
    from skimage import measure
    m = labels == idx
    if not m.any():
        return None
    zs, ys, xs = np.where(m)
    pad = 3
    sl = tuple(slice(max(int(a.min()) - pad, 0), min(int(a.max()) + pad + 1, n)) for a, n in zip((zs, ys, xs), m.shape))
    f = np.pad(m[sl].astype(np.float32), 1)
    sigma = np.maximum(smooth_mm / vol.spacing, smooth_min_vox)  # thick slices: smooth at least this many voxels
    if sigma.max() > 0:
        f = ndi.gaussian_filter(f, sigma=sigma)
    verts, faces, _, _ = measure.marching_cubes(f, level=0.5, spacing=tuple(vol.spacing), allow_degenerate=False)
    offset = (np.array([s.start for s in sl], dtype=float) - 1.0) * vol.spacing
    verts_lps = vol.origin + (verts + offset)[:, ::-1]
    mesh = trimesh.Trimesh(verts_lps, faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: p.area)
    if taubin_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=taubin_iters)
    mesh.fix_normals()
    bl = mesh.copy()
    bl.apply_scale(MM_TO_BLENDER)
    ply, stl = out_dir / f"{bone}.ply", out_dir / f"{bone}.stl"
    bl.export(ply)
    bl.export(stl)
    lo, hi = mesh.bounds
    info = {"name": bone, "method": "nnInteractive_v1.0 + joint resolve" + (" on fused planes" if vol.planes == "fused" else ""),
            "group": vol.group["name"], "series": vol.series_names, "planes": vol.planes,
            "spacing_mm_xyz": vol.spacing[::-1].tolist(),
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "volume_cm3": float(m.sum() * np.prod(vol.spacing) / 1000), "bbox_lps_mm_min": lo.tolist(),
            "bbox_lps_mm_max": hi.tolist(), "size_mm": (hi - lo).tolist(), "watertight": bool(mesh.is_watertight),
            "coords": "Blender = LPS mm * 0.001 (x=left, y=posterior, z=superior), scanner origin kept"}
    info.update(extra)
    (out_dir / f"{bone}.json").write_text(json.dumps(info, indent=1))
    return info


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pipeline run folder (has run.json, series/, totalseg/)")
    ap.add_argument("--bones", nargs="+", default=["all"], help="bone names (T8, rib_left_3, ...) or 'all'")
    ap.add_argument("--out", help="output folder (default RUN/nni_fused, or RUN/nni with --planes axial)")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="mps", help="mps | cpu | cuda")
    ap.add_argument("--atlas-labels", help="fitted atlas labels, used as guide where TotalSegmentator has none")
    ap.add_argument("--planes", choices=["fused", "axial"], default="fused",
                    help="fused: isotropic crops fused from axial+coronal+sagittal (fuse_series.py); axial: native axial only")
    ap.add_argument("--fused-mm", type=float, default=0.6, help="isotropic voxel size of the fused grid")
    ap.add_argument("--margin-mm", type=float, default=25.0, help="crop margin around the guide label")
    ap.add_argument("--group-margin-mm", type=float, default=5.0,
                    help="a group is used when its ORIGINAL axial volume contains the guide label with this margin; "
                         "0 lets a bone at the very edge of the finer series still use it")
    ap.add_argument("--pos-points", type=int, default=2)
    ap.add_argument("--neg-points", type=int, default=1, help="negative prompts per neighbouring bone")
    ap.add_argument("--point-sep-mm", type=float, default=20.0)
    ap.add_argument("--group-override", nargs="*", default=[], metavar="BONE=GROUP",
                    help="use this acquisition group for a bone regardless of label coverage "
                         "(e.g. rib_right_11=Lung when the rib just touches the edge of the finer series)")
    ap.add_argument("--force", action="store_true", help="re-segment bones that already have a mask")
    ap.add_argument("--skip-segment", action="store_true", help="only resolve + mesh existing masks")
    ap.add_argument("--resolve-with", nargs="*", default=[], help="extra existing masks to include when resolving")
    ap.add_argument("--smooth-mm", type=float, default=0.5, help="Gaussian on the mask before the iso-surface")
    ap.add_argument("--smooth-min-vox", type=float, default=0.0,
                    help="minimum Gaussian sigma in voxels per axis, so 3 mm slices do not leave stair steps "
                         "(0.8 vox = 2.4 mm across 3 mm slices, 0.36 mm in a 0.45 mm plane)")
    ap.add_argument("--smooth-iters", type=int, default=10, help="Taubin iterations")
    ap.add_argument("--measured-max-hu", type=float, default=450.0,
                    help="interface denser than this = no visible joint space = boundary inferred by the model")
    ap.add_argument("--blender", action="store_true", help="import the meshes into the running Blender")
    ap.add_argument("--collection", default="bones")
    ap.add_argument("--run-tag", default="ct2021-nni-fused")
    ap.add_argument("--suffix", default="", help="object name suffix in Blender (pilot: keep old bones)")
    ap.add_argument("--replace", action="store_true", help="remove objects of other runs from the collection first")
    args = ap.parse_args(argv)

    run = Path(args.run)
    out = Path(args.out) if args.out else run / ("nni_fused" if args.planes == "fused" else "nni")
    masks_dir, meshes_dir, qa_dir = out / "masks", out / "meshes", out / "qa"
    for d in (masks_dir, meshes_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)
    bones = pipeline.ALL_BONES if args.bones == ["all"] else [pipeline.canonical(b) for b in args.bones]
    bones = [b for b in bones if b != "costal_cartilages"]
    groups = load_groups(run)
    print("groups:", ", ".join(f"{g['name']} <- " + ("+".join(s.name for s in g["series"]) if args.planes == "fused"
                                                       else Path(g["volume"]).name) for g in groups))
    if args.planes == "fused":
        print(f"fused isotropic grid {args.fused_mm} mm; series:", {g["name"]: [repr(s) for s in g["series"]] for g in groups})

    volumes = {}

    def volume(g):
        if g["name"] not in volumes:
            volumes[g["name"]] = Volume(g, args.atlas_labels, planes=args.planes, fused_mm=args.fused_mm)
        return volumes[g["name"]]

    overrides = {pipeline.canonical(k): v for k, v in (o.split("=", 1) for o in args.group_override)}
    assignment = {}
    for b in bones:
        if b in overrides:
            assignment[b] = overrides[b]
            continue
        g, _ = pipeline.choose_group(b, groups, margin=args.group_margin_mm, atlas_labels=args.atlas_labels)
        if g is None:
            print(f"!! {b}: no guide label in any group, skipped")
            continue
        assignment[b] = g["name"]

    # 1. segment
    if not args.skip_segment:
        seg = Segmenter(args.model_dir, args.device)
        for b, gname in assignment.items():
            if (masks_dir / f"{b}.nii.gz").exists() and not args.force:
                print(f"  cached {b}")
                continue
            g = next(x for x in groups if x["name"] == gname)
            try:
                segment_bone(b, volume(g), seg, args, masks_dir)
            except (RuntimeError, ValueError, MemoryError) as exc:
                print(f"!! {b}: {exc}")

    # 2. resolve per group, then across groups (finer series wins), then contact report + mesh
    results = {}
    contacts = {}
    resolved = []
    for g in groups:
        members = [b for b, gn in assignment.items() if gn == g["name"]]
        extra = [pipeline.canonical(b) for b in args.resolve_with]
        extra = [b for b in extra if b not in members and (masks_dir / f"{b}.nii.gz").exists()
                 and json.loads((masks_dir / f"{b}.json").read_text())["group"] == g["name"]]
        if not members:
            continue
        vol = volume(g)
        labels, ids = resolve_group(members + extra, vol, masks_dir)
        resolved.append((g, vol, labels, ids, members))
    carve_across_groups(resolved)
    for g, vol, labels, ids, members in resolved:
        contacts.update(contact_report(labels, ids, vol, args.measured_max_hu))
        for i, b in ids.items():
            if b not in members:
                continue
            seg_info = json.loads((masks_dir / f"{b}.json").read_text())
            pairs = {k.split(":")[1]: v for k, v in contacts.items() if k.startswith(b + ":")}
            info = mesh_bone(b, i, labels, vol, meshes_dir, args.smooth_mm, args.smooth_iters,
                             {"segment": {k: seg_info[k] for k in ("seconds", "hu_median", "frac_below_100hu", "guide_voxels")},
                              "contacts": pairs}, smooth_min_vox=args.smooth_min_vox)
            if info:
                results[b] = info
                flags = ", ".join(f"{k}={v['boundary']}" for k, v in pairs.items())
                print(f"  meshed {b}: {info['volume_cm3']:.1f} cm^3, {info['faces']} faces"
                      + (f"  [{flags}]" if flags else ""))
    # merge with earlier runs in the same output folder (a partial re-run must not drop the other bones)
    def merged(path, key, new):
        prev = json.loads(path.read_text()) if path.exists() else {}
        prev = prev.get(key, prev) if key else prev
        prev = {k: v for k, v in prev.items() if not any(b in k.split(":") for b in results)} if key is None else \
            {k: v for k, v in prev.items() if k not in results}
        prev.update(new)
        return prev
    all_contacts = merged(qa_dir / "contacts.json", None, contacts)
    (qa_dir / "contacts.json").write_text(json.dumps(all_contacts, indent=1))
    all_results = merged(out / "results.json", "bones", results)
    (out / "results.json").write_text(json.dumps({"run_tag": args.run_tag, "bones": all_results}, indent=1))

    # 3. Blender
    if args.blender and results:
        try:
            if args.replace:
                print(blender_bridge.run_code(pipeline.REMOVE_OTHER_RUNS % {"collection": args.collection,
                                                                             "run_tag": args.run_tag}))
            for b, info in results.items():
                props = {"ct_run": args.run_tag, "ct_method": info["method"], "ct_group": info["group"], "ct_planes": info["planes"],
                         "ct_series": info["series"], "ct_voxel_mm": [round(x, 3) for x in info["spacing_mm_xyz"]],
                         "ct_inferred_boundaries": [k for k, v in info["contacts"].items() if v["boundary"] == "inferred"]}
                print("  " + str(blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                    "path": str(meshes_dir / f"{b}.ply"), "name": b + args.suffix, "collection": args.collection,
                    "decimate": 0.0, "props": props})).strip())
        except (ConnectionRefusedError, OSError) as exc:
            print(f"!! Blender not reachable on localhost:9876 ({exc}); meshes stay on disk")
    print(f"done: {len(results)} meshes in {meshes_dir}")


if __name__ == "__main__":
    main()
