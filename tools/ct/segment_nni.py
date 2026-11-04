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
import contextlib
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

# Muscles with a TotalSegmentator guide label (tasks: total, abdominal_muscles, headneck_muscles).
# psoas_major (abdominal_muscles) is left out because iliopsoas (total) already covers it plus the
# iliacus; autochthon (total) is left out because erector_spinae + transversospinalis split it.
MUSCLES_TOTAL = [f"gluteus_{m}_{s}" for m in ("maximus", "medius", "minimus") for s in ("left", "right")] + \
                ["iliopsoas_left", "iliopsoas_right"]
MUSCLES_ABDOMINAL = [f"{m}_{s}" for m in ("pectoralis_major", "rectus_abdominis", "serratus_anterior", "latissimus_dorsi",
                                          "trapezius", "external_oblique", "internal_oblique", "erector_spinae",
                                          "transversospinalis", "quadratus_lumborum") for s in ("left", "right")]
MUSCLES_HEADNECK = [f"{m}_{s}" for m in ("sternocleidomastoid", "levator_scapulae", "anterior_scalene", "middle_scalene",
                                         "posterior_scalene", "prevertebral", "platysma") for s in ("left", "right")]
MUSCLES = MUSCLES_TOTAL + MUSCLES_ABDOMINAL + MUSCLES_HEADNECK

# Organs and vessels with a free TotalSegmentator guide label (task total), grouped by the Blender
# collection they are imported into.  Kidney cysts are left out; the heart is one body here (the
# chambers need the licensed heartchambers_highres task).
ORGAN_COLLECTIONS = {
    "thorax": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
               "lung_lower_lobe_right", "trachea", "esophagus", "thyroid_gland", "heart", "atrial_appendage_left"],
    "abdomen": ["liver", "spleen", "kidney_right", "kidney_left", "adrenal_gland_right", "adrenal_gland_left",
                "gallbladder", "pancreas", "stomach", "duodenum", "small_bowel", "colon", "urinary_bladder", "prostate"],
    "vessels": ["aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
                "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left",
                "brachiocephalic_vein_right", "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
                "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right"],
    "nervous": ["spinal_cord"],
}
ORGANS = [o for col in ORGAN_COLLECTIONS.values() for o in col]
COLLECTION_OF = {o: col for col, names in ORGAN_COLLECTIONS.items() for o in names}
# Heart in detail: the licensed TotalSegmentator tasks heartchambers_highres (the chambers, myocardium,
# ascending aorta, pulmonary artery), coronary_arteries and aortic_sinuses as guides, refined on a
# 0.6 mm fused crop into Blender collection `heart`.  heartchambers' "aorta" is stored as heart_aorta
# so it does not overwrite the whole-aorta label of task total (collection vessels).
HEART = ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "heart_atrium_right", "heart_ventricle_right",
         "heart_aorta", "pulmonary_artery", "coronary_arteries",
         "left_ventricular_outflow_tract", "right_coronary_cusp", "left_coronary_cusp", "non_coronary_cusp"]
HEART_ENVELOPE = ["heart", "atrial_appendage_left", "aorta"]  # organ-run masks the heart parts lie inside: not negatives
# parts that lie inside another part of the same run (cusps in the aortic root, the outflow tract between the left
# ventricle and the aorta, coronaries on the myocardium): they keep every voxel they claim, in this order, or the
# host (which has the only uncontested core) would take them whole
NESTED = {"heart": ["right_coronary_cusp", "left_coronary_cusp", "non_coronary_cusp",
                    "left_ventricular_outflow_tract", "coronary_arteries"]}
# Intervertebral discs: TotalSegmentator's one intervertebral_discs label (task vertebrae_body, licensed)
# split per disc by split_discs.py (disc_T7_T8 ... disc_L5_S1), refined on a 0.6 mm crop with the
# vertebra masks as negatives and carve, into Blender collection `discs`.
import split_discs  # noqa: E402
DISCS = split_discs.DISCS
STRUCTURES = {"muscles": MUSCLES, "organs": ORGANS, "heart": HEART, "discs": DISCS}  # soft tissue (bones: pipeline.ALL_BONES)
SINGULAR = {"bones": "bone", "muscles": "muscle", "organs": "organ", "heart": "heart", "discs": "disc"}
GUIDE_IS_SPACE = ("discs",)  # the guide is the whole space the structure fills (the inter-body gap): the mask is unioned with it
FINE = ("bones", "heart", "discs")  # 0.6 mm crops, resolution-weighted fusion (sharp edges decide, not noise)


# ----------------------------------------------------------------------------- groups / labels
SOFT_TISSUE_SERIES = ["401_axial", "402_coronal", "403_sagittal"]  # the 3 mm soft-kernel chest+abdomen acquisition


def load_groups(run, merged=False, extra_series=None):
    """Acquisition groups from run.json: ORIGINAL axial volume + TotalSegmentator folder each.

    merged=True returns ONE group "Body" spanning every acquisition (series_groups = the series of
    each acquisition, totalseg = every group's label folder), for structures such as the erector
    spinae that run through both the chest and the abdomen scan.  Valid because the acquisitions
    of this scan line up to 0.01 mm (fuse_series.estimate_shift on their overlap).  extra_series
    names series files (RUN/series/<name>.nrrd|.nii.gz) of a further acquisition that only the
    merged group uses: the 3 mm soft-kernel chest+abdomen scan, excluded for bones (the user's
    rule) but the lowest-noise source for soft tissue (user, 2026-11-01: "use 3mm if its less
    noise for soft tissues, use all 3 planes")."""
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
    if merged and len(groups) > 1:
        acquisitions = list(groups)
        for names in (extra_series or []):
            paths = []
            for n in names:
                for ext in (".nrrd", ".nii.gz"):
                    if (run / "series" / f"{n}{ext}").exists():
                        paths.append(run / "series" / f"{n}{ext}")
                        break
            ss = [fuse_series.Series(str(p)) for p in paths]
            axial = next((x for x in ss if x.normal_axis == 2), None)
            if axial is None:
                print(f"!! extra acquisition {names}: no axial series found, skipped")
                continue
            o = np.array(axial.img.GetOrigin()); e = o + (np.array(axial.img.GetSize()) - 1) * np.array(axial.img.GetSpacing())
            acquisitions.append({"name": "+".join(names), "volume": paths[ss.index(axial)], "totalseg": None,
                                 "series": ss, "bounds": [np.minimum(o, e), np.maximum(o, e)], "volumes": paths})
        lo = np.min([g["bounds"][0] for g in acquisitions], axis=0)
        hi = np.max([g["bounds"][1] for g in acquisitions], axis=0)
        return [{"name": "Body", "volume": groups[0]["volume"], "totalseg": [g["totalseg"] for g in groups],
                 "series_groups": [g["series"] for g in acquisitions],
                 "series": [s for g in acquisitions for s in g["series"]],
                 "bounds": [lo, hi], "volumes": [v for g in acquisitions for v in g["volumes"]],
                 "members": [g["name"] for g in acquisitions]}]
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
        self.series_groups = group.get("series_groups")
        if self.series_groups:  # merged acquisitions: isotropic grid over the union of their extents
            assert planes == "fused", "a merged group needs --planes fused"
            self.hu = None
            self.series = group["series"]
            self.origin = np.array(group["bounds"][0], dtype=float)
            extent = np.array(group["bounds"][1], dtype=float) - self.origin
            self.spacing = np.array([fused_mm] * 3)
            self.shape = tuple(int(np.floor(e / fused_mm)) + 1 for e in extent[::-1])
        elif planes == "axial":
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
        self._bounds = {}
        self.totalseg_dirs = group["totalseg"] if isinstance(group["totalseg"], list) else [group["totalseg"]]

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
            if self.series_groups and len(self.series_groups) > 1:
                self._crops[key] = fuse_series.fuse_crop_multi(self.series_groups, origin, float(self.spacing[0]), shape,
                                                               quality=getattr(self, "quality", "resolution"))
            else:
                self._crops[key] = fuse_series.fuse_crop(self.series, origin, float(self.spacing[0]), shape)
            while len(self._crops) > self.MAX_CROPS:
                self._crops.pop(next(iter(self._crops)))
        return self._crops[key]

    def label_paths(self, name):
        """Guide label files for `name`: one per acquisition folder that has it (else the atlas label)."""
        out = []
        for d in self.totalseg_dirs:
            p = pipeline.label_path(name, {"totalseg": d}, None)
            if p is not None:
                out.append(p)
        if not out and self.atlas_labels:
            p = pipeline.label_path(name, {"totalseg": self.totalseg_dirs[0]}, self.atlas_labels)
            if p is not None:
                out.append(p)
        return out

    def guide_bounds(self, name):
        """(lo, hi) LPS bounds of the guide label over all its files, or None when empty/missing."""
        if name not in self._bounds:
            bs = [b for b in (pipeline.label_bounds(p) for p in self.label_paths(name)) if b is not None]
            self._bounds[name] = None if not bs else (np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0),
                                                      int(sum(b[2] for b in bs)))
        return self._bounds[name]

    def box_from_bounds(self, lo, hi, margin_mm):
        """(z, y, x) slice tuple of this grid covering LPS bounds lo..hi plus a margin, clipped to the grid."""
        lo_i = np.floor((np.asarray(lo) - margin_mm - self.origin) / self.spacing[::-1]).astype(int)[::-1]
        hi_i = np.ceil((np.asarray(hi) + margin_mm - self.origin) / self.spacing[::-1]).astype(int)[::-1] + 1
        return tuple(slice(max(int(a), 0), min(int(b), int(n))) for a, b, n in zip(lo_i, hi_i, self.shape))

    def crop_ref(self, sl):
        """A SimpleITK image covering the crop window `sl` of this grid (to resample other images into)."""
        size = [int(sl[2].stop - sl[2].start), int(sl[1].stop - sl[1].start), int(sl[0].stop - sl[0].start)]
        ref = sitk.Image(size, sitk.sitkUInt8)
        ref.SetSpacing(self.spacing[::-1].tolist())
        ref.SetOrigin((self.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * self.spacing[::-1]).tolist())
        return ref

    def mask_crop(self, path, sl):
        """A saved label/mask file resampled (nearest) into the crop window `sl`; bool (z, y, x)."""
        img = sitk.ReadImage(str(path))
        img = sitk.Resample(img, self.crop_ref(sl), sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.GetArrayFromImage(img) > 0

    def label_crop(self, name, sl):
        """Guide label of `name` inside the crop window `sl` (union over acquisitions), or None."""
        paths = self.label_paths(name)
        if not paths:
            return None
        out = None
        for p in paths:
            arr = self.mask_crop(p, sl)
            out = arr if out is None else (out | arr)
        return out

    def label(self, bone):
        """Full-grid guide label (kept for callers that need it; segment_structure works on crops)."""
        if bone not in self._labels:
            paths = self.label_paths(bone)
            if not paths:
                self._labels[bone] = None
            else:
                out = None
                for path in paths:
                    img = sitk.ReadImage(str(path))
                    if img.GetSize() != self.ref.GetSize() or not np.allclose(img.GetSpacing(), self.ref.GetSpacing()) \
                            or not np.allclose(img.GetOrigin(), self.ref.GetOrigin()):
                        img = sitk.Resample(img, self.ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                    arr = sitk.GetArrayFromImage(img) > 0
                    out = arr if out is None else (out | arr)
                self._labels[bone] = out
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


def guided_components(mask, guide, log=print, min_frac=0.02):
    """Soft tissue: keep every component of the model output that overlaps the guide label, plus any
    that is at least min_frac of the output; stray islands go.  Sheet muscles break at their
    aponeuroses and the rectus at its tendinous intersections, so the largest component alone lost
    the upper half of the external oblique and the rectus abdominis.  Falls back to the largest
    component when nothing overlaps the guide."""
    comp, n = ndi.label(mask)
    if n <= 1:
        return mask
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, comp, index=idx)
    hits = ndi.sum(guide, comp, index=idx)
    keep = [int(i) for i in idx if hits[i - 1] > 0 or sizes[i - 1] >= min_frac * sizes.sum()]
    if not keep:
        keep = [int(np.argmax(sizes)) + 1]
    out = np.isin(comp, keep)
    log(f"    components: {n} in the model output, kept {len(keep)} ({out.sum() / max(mask.sum(), 1):.0%} of it)")
    return out


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


def neighbour_masks(target, vol, sl, args, targets, bone_masks):
    """[(name, bool crop)] of the structures that may touch `target` inside the crop window `sl`.

    bones:   the joint graph (pipeline.neighbours) as guide labels.
    muscles / organs: every other target of the same type whose guide reaches into the crop, plus
             the already segmented structures (saved bone masks; for organs the muscle masks too) that
             reach into it; the largest --max-neighbours (by voxels in the crop) are kept, each getting
             --neg-points negative prompts."""
    out = []
    if args.structures == "bones":
        for nb in pipeline.neighbours(target):
            arr = vol.label_crop(nb, sl)
            if arr is not None and arr.any():
                out.append((nb, arr))
        return out
    lo = vol.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * vol.spacing[::-1]
    hi = vol.origin + np.array([sl[2].stop, sl[1].stop, sl[0].stop]) * vol.spacing[::-1]
    cands = []
    for m in targets:
        if m == target:
            continue
        b = vol.guide_bounds(m)
        if b is None or np.any(b[1] < lo) or np.any(b[0] > hi):
            continue
        arr = vol.label_crop(m, sl)
        if arr is not None and arr.any():
            cands.append((m, arr, int(arr.sum())))
    for name, path, blo, bhi in bone_masks:
        if np.any(bhi < lo) or np.any(blo > hi):
            continue
        arr = vol.mask_crop(path, sl)
        if arr.any():
            cands.append((name, arr, int(arr.sum())))
    cands.sort(key=lambda c: -c[2])
    return [(n, a) for n, a, _ in cands[:args.max_neighbours]]


def list_bone_masks(masks_dir):
    """[(bone, path, lo_lps, hi_lps)] of the saved bone masks (their NIfTI extents)."""
    out = []
    for p in sorted(Path(masks_dir).glob("*.nii.gz")):
        img = sitk.ReadImage(str(p))
        o = np.array(img.GetOrigin()); e = o + (np.array(img.GetSize()) - 1) * np.array(img.GetSpacing())
        out.append((p.name[:-7], p, np.minimum(o, e), np.maximum(o, e)))
    return out


def touched_faces(mask, sl, shape):
    """(axis, side) of every crop face the mask reaches that is not the grid's own boundary."""
    faces = []
    for ax in range(3):
        if sl[ax].start > 0 and np.take(mask, 0, axis=ax).any():
            faces.append((ax, 0))
        if sl[ax].stop < shape[ax] and np.take(mask, -1, axis=ax).any():
            faces.append((ax, 1))
    return faces


def finish_mask(mask, init, spacing, args, log=print):
    """The two guards after the last refinement pass (soft tissue only); returns (mask, leak_clipped, guide_added).
    A first-pass mask over --grow-max-ratio x the guide leaked along a neighbour (two thoracic discs ran up
    the spinal canal at 15 cm3 against a 1 cm3 guide): it is clipped to the guide dilated by --leak-clip-mm.
    For GUIDE_IS_SPACE kinds the guide is the space the structure fills, so the mask is unioned with it."""
    leak_clipped = False
    guide_added = 0
    if args.structures == "bones":
        return mask, leak_clipped, guide_added
    if mask.sum() > args.grow_max_ratio * max(int(init.sum()), 1):
        it = int(np.ceil(args.leak_clip_mm / float(spacing.min())))
        clip = ndi.binary_dilation(init, iterations=it) if it > 0 else init
        before = int(mask.sum())
        mask = ndi.binary_fill_holes(mask & clip)
        leak_clipped = True
        log(f"mask {before * np.prod(spacing) / 1000:.1f} cm^3 is over {args.grow_max_ratio:g}x the guide (leak along a "
            f"neighbour); clipped to the guide + {args.leak_clip_mm:g} mm: {mask.sum() * np.prod(spacing) / 1000:.1f} cm^3")
    if args.structures in GUIDE_IS_SPACE:
        # nothing further than --space-clip-mm from the guide (a disc never leaves the gap between its vertebral
        # bodies; a thin lip of a leak up the spinal canal survived a footprint-only clip), the whole guide inside it
        near = ndi.distance_transform_edt(~init, sampling=spacing) <= args.space_clip_mm
        outside = int((mask & ~near).sum())
        mask &= near
        guide_added = int((init & ~mask).sum())
        mask = ndi.binary_fill_holes(mask | init)
        log(f"guide space: {outside * np.prod(spacing) / 1000:.1f} cm^3 further than {args.space_clip_mm:g} mm from the guide "
            f"dropped, {guide_added * np.prod(spacing) / 1000:.1f} cm^3 of the gap the model left out added")
    return mask, leak_clipped, guide_added


def segment_structure(bone, vol, seg, args, out_dir, neighbours, log=print):
    """Refine one structure; writes <name>.nii.gz (crop) + <name>.json.  Returns the json dict.
    `neighbours(sl)` returns [(name, bool crop)] of the structures that may touch it in window sl."""
    b = vol.guide_bounds(bone)
    if b is None:
        raise RuntimeError(f"no guide label for {bone} in group {vol.group['name']}")
    sl = vol.box_from_bounds(b[0], b[1], args.margin_mm)
    init = vol.label_crop(bone, sl)
    if init is None or not init.any():
        raise RuntimeError(f"empty guide label for {bone} in group {vol.group['name']}")
    t_hu = time.time()
    hu = vol.hu_crop(sl)
    if vol.hu is None:
        log(f"  {bone}: fused {hu.shape} crop from {vol.series_names} in {time.time() - t_hu:.0f} s")
    sep = args.point_sep_mm
    if args.structures != "bones":  # spread the prompts along a long or sheet muscle instead of clustering in its thickest belly
        ext = (np.ptp(np.array(np.where(init)), axis=1) + 1) * vol.spacing
        sep = max(sep, float(ext.max()) / max(args.pos_points, 1))
    pos = interior_points(init, vol.spacing, args.pos_points, sep)

    def negatives(init_, sl_):
        neg_ = []
        near = None
        for nb, nl in neighbours(sl_):
            # the part of the neighbour nearest to this bone matters most: restrict to within 40 mm of it
            if near is None:
                near = ndi.distance_transform_edt(~init_, sampling=vol.spacing) < 40.0
            cand = nl & near
            pts = interior_points(cand if cand.any() else nl, vol.spacing, args.neg_points, args.point_sep_mm)
            neg_ += [(nb, p) for p in pts]
        return neg_

    def clean(mask_, init_):
        mask_ = largest_component(mask_) if args.structures == "bones" else guided_components(mask_, init_, log)
        return ndi.binary_fill_holes(mask_)

    neg = negatives(init, sl)
    log(f"  {bone}: crop {hu.shape}, {len(pos)} positive, {len(neg)} negative prompts")
    t0 = time.time()
    mask = clean(seg.run(hu, init, pos, neg, log=log), init)
    # A guide that stops short (TotalSegmentator's rectus abdominis ends ~10 cm above the pubis) gives
    # a crop the real structure runs out of: the model fills it to the face and the mesh ends in a
    # flat cut.  While the mask reaches a crop face that is not the scan's edge, grow the crop there
    # and refine again from the mask (up to --max-grow times, --grow-mm per step).
    grown = 0
    grow_rejected = False
    while grown < args.max_grow:
        faces = touched_faces(mask, sl, vol.shape)
        if not faces:
            break
        step = np.ceil(args.grow_mm / vol.spacing).astype(int)
        new_sl = list(sl)
        for ax, side in faces:
            s_ = sl[ax]
            new_sl[ax] = slice(max(s_.start - int(step[ax]), 0), s_.stop) if side == 0 else \
                slice(s_.start, min(s_.stop + int(step[ax]), vol.shape[ax]))
        new_sl = tuple(new_sl)
        shift = np.array([o.start - n.start for o, n in zip(sl, new_sl)])
        prev = np.zeros(tuple(n.stop - n.start for n in new_sl), dtype=bool)
        prev[tuple(slice(int(sh), int(sh) + m) for sh, m in zip(shift, mask.shape))] = mask
        sl = new_sl
        hu = vol.hu_crop(sl)
        init = vol.label_crop(bone, sl)
        pos = [(np.array(p) + shift).tolist() for p in pos]
        neg = negatives(init, sl)
        grown += 1
        log(f"  {bone}: mask reaches crop face(s) {faces}; grow {grown}/{args.max_grow}: crop {hu.shape}, "
            f"refining again from the mask ({len(neg)} negative prompts)")
        grown_mask = clean(seg.run(hu, prev, pos, neg, log=log), init | prev)
        if grown_mask.sum() > args.grow_max_ratio * max(int(init.sum()), 1):
            log(f"  {bone}: grow step {grown} rejected: mask {grown_mask.sum() * np.prod(vol.spacing) / 1000:.0f} cm^3 is "
                f"over {args.grow_max_ratio:g}x the guide (leak along a neighbour); keeping the previous mask")
            mask = prev
            grow_rejected = True
            break
        mask = grown_mask
    if grown and touched_faces(mask, sl, vol.shape):
        log(f"  {bone}: still reaches crop face(s) {touched_faces(mask, sl, vol.shape)} after {grown} grow steps")
    mask, leak_clipped, guide_added = finish_mask(mask, init, vol.spacing, args, log=lambda m: log(f"  {bone}: {m}"))
    elapsed = time.time() - t0
    vox_mm3 = float(np.prod(vol.spacing))
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(vol.spacing[::-1].tolist())
    img.SetOrigin((vol.origin + np.array([sl[2].start, sl[1].start, sl[0].start]) * vol.spacing[::-1]).tolist())
    sitk.WriteImage(img, str(out_dir / f"{bone}.nii.gz"), useCompression=True)
    info = {"bone": bone, "structure": SINGULAR[args.structures], "group": vol.group["name"], "series": vol.series_names,
            "planes": vol.planes,
            "spacing_mm_xyz": vol.spacing[::-1].tolist(), "crop_zyx": [[s.start, s.stop] for s in sl],
            "guide_voxels": int(init.sum()), "voxels": int(mask.sum()), "volume_cm3": float(mask.sum() * vox_mm3 / 1000),
            "pos_points": pos, "neg_points": neg, "seconds": round(elapsed, 1), "grow_steps": grown, "grow_rejected": grow_rejected,
            "leak_clipped": leak_clipped, "guide_added_voxels": guide_added,
            "touches_crop_face": [list(f) for f in touched_faces(mask, sl, vol.shape)],
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


def full_mask(arr, sl, box):
    """`arr` (a mask over slices `sl` of the full grid) placed into a zero array over slices `box`."""
    full = np.zeros(tuple(b_.stop - b_.start for b_ in box), dtype=bool)
    inter = tuple(slice(max(b_.start, s.start), min(b_.stop, s.stop)) for b_, s in zip(box, sl))
    if any(i.stop <= i.start for i in inter):
        return full
    full[tuple(slice(i.start - b_.start, i.stop - b_.start) for i, b_ in zip(inter, box))] = \
        arr[tuple(slice(i.start - s.start, i.stop - s.start) for i, s in zip(inter, sl))]
    return full


def resolve_group(bones, vol, masks_dir, log=print, guide_tiebreak=False, priority=()):
    """Full-grid label volume for all `bones` of one group; contested voxels go to the nearer core.

    priority: structures nested inside another (NESTED) keep their contested voxels, first in the list first.

    guide_tiebreak (soft tissue): a contested voxel that exactly one of the two structures' guide
    labels covers goes to that structure first.  The nearer-core rule alone favours an over-grown
    neighbour (the left latissimus dorsi, 1.4x its guide, took 87 cm3 of the external oblique's
    upper digitations); the TotalSegmentator guides of one task are mutually exclusive, so they
    settle most of such overlaps where they are."""
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
    settled = 0
    nested = 0
    for b in priority:
        for i, (bb, arr, sl) in parts.items():
            if bb != b:
                continue
            take = arr & contested[sl]
            if take.any():
                lab = labels[sl]
                lab[take] = i
                labels[sl] = lab
                con = contested[sl]
                con[take] = False
                contested[sl] = con
                nested += int(take.sum())
    if contested.any() and guide_tiebreak:
        owner = np.zeros(vol.shape, dtype=np.int16)  # 0 none, -1 two or more guides, else the one guide
        for i, (b, arr, sl) in parts.items():
            mine = arr & contested[sl]
            if not mine.any():
                continue
            g = vol.label_crop(b, sl)
            if g is None:
                continue
            hit = g & mine
            o = owner[sl]
            o[hit & (o != 0)] = -1
            o[hit & (o == 0)] = i
            owner[sl] = o
        take = contested & (owner > 0)
        labels[take] = owner[take]
        contested &= ~take
        settled = int(take.sum())
        del owner
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
                core = full_mask(arr, sl, box)  # wholly inside its neighbours: its own mask is the core
            d = ndi.distance_transform_edt(~core, sampling=vol.spacing)
            full = full_mask(arr, sl, box)
            win = contested[box] & full & (d < best_d[box])
            bd, bi = best_d[box], best_i[box]
            bd[win] = d[win]
            bi[win] = i
            best_d[box], best_i[box] = bd, bi
        labels[contested] = best_i[contested]
        contested_total = int(contested.sum())
        del best_d, best_i
    log(f"  group {vol.group['name']}: {len(parts)} bones, {nested} contested voxels kept by nested parts, "
        f"{settled} settled by the guide labels, {contested_total} by the nearer core")
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


def carve_masks(labels, vol, bone_masks, log=print):
    """Zero the label grid wherever a saved mask of an earlier structure type is (bones win over
    muscles, bones and muscles win over organs, always)."""
    total = 0
    for name, path, blo, bhi in bone_masks:
        sl = vol.box_from_bounds(blo, bhi, 0.0)
        if any(s.stop <= s.start for s in sl):
            continue
        arr = vol.mask_crop(path, sl)
        hit = arr & (labels[sl] > 0)
        if hit.any():
            labels[sl][hit] = 0
            total += int(hit.sum())
    log(f"  earlier structures carved out of the labels: {total} voxels")


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
    # The Gaussian smoothing can pinch a label apart where it is thin (the rectus abdominis at its
    # tendinous intersections, the external oblique's digitations, a vessel between two prompts), so
    # keep every piece of real size and drop only marching-cubes specks: the earlier largest-piece
    # rule silently threw away the upper two bellies of a rectus whose voxels were all there.
    parts = mesh.split(only_watertight=False)
    n_parts = 1
    if len(parts) > 1:
        big = max(p.area for p in parts)
        keep = [p for p in parts if p.area >= 0.01 * big]
        n_parts = len(keep)
        mesh = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
    if taubin_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=taubin_iters)
    mesh.fix_normals(multibody=n_parts > 1)
    bl = mesh.copy()
    bl.apply_scale(MM_TO_BLENDER)
    ply, stl = out_dir / f"{bone}.ply", out_dir / f"{bone}.stl"
    bl.export(ply)
    bl.export(stl)
    lo, hi = mesh.bounds
    info = {"name": bone, "method": "nnInteractive_v1.0 + joint resolve" + (" on fused planes" if vol.planes == "fused" else ""),
            "group": vol.group["name"], "series": vol.series_names, "planes": vol.planes,
            "spacing_mm_xyz": vol.spacing[::-1].tolist(),
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "components": int(n_parts),
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
    ap.add_argument("--bones", nargs="+", default=["all"], help="structure names (T8, rib_left_3, gluteus_medius_left, ...) or 'all'")
    ap.add_argument("--structures", choices=["bones", "muscles", "organs", "heart", "discs"], default="bones",
                    help="muscles: TotalSegmentator muscle guides (total + abdominal_muscles + headneck_muscles tasks) on ONE "
                         "merged volume over all acquisitions; other muscles and the saved bone masks are the neighbours, "
                         "bones are carved out afterwards; defaults become RUN/nni_fused_muscles, collection muscles, "
                         "run tag ct2021-nni-fused-muscles, --fused-mm 0.8, --measured-max-hu 30 (fat plane = measured). "
                         "organs: the same for the organs and vessels of task total, with the bone AND muscle masks as "
                         "neighbours / carved out; imported into the collections thorax, abdomen, vessels, nervous")
    ap.add_argument("--bone-masks", help="saved bone masks used as neighbours and carved out (default RUN/nni_fused/masks)")
    ap.add_argument("--muscle-masks", help="organs: saved muscle masks used likewise (default RUN/nni_fused_muscles/masks)")
    ap.add_argument("--max-neighbours", type=int, default=12, help="muscles: neighbours kept per structure (largest in the crop)")
    ap.add_argument("--soft-tissue-series", nargs="*", default=SOFT_TISSUE_SERIES, metavar="SERIES",
                    help="muscles: series files of a further acquisition fused in as well (default the 3 mm soft-kernel "
                         "chest+abdomen scan 401/402/403; it wins where its noise is lower); pass nothing to disable")
    ap.add_argument("--out", help="output folder (default RUN/nni_fused, or RUN/nni with --planes axial)")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="mps", help="mps | cpu | cuda")
    ap.add_argument("--atlas-labels", help="fitted atlas labels, used as guide where TotalSegmentator has none")
    ap.add_argument("--planes", choices=["fused", "axial"], default="fused",
                    help="fused: isotropic crops fused from axial+coronal+sagittal (fuse_series.py); axial: native axial only")
    ap.add_argument("--fused-mm", type=float, default=None, help="isotropic voxel size of the fused grid (0.6 bones, 0.8 muscles)")
    ap.add_argument("--margin-mm", type=float, default=25.0, help="crop margin around the guide label")
    ap.add_argument("--grow-mm", type=float, default=40.0, help="crop growth per step where the mask reaches a crop face")
    ap.add_argument("--leak-clip-mm", type=float, default=3.0,
                    help="a first-pass mask over --grow-max-ratio x the guide is clipped to the guide dilated by this (soft tissue)")
    ap.add_argument("--space-clip-mm", type=float, default=1.0,
                    help="GUIDE_IS_SPACE kinds (discs): the mask is kept within this distance of the guide and unioned with it")
    ap.add_argument("--grow-max-ratio", type=float, default=1.8,
                    help="a grow step whose mask exceeds this multiple of the guide volume is rejected (leak guard)")
    ap.add_argument("--max-grow", type=int, default=None,
                    help="grow-and-refine steps when the mask reaches a crop face (default 3 for muscles / organs, 0 for bones)")
    ap.add_argument("--group-margin-mm", type=float, default=5.0,
                    help="a group is used when its ORIGINAL axial volume contains the guide label with this margin; "
                         "0 lets a bone at the very edge of the finer series still use it")
    ap.add_argument("--pos-points", type=int, default=None,
                    help="positive prompts inside the guide (default 2 for bones, 6 for soft tissue, spread along its length)")
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
    ap.add_argument("--measured-max-hu", type=float, default=None,
                    help="interface denser than this = no visible gap = boundary inferred by the model "
                         "(default 450 for bones: joint space; 30 for muscles / organs: a thin fat plane, blurred)")
    ap.add_argument("--blender", action="store_true", help="import the meshes into the running Blender")
    ap.add_argument("--collection", default=None, help="Blender collection (default bones / muscles)")
    ap.add_argument("--run-tag", default=None, help="default ct2021-nni-fused / ct2021-nni-fused-muscles")
    ap.add_argument("--suffix", default="", help="object name suffix in Blender (pilot: keep old bones)")
    ap.add_argument("--replace", action="store_true", help="remove objects of other runs from the collection first")
    ap.add_argument("--quiet", action="store_true",
                    help="console shows one progress line per structure; everything else (model output, prompts) "
                         "goes to --log-file")
    ap.add_argument("--log-file", help="with --quiet: where the detailed output goes (default OUT/qa/segment.log)")
    args = ap.parse_args(argv)
    if not args.quiet:
        return run(args, lambda msg: print(msg, flush=True))
    out = Path(args.out) if args.out else Path(args.run) / (
        f"nni_fused_{args.structures}" if args.structures != "bones" else "nni_fused" if args.planes == "fused" else "nni")
    log_path = Path(args.log_file) if args.log_file else out / "qa" / "segment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with open(log_path, "a") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv or sys.argv[1:])}")
        return run(args, lambda msg: (print(msg, file=console, flush=True), print(msg, flush=True)))


def run(args, say):
    """The pipeline proper; `say` prints a progress line to the console (and the log when quiet)."""
    run = Path(args.run)
    kind = args.structures
    muscles = kind != "bones"  # soft tissue (muscles, organs): merged volume, noise-weighted fusion, saved masks carved out
    if args.fused_mm is None:
        args.fused_mm = 0.6 if kind in FINE else 0.8  # heart: coronaries and cusps are 2-4 mm; discs 5-10 mm thick
    if args.pos_points is None:
        args.pos_points = 6 if muscles else 2
    if args.max_grow is None:
        args.max_grow = 3 if muscles else 0  # the bone guides cover their bones; a short soft-tissue guide must not cut the mask
    if args.measured_max_hu is None:
        args.measured_max_hu = 30.0 if muscles else 450.0  # a 1-2 mm fat plane blurs to +20-40 HU at 0.8 mm
    per_structure_collection = args.collection is None and kind == "organs"
    if args.collection is None:
        args.collection = kind
    if args.run_tag is None:
        args.run_tag = f"ct2021-nni-fused-{kind}" if muscles else "ct2021-nni-fused"
    out = Path(args.out) if args.out else run / (f"nni_fused_{kind}" if muscles else "nni_fused" if args.planes == "fused" else "nni")
    masks_dir, meshes_dir, qa_dir = out / "masks", out / "meshes", out / "qa"
    for d in (masks_dir, meshes_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)
    if muscles:
        bones = STRUCTURES[kind] if args.bones == ["all"] else list(args.bones)
        bone_masks = list_bone_masks(args.bone_masks or run / "nni_fused" / "masks")
        if kind in ("organs", "heart"):
            mdir = Path(args.muscle_masks) if args.muscle_masks else run / "nni_fused_muscles" / "masks"
            bone_masks += list_bone_masks(mdir) if mdir.exists() else []
        if kind == "heart":  # the organ masks around the heart (lungs, oesophagus, cavae...) but not its own envelope
            odir = run / "nni_fused_organs" / "masks"
            bone_masks += [m for m in (list_bone_masks(odir) if odir.exists() else []) if m[0] not in HEART_ENVELOPE]
        print(f"{kind}: {len(bones)} targets; {len(bone_masks)} saved masks of earlier structures as neighbours / carve")
    else:
        bones = pipeline.ALL_BONES if args.bones == ["all"] else [pipeline.canonical(b) for b in args.bones]
        bones = [b for b in bones if b != "costal_cartilages"]
        bone_masks = []
    groups = load_groups(run, merged=muscles, extra_series=[args.soft_tissue_series] if muscles and args.soft_tissue_series else None)
    print("groups:", ", ".join(f"{g['name']} <- " + ("+".join(s.name for s in g["series"]) if args.planes == "fused"
                                                       else Path(g["volume"]).name) for g in groups))
    if args.planes == "fused":
        print(f"fused isotropic grid {args.fused_mm} mm; series:", {g["name"]: [repr(s) for s in g["series"]] for g in groups})

    volumes = {}

    def volume(g):
        if g["name"] not in volumes:
            volumes[g["name"]] = Volume(g, args.atlas_labels, planes=args.planes, fused_mm=args.fused_mm)
            # soft tissue: where acquisitions overlap, the lower-noise one wins; bones: the finer one
            # (heart: contrast-filled chambers and 2-4 mm coronaries, sharpness decides as for bones)
            volumes[g["name"]].quality = "resolution" if kind in FINE else "noise"
        return volumes[g["name"]]

    overrides = {pipeline.canonical(k): v for k, v in (o.split("=", 1) for o in args.group_override)}
    assignment = {}
    for b in bones:
        if b in overrides:
            assignment[b] = overrides[b]
            continue
        if muscles:  # one merged group; keep the structures that have a guide label somewhere
            if volume(groups[0]).guide_bounds(b) is None:
                print(f"!! {b}: no guide label in any acquisition, skipped")
                continue
            assignment[b] = groups[0]["name"]
            continue
        g, _ = pipeline.choose_group(b, groups, margin=args.group_margin_mm, atlas_labels=args.atlas_labels)
        if g is None:
            print(f"!! {b}: no guide label in any group, skipped")
            continue
        assignment[b] = g["name"]

    # 1. segment
    say(f"{args.structures}: {len(assignment)} structures to segment on {args.planes} planes "
        f"({args.fused_mm} mm) -> {out}")
    if not args.skip_segment:
        seg = Segmenter(args.model_dir, args.device)
        items = list(assignment.items())
        for k, (b, gname) in enumerate(items, 1):
            if (masks_dir / f"{b}.nii.gz").exists() and not args.force:
                say(f"[{k}/{len(items)}] {b}: cached")
                continue
            g = next(x for x in groups if x["name"] == gname)
            vol = volume(g)
            try:
                cands = STRUCTURES[kind] if muscles else bones  # a single-structure run still sees every other one as a neighbour
                info = segment_structure(b, vol, seg, args, masks_dir,
                                         lambda sl, _b=b, _v=vol, _c=cands: neighbour_masks(_b, _v, sl, args, _c, bone_masks))
                say(f"[{k}/{len(items)}] {b}: {info['seconds']:.0f} s, {info['volume_cm3']:.1f} cm^3, "
                    f"median {info['hu_median']:.0f} HU, {len(info['neg_points'])} neighbours ({gname})")
            except (RuntimeError, ValueError, MemoryError) as exc:
                say(f"!! [{k}/{len(items)}] {b}: {exc}")

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
        labels, ids = resolve_group(members + extra, vol, masks_dir, guide_tiebreak=muscles, priority=NESTED.get(kind, ()))
        resolved.append((g, vol, labels, ids, members))
    carve_across_groups(resolved)
    if muscles:
        for g, vol, labels, ids, members in resolved:
            carve_masks(labels, vol, bone_masks)
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
                say(f"  meshed {b}: {info['volume_cm3']:.1f} cm^3, {info['faces']} faces"
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
            collection_of = (lambda b: COLLECTION_OF.get(b, args.collection)) if per_structure_collection else (lambda b: args.collection)
            if args.replace:
                for col in sorted({collection_of(b) for b in results}):
                    print(blender_bridge.run_code(pipeline.REMOVE_OTHER_RUNS % {"collection": col, "run_tag": args.run_tag}))
            for b, info in results.items():
                props = {"ct_run": args.run_tag, "ct_structure": SINGULAR[args.structures], "ct_method": info["method"],
                         "ct_group": info["group"], "ct_planes": info["planes"],
                         "ct_series": info["series"], "ct_voxel_mm": [round(x, 3) for x in info["spacing_mm_xyz"]],
                         "ct_inferred_boundaries": [k for k, v in info["contacts"].items() if v["boundary"] == "inferred"]}
                print("  " + str(blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                    "path": str(meshes_dir / f"{b}.ply"), "name": b + args.suffix, "collection": collection_of(b),
                    "decimate": 0.0, "props": props})).strip())
            cols = sorted({collection_of(b) for b in results})
            say(f"Blender: {len(results)} objects imported into collection(s) {', '.join(cols)} (run tag {args.run_tag})")
        except (ConnectionRefusedError, OSError) as exc:
            say(f"!! Blender not reachable on localhost:9876 ({exc}); meshes stay on disk")
    say(f"done: {len(results)} meshes in {meshes_dir}")
    return results


if __name__ == "__main__":
    main()
