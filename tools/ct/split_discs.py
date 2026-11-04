#!/usr/bin/env python3
"""Split TotalSegmentator's single `intervertebral_discs` label (task vertebrae_body, licensed) into
one guide label per disc, named after the vertebrae it lies between (disc_T7_T8, disc_L5_S1).

Each connected component of the disc label is assigned to the vertebra whose refined mask centroid
is nearest above it and the one nearest below (superior = +z in LPS), so the naming follows the
already segmented spine (RUN/nni_fused/masks/vertebrae_*.nii.gz) and never the label order.
Components under --min-cm3 are dropped; components between C1 and C2 are not discs and are skipped;
several components with the same pair (a disc cut in two by the label) are merged.

The label itself is far too small for the thin thoracic discs (T4/T5: 0.4 cm3 in a 2.4 cm3 gap), so each
disc guide is then filled out to the gap between its two refined vertebra masks: every voxel column (within
--reach-mm of the label piece's footprint) between the lower edge of the upper vertebra and the upper edge of
the lower one, where that gap is at most --max-gap-mm (the facet joints are narrower and further away).
Columns through a rib head (the costovertebral joint sits between the two bodies' demifacets) are left out.
nnInteractive then gets its positive points over the whole disc space instead of its centre.

  python tools/ct/split_discs.py RUN/totalseg/Delayed/intervertebral_discs.nii.gz RUN/nni_fused/masks RUN/totalseg/Delayed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402

SPINE = list(pipeline.VERTEBRAE) + ([] if "sacrum" in pipeline.VERTEBRAE else ["sacrum"])


def short(v):
    return "S1" if v == "sacrum" else v.replace("vertebrae_", "")


def disc_name(upper, lower):
    return f"disc_{short(upper)}_{short(lower)}"


DISCS = [disc_name(a, b) for a, b in zip(SPINE[1:], SPINE[2:])]  # C2_C3 ... L5_S1 (no disc between C1 and C2)


def fill_gap(piece, img, upper_path, lower_path, max_gap_mm=12.0, reach_mm=8.0, body_path=None,
             min_body_mm=6.0, z_margin_mm=1.5, rib_paths=()):
    """`piece` (bool, on `img`'s grid) grown into the gap between the two vertebra masks; returns the new mask.
    body_path: the task's `vertebrae_body` label (bodies without the arch); only body voxels then bound the
    gap, or the foramen between the pedicles and the canal below a tilted body would be filled as disc too."""
    sp = np.array(img.GetSpacing())  # (x, y, z)
    zs, ys, xs = np.where(piece)
    pad = np.ceil(np.array([30.0, 30.0, 30.0]) / sp[::-1]).astype(int)
    sl = tuple(slice(max(int(a.min()) - p, 0), min(int(a.max()) + p + 1, n))
               for a, p, n in zip((zs, ys, xs), pad, piece.shape))
    crop = sitk.RegionOfInterest(img, [int(s.stop - s.start) for s in sl[::-1]], [int(s.start) for s in sl[::-1]])

    def on_crop(path):
        m = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(sitk.Resample(m, crop, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0

    a, b = on_crop(upper_path), on_crop(lower_path)
    if body_path is not None and Path(body_path).exists():
        # only the vertebral bodies bound the gap: in a kyphotic segment a column through the upper body
        # meets the lower vertebra's arch 10 mm further down, and that stretch of spinal canal is not disc
        body = on_crop(body_path)
        a, b = a & body, b & body
    if not a.any() or not b.any():
        return piece
    hi, lo = (a, b) if np.where(a)[0].mean() > np.where(b)[0].mean() else (b, a)  # +z index = superior (LPS, identity direction)
    nz = a.shape[0]
    zi = np.arange(nz)[:, None, None]
    bottom_of_hi = np.where(hi, zi, nz).min(0)
    top_of_lo = np.where(lo, zi, -1).max(0)
    gap = bottom_of_hi - top_of_lo - 1
    # a column must pass through both bodies properly (min_body_mm of each): one grazing the concave
    # posterior wall meets the lower body only at its mid-height bulge and its "gap" runs down the canal
    thick = (hi.sum(0) * sp[2] >= min_body_mm) & (lo.sum(0) * sp[2] >= min_body_mm)
    cols = thick & (gap > 0) & (gap * sp[2] <= max_gap_mm)
    foot = piece[sl].any(0)
    reach = ndi.binary_dilation(foot, iterations=int(np.ceil(reach_mm / sp[0])))
    cols &= reach
    for rib in rib_paths:  # a rib head between the bodies: that column is the costovertebral joint, not disc
        r = on_crop(rib)
        if not r.any():
            continue
        r = ndi.binary_dilation(r, iterations=int(np.ceil(2.0 / sp[0])))
        in_gap = (r & (zi > top_of_lo[None]) & (zi < bottom_of_hi[None])).any(0)
        cols &= ~in_gap
    if not cols.any():
        return piece
    # the disc is a tilted slab: fit its mid-plane through the ordinary columns and keep the fill within
    # half the typical gap + z_margin_mm of it (a column whose gap runs on down a demifacet or the
    # concave posterior wall still contributes only its slab part)
    g = gap[cols].astype(float)
    typical = float(np.median(g))
    ordinary = cols & (gap <= typical * 1.5 + 1)
    yy, xx = np.where(ordinary)
    mid = (bottom_of_hi[ordinary] + top_of_lo[ordinary]) / 2.0
    A = np.c_[yy * sp[1], xx * sp[0], np.ones_like(yy, dtype=float)]
    coef = np.linalg.lstsq(A, mid * sp[2], rcond=None)[0] if len(yy) >= 3 else np.array([0.0, 0.0, float(mid.mean()) * sp[2]])
    Y, X = np.mgrid[0:cols.shape[0], 0:cols.shape[1]]
    plane = (coef[0] * Y * sp[1] + coef[1] * X * sp[0] + coef[2]) / sp[2]  # in slices
    half = (typical / 2.0) + z_margin_mm / sp[2]
    slab = np.abs(zi - plane[None]) <= half
    fill = cols[None] & (zi > top_of_lo[None]) & (zi < bottom_of_hi[None]) & slab
    out = piece.copy()
    out[sl] |= fill
    return out


def vertebra_centroids(mask_dir):
    """{vertebra: LPS centroid} of the refined vertebra masks (and the sacrum)."""
    out = {}
    for v in SPINE:
        p = Path(mask_dir) / f"{v}.nii.gz"
        if not p.exists():
            continue
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img) > 0
        if not arr.any():
            continue
        idx = np.array(ndi.center_of_mass(arr))[::-1]  # (x, y, z) continuous index
        out[v] = np.array(img.TransformContinuousIndexToPhysicalPoint(idx.tolist()))
    return out


def split(discs_path, mask_dir, out_dir, log=print, min_cm3=0.2, max_gap_mm=12.0, reach_mm=8.0):
    """Write out_dir/disc_X_Y.nii.gz for every disc found in `discs_path`; returns {name: cm3}."""
    img = sitk.ReadImage(str(discs_path))
    arr = sitk.GetArrayFromImage(img) > 0
    vox_cm3 = float(np.prod(img.GetSpacing())) / 1000.0
    cents = vertebra_centroids(mask_dir)
    if len(cents) < 2:
        raise RuntimeError(f"fewer than two vertebra masks in {mask_dir}")
    order = [v for v in SPINE if v in cents]
    lab, n = ndi.label(arr)
    pieces = {}
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == i
        cm3 = comp.sum() * vox_cm3
        if cm3 < min_cm3:
            continue
        local = np.array(ndi.center_of_mass(comp))
        idx = (local + np.array([s.start for s in sl]))[::-1]
        c = np.array(img.TransformContinuousIndexToPhysicalPoint(idx.tolist()))
        above = [v for v in order if cents[v][2] > c[2]]
        below = [v for v in order if cents[v][2] <= c[2]]
        if not above or not below:
            log(f"  disc piece of {cm3:.1f} cm3 beyond the segmented spine's ends, skipped")
            continue
        upper = min(above, key=lambda v: cents[v][2] - c[2])
        lower = min(below, key=lambda v: c[2] - cents[v][2])
        if upper == "vertebrae_C1":
            log(f"  disc piece of {cm3:.1f} cm3 between C1 and C2 (no disc there), skipped")
            continue
        name = disc_name(upper, lower)
        if SPINE.index(lower) != SPINE.index(upper) + 1:
            log(f"  !! {name}: {short(upper)} and {short(lower)} are not consecutive (a vertebra mask is missing?)")
        full = np.zeros_like(arr)
        full[sl] = comp
        pieces[name] = ((pieces[name][0] | full) if name in pieces else full, upper, lower)
    out = {}
    raw = {}
    for name, (m, upper, lower) in pieces.items():
        raw[name] = m.sum() * vox_cm3
        if max_gap_mm > 0:
            ribs = [Path(mask_dir) / f"rib_{side}_{v[len('vertebrae_T'):]}.nii.gz" for v in (upper, lower) if v.startswith("vertebrae_T")
                    for side in ("left", "right")]
            m = fill_gap(m, img, Path(mask_dir) / f"{upper}.nii.gz", Path(mask_dir) / f"{lower}.nii.gz",
                         max_gap_mm=max_gap_mm, reach_mm=reach_mm, body_path=Path(discs_path).parent / "vertebrae_body.nii.gz",
                         rib_paths=[r for r in ribs if r.exists()])
        o = sitk.GetImageFromArray(m.astype(np.uint8))
        o.CopyInformation(img)
        sitk.WriteImage(o, str(Path(out_dir) / f"{name}.nii.gz"), useCompression=True)
        out[name] = m.sum() * vox_cm3
    log(f"  {len(out)} discs from {n} label pieces (label -> filled to the vertebral gap, cm3): "
        + ", ".join(f"{k} {raw[k]:.1f}->{v:.1f}" for k, v in out.items()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("discs", help="intervertebral_discs.nii.gz from task vertebrae_body")
    ap.add_argument("masks", help="folder of the refined vertebra masks (RUN/nni_fused/masks)")
    ap.add_argument("out", help="folder for the per-disc labels (the acquisition's totalseg folder)")
    ap.add_argument("--min-cm3", type=float, default=0.2)
    ap.add_argument("--max-gap-mm", type=float, default=12.0, help="fill the vertebral gap only where it is at most this wide (0: no fill)")
    ap.add_argument("--reach-mm", type=float, default=8.0, help="fill only columns within this distance of the label piece's footprint")
    args = ap.parse_args(argv)
    split(args.discs, args.masks, args.out, min_cm3=args.min_cm3, max_gap_mm=args.max_gap_mm, reach_mm=args.reach_mm)


if __name__ == "__main__":
    main()
