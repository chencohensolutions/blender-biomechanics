#!/usr/bin/env python
"""Extract one bone from a fused CT volume as a high-detail mesh in Blender coordinates.

Pipeline
  1. bone voxels = fused HU > --level (default 250, a cortical-bone iso-value)
  2. guide labels (TotalSegmentator NIfTI masks, 1.5 mm) are resampled onto the fused grid;
     bone clearly nearer to one label (and close to it) seeds that label (a vertebra label is
     distrusted within --rib-band-mm of a rib label, since the coarse labels put rib heads on
     the vertebra); the dense cortex (> --split-level) is split first by a watershed on the
     distance transform so thin joints bridged only by partial volume fall apart, then the
     softer shell floods from the cores along the brightest paths so the fronts meet in the
     joint space.  A rib prior extends each rib label along the rib towards the spine so rib
     heads are not left on the vertebra.  --split hu / cascade are alternatives
  3. the target bone = its watershed basin, largest component, plus the spongy bone
     (> 150 HU) inside its silhouette; a 2 mm closing seals the cortical pores so the
     interior fills to a solid body; large low-density pockets the closing bridged
     (canal and foramen contents) are carved back out
  4. the surface is the sub-voxel iso-surface of the *intensity* at --level, restricted to
     the target bone (interior forced solid, neighbours masked out), so the cortex sits
     where the scan says it is: joint gaps are the scan's gaps
  5. optional Taubin smoothing (no shrinkage), written as PLY + STL in metres with the
     scanner's LPS axes (Blender: +X = patient left, +Y = posterior, +Z = superior)

  python tools/ct/extract_bone.py --fused sacrum_fused.nrrd --target sacrum \
      --labels totalseg_601/sacrum.nii.gz totalseg_601/vertebrae_L5.nii.gz \
               totalseg_601/hip_left.nii.gz totalseg_601/hip_right.nii.gz \
      --out-dir meshes/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure, segmentation

MM_TO_BLENDER = 0.001


def resample_label(label_path, ref):
    img = sitk.ReadImage(str(label_path))
    res = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(res) > 0


def ball(radius_vox):
    r = int(np.ceil(radius_vox))
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= radius_vox * radius_vox


def dilate_mm(mask, radius_mm, spacing):
    if radius_mm <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= radius_mm


def erode_mm(mask, radius_mm, spacing):
    if radius_mm <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(mask, sampling=spacing) > radius_mm


def crop_slices(mask, pad_vox):
    zs, ys, xs = np.where(mask)
    return tuple(slice(max(int(a.min()) - pad_vox, 0), int(a.max()) + pad_vox + 1) for a in (zs, ys, xs))


def majority_by_component(comp, values, n_comp):
    """For every component id (1..n_comp) the most frequent non-zero value inside it, else 0."""
    sel = values > 0
    if not sel.any():
        return np.zeros(n_comp + 1, dtype=np.int32)
    vmax = int(values.max()) + 1
    key = comp[sel].astype(np.int64) * vmax + values[sel]
    uniq, cnt = np.unique(key, return_counts=True)
    order = np.argsort(-cnt)
    out = np.zeros(n_comp + 1, dtype=np.int32)
    for k in uniq[order]:
        c, v = int(k // vmax), int(k % vmax)
        if out[c] == 0:
            out[c] = v
    return out


def cascade_split(hu, work, markers, levels, fallback=None, log=print):
    """Brightness cascade: the densest cortex forms separate cores first (seeded by the guide
    markers), each dimmer layer joins the core it touches, and a layer that merges two cores is
    cut by a distance-transform watershed seeded by those cores, i.e. at the thinnest neck of
    that layer.  Thin joints that only bridge at low density are therefore cut where they
    bridge, not where the labels guess."""
    basins = np.zeros(hu.shape, dtype=np.int32)
    for lvl in sorted(set(levels), reverse=True):
        mask = work & (hu > lvl)
        comp, n = ndi.label(mask)
        if n == 0:
            continue
        # distinct basins present in each component
        inside = mask & (basins > 0)
        pairs = np.unique(np.stack([comp[inside], basins[inside]]), axis=1) if inside.any() else np.zeros((2, 0), int)
        n_basins = np.bincount(pairs[0], minlength=n + 1)
        comp_basin = np.zeros(n + 1, dtype=np.int32)
        single = n_basins == 1
        comp_basin[pairs[0][single[pairs[0]]]] = pairs[1][single[pairs[0]]]
        # components without a basin yet: seed from the majority guide marker inside them
        seeds = majority_by_component(comp, np.where(mask & (comp_basin[comp] == 0), markers, 0), n)
        comp_basin = np.where(comp_basin > 0, comp_basin, seeds)
        if fallback is not None:
            # still unseeded pieces take the nearest label (except where the labels are distrusted),
            # so a neighbour's unlabelled core is not swallowed when the layers merge
            seeds = majority_by_component(comp, np.where(mask & (comp_basin[comp] == 0), fallback, 0), n)
            comp_basin = np.where(comp_basin > 0, comp_basin, seeds)
        multi = np.flatnonzero(n_basins > 1)
        assigned = comp_basin[comp]
        basins = np.where(mask & (assigned > 0) & ~np.isin(comp, multi), assigned, basins)
        if multi.size:
            mm = np.isin(comp, multi)
            ws = segmentation.watershed(-ndi.distance_transform_edt(mm), markers=np.where(mm, basins, 0), mask=mm)
            basins = np.where(mm, ws, basins)
        log(f"  level {lvl:4.0f} HU: {n:5d} pieces, {multi.size:3d} merged and cut, "
            f"{int((mask & (basins > 0)).sum()):8d} voxels assigned")
    return basins


def extend_rib_markers(labels, names, markers, work, spacing, extend_mm, radius_mm, core_erode_mm,
                       dense=None, log=print):
    """Rib prior: the coarse rib label stops ~25 mm short of the rib head.  From the label's medial
    end, a cylinder of `radius_mm` is extended `extend_mm` along the rib's local direction towards
    the spine; bone inside it that connects to the rib label and is not deep inside a vertebra
    (vertebra labels eroded by `core_erode_mm`) is marked as that rib."""
    rib_ids = [i for i, k in enumerate(names, start=1) if k.startswith("rib_")]
    if not rib_ids:
        return markers
    non_rib = np.zeros(markers.shape, bool)
    for i, k in enumerate(names, start=1):
        if not k.startswith("rib_"):
            non_rib |= labels[k]
    if not non_rib.any():
        return markers
    to_vert = ndi.distance_transform_edt(~non_rib, sampling=spacing)
    core = erode_mm(non_rib, core_erode_mm, spacing)
    zz, yy, xx = np.indices(markers.shape, dtype=np.float32)
    coords = np.stack([xx, yy, zz], axis=-1) * spacing  # mm, (z,y,x,3) in x,y,z order
    del xx, yy, zz
    for i in rib_ids:
        lab = labels[names[i - 1]]
        if lab.sum() < 200:
            continue
        pts = np.argwhere(lab)[:, ::-1].astype(np.float32) * spacing  # (n,3) x,y,z mm
        d_vert = to_vert[lab]
        end = pts[np.argmin(d_vert)]
        near = pts[np.linalg.norm(pts - end, axis=1) < 25.0]
        if len(near) < 50:
            continue
        axis = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0]
        vert_centroid = np.argwhere(non_rib)[:, ::-1].mean(0) * spacing
        if np.dot(axis, vert_centroid - end) < 0:
            axis = -axis
        rel = coords - end
        along = rel @ axis
        radial = np.linalg.norm(rel - along[..., None] * axis, axis=-1)
        cyl = (along >= -2.0) & (along <= extend_mm) & (radial <= radius_mm)
        # only dense cortex connected to the rib label is claimed: at that density the rib head is
        # separated from the vertebral body by the joint gap, so the claim cannot cross into the body
        base = work if dense is None else dense
        cand = (cyl & base & ~core) | lab
        comp, n = ndi.label(cand)
        keep = np.unique(comp[lab])
        keep = keep[keep > 0]
        grown = np.isin(comp, keep) & cyl & base & ~core & ~lab
        markers[grown] = i
        log(f"  rib prior {names[i - 1]}: medial end {np.round(end, 1).tolist()} mm, axis {np.round(axis, 2).tolist()}, "
            f"+{int(grown.sum())} voxels claimed")
    return markers


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask, n
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fused", required=True, help="fused isotropic CT volume (fuse_planes.py)")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="guide label masks (NIfTI/NRRD); the file stem is the label name")
    ap.add_argument("--target", required=True, nargs="+",
                    help="stem(s) of the label(s) to extract, e.g. sacrum; several targets are cut in ONE "
                         "shared partition so touching bones stay consistent (joint re-extraction)")
    ap.add_argument("--atlas-skip", nargs="*",
                    default=["femur_left", "femur_right", "humerus_left", "humerus_right", "vertebrae_C6"],
                    help="bones whose atlas fit is unreliable (partly scanned): keep the coarse label for them")
    ap.add_argument("--atlas-labels", default=None,
                    help="directory of fitted-atlas label volumes <bone>.nii.gz (register_atlas.py); where "
                         "present they decide bone ownership instead of the coarse labels")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None, help="output / Blender object name (default: target)")
    ap.add_argument("--level", type=float, default=250.0, help="HU iso-value for the bone surface")
    ap.add_argument("--denoise-mm", type=float, default=0.6,
                    help="Gaussian sigma applied to the fused volume before thresholding (0 = off)")
    ap.add_argument("--band-mm", type=float, default=4.0,
                    help="bone within this distance of a guide-label boundary is split by the watershed; "
                         "the rest follows the nearest guide label")
    ap.add_argument("--rib-band-mm", type=float, default=20.0,
                    help="band used where a non-rib label is nearest and a rib label is runner-up "
                         "(the coarse labels put rib heads on the vertebra)")
    ap.add_argument("--rib-extend-mm", type=float, default=30.0,
                    help="rib prior: extend each rib label this far along the rib towards the spine (0 = off)")
    ap.add_argument("--rib-radius-mm", type=float, default=6.0, help="radius of the rib-prior cylinder")
    ap.add_argument("--rib-core-erode-mm", type=float, default=6.0,
                    help="bone deeper than this inside a vertebra label is never claimed by the rib prior")
    ap.add_argument("--label-reach-mm", type=float, default=4.0,
                    help="a guide label is trusted only within this distance of itself")
    ap.add_argument("--split", choices=["distance", "hu", "cascade"], default="distance",
                    help="cascade: brightness levels from dense cortex down, merged pieces cut at their "
                         "thinnest neck; hu: watershed on inverted intensity; distance: watershed on the "
                         "inverted distance transform")
    ap.add_argument("--levels", nargs="+", type=float, default=[900.0, 750.0, 600.0, 480.0, 380.0, 310.0],
                    help="HU levels of the cascade (descending; --level is appended)")
    ap.add_argument("--split-sigma-mm", type=float, default=0.5, help="smoothing of the intensity before the split")
    ap.add_argument("--split-level", type=float, default=500.0,
                    help="if above --level: split the bones on the cortex above this HU first, then grow the "
                         "softer shell onto the cores along the brightest paths (opens thin joints bridged by "
                         "partial volume)")
    ap.add_argument("--guide-margin-mm", type=float, default=8.0,
                    help="bone voxels farther than this from the (closed) target guide label are ignored")
    ap.add_argument("--guide-close-mm", type=float, default=5.0,
                    help="closing + hole fill applied to the target guide label before the ROI (0 = off)")
    ap.add_argument("--carve-level", type=float, default=60.0,
                    help="HU below which filled pockets count as canal/foramen contents, not marrow")
    ap.add_argument("--carve-min-mm3", type=float, default=50.0,
                    help="smallest low-density pocket that is carved back out of the solid")
    ap.add_argument("--close-mm", type=float, default=0.0,
                    help="optional binary closing of the bone mask (seals thin cortex gaps; 0 = off)")
    ap.add_argument("--fill-level", type=float, default=150.0,
                    help="HU above which voxels next to the bone count as spongy bone for the solid fill")
    ap.add_argument("--fill-guide-margin-mm", type=float, default=2.0,
                    help="the solid fill stays within this distance of the bone's own closed guide label")
    ap.add_argument("--core-level", type=float, default=100.0,
                    help="HU above which voxels deep inside the silhouette count as marrow (fatty marrow is dim)")
    ap.add_argument("--envelope-mm", type=float, default=8.0,
                    help="closing radius defining the bone silhouette inside which spongy bone is filled")
    ap.add_argument("--solid-close-mm", type=float, default=2.0,
                    help="closing radius that seals cortical pores so the interior fills (0 = keep the shell)")
    ap.add_argument("--smooth-iters", type=int, default=10, help="Taubin smoothing iterations (0 = off)")
    ap.add_argument("--also-labelmap", action="store_true", help="write the per-bone label volume (NRRD)")
    args = ap.parse_args(argv)

    args.targets = list(args.target)
    args.target = args.targets[0]  # primary target for messages and the ROI
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_img = sitk.ReadImage(args.fused, sitk.sitkFloat32)
    spacing = float(fused_img.GetSpacing()[0])
    origin = np.array(fused_img.GetOrigin())
    direction = np.array(fused_img.GetDirection()).reshape(3, 3)
    assert np.allclose(direction, np.eye(3)), "fused volume must be axis-aligned (fuse_planes.py output)"
    hu = sitk.GetArrayFromImage(fused_img)  # (z, y, x)
    print(f"fused: shape={hu.shape} spacing={spacing:.4f} mm")
    if args.denoise_mm > 0:
        # sharp reconstruction kernels leave speckle that becomes spikes on the iso-surface; a
        # Gaussian narrower than the slice thickness removes it without moving a symmetric edge
        hu = ndi.gaussian_filter(hu, args.denoise_mm / spacing)

    labels = {Path(p).name.split(".")[0]: resample_label(p, fused_img) for p in args.labels}
    for t in args.targets:
        if t not in labels:
            raise SystemExit(f"target {t!r} not among labels {sorted(labels)}")
    coarse = dict(labels)  # the positionally accurate but anatomically coarse guides (TotalSegmentator)
    if args.atlas_labels:
        # anatomical prior: the fitted atlas bone says which bone a region belongs to (rib heads
        # on the rib, articular processes on their own vertebra); where an atlas label exists it
        # replaces the coarse label for the ownership decision, the coarse label keeps the ROI
        atlas_dir = Path(args.atlas_labels)
        swapped = []
        skip = set(args.atlas_skip or [])
        atlas = {}
        for key in list(labels):
            cand = atlas_dir / f"{key}.nii.gz"
            if key in skip or not cand.exists():
                continue
            atl = resample_label(cand, fused_img)
            if atl.any():
                atlas[key] = atl
                swapped.append(key)
        # a bone owns its atlas shape plus whatever its coarse label covers that no other bone's
        # atlas claims (the coccyx stays with the sacrum, the rib head leaves the vertebra)
        for key, atl in atlas.items():
            claimed_by_others = np.zeros(atl.shape, bool)
            for other, oatl in atlas.items():
                if other != key:
                    claimed_by_others |= oatl
            labels[key] = atl | (coarse[key] & ~claimed_by_others)
        print(f"  atlas labels used for: {swapped}")
    for key, mask in labels.items():
        print(f"  guide {key}: {int(mask.sum())} voxels on fused grid")
    for t in args.targets:
        if not labels[t].any():
            raise SystemExit(f"guide label {t!r} is empty on the fused grid")
    targets_union = np.zeros(hu.shape, bool)
    for t in args.targets:
        targets_union |= labels[t]

    # crop everything to the targets' neighbourhood first (the fused grid is much larger)
    crop = crop_slices(targets_union, int(np.ceil((args.guide_margin_mm + args.guide_close_mm) / spacing)) + 3)
    crop_offset_vox = np.array([c.start for c in crop], dtype=float)
    hu = hu[crop]
    labels = {k: v[crop] for k, v in labels.items()}
    coarse = {k: v[crop] for k, v in coarse.items()}
    print(f"cropped to {hu.shape} around {args.target}")

    bone = hu > args.level
    if args.close_mm > 0:
        bone = ndi.binary_closing(bone, structure=ball(args.close_mm / spacing))

    # restrict the problem to the neighbourhood of the target; the coarse guide label can have
    # holes (TotalSegmentator left the S1 body out of the sacrum), so close and fill it first
    guide = np.zeros(hu.shape, bool)
    for t in args.targets:
        guide |= labels[t] | coarse[t]
    if args.guide_close_mm > 0:
        guide = ndi.binary_fill_holes(erode_mm(dilate_mm(guide, args.guide_close_mm, spacing), args.guide_close_mm, spacing))
    roi = dilate_mm(guide, args.guide_margin_mm, spacing)
    work = bone & roi
    print(f"bone voxels in ROI: {int(work.sum())}")
    if not work.any():
        # e.g. a costal cartilage that is not calcified: the prior exists but nothing exceeds --level
        raise SystemExit(f"no voxels above {args.level:.0f} HU inside the ROI of {args.target}; nothing to extract")

    # Markers: every bone voxel whose nearest guide label beats the runner-up by more than
    # --band-mm is assigned to that label outright (the coarse labels are reliable away from
    # their boundaries); only the band around label boundaries is left to the watershed on the
    # distance transform, which splits touching bones at the real geometric gap.
    names = sorted(labels)
    dists = np.stack([ndi.distance_transform_edt(~labels[k], sampling=spacing) for k in names])
    order = np.argsort(dists, axis=0)
    nearest = order[0]
    first = np.take_along_axis(dists, nearest[None], 0)[0]
    second = np.take_along_axis(dists, order[1][None], 0)[0] if len(names) > 1 else np.full(hu.shape, np.inf)
    # a label is trusted only close to itself (--label-reach-mm): bone the coarse labels missed
    # (e.g. a rib segment) is left to the watershed instead of defaulting to the nearest label
    # the band is wider between a rib and anything else: the coarse labels put the rib head,
    # neck and tubercle on the vertebra, so there the joint must be found by the watershed
    is_rib = np.array([k.startswith("rib_") for k in names])
    pair_band = np.full((len(names), len(names)), args.band_mm, dtype=np.float32)
    # row = nearest label, column = runner-up: a non-rib label is distrusted near a rib label,
    # the rib label itself is trusted (it is right where it exists, it just stops too early)
    pair_band[np.ix_(~is_rib, is_rib)] = args.rib_band_mm
    runner = order[1] if len(names) > 1 else nearest
    band = pair_band[nearest, runner]
    confident = work & (second - first > band) & (first <= args.label_reach_mm)
    markers = np.where(confident, nearest + 1, 0).astype(np.int32)
    distrusted = work & (band > args.band_mm) & (second - first <= band)  # non-rib label next to a rib
    fallback = np.where(work & ~distrusted, nearest + 1, 0).astype(np.int32)
    del band, runner
    if args.rib_extend_mm > 0:
        dense = work & (hu > args.split_level) if args.split_level > args.level else None
        markers = extend_rib_markers(labels, names, markers, work, spacing, args.rib_extend_mm,
                                     args.rib_radius_mm, args.rib_core_erode_mm, dense)
    for i, key in enumerate(names, start=1):
        print(f"  marker {i} {key}: {int((markers == i).sum())} voxels")
    print(f"  undecided bone voxels: {int((work & ~confident).sum())}")
    del dists, order, first, second
    if args.split == "distance":
        elevation = -ndi.distance_transform_edt(work)  # fronts meet at the thinnest neck
    else:
        # fronts meet on the lowest-density path between the markers, i.e. in the joint space
        # (cartilage / disc), not at a thin neck of solid bone such as a rib neck
        elevation = -ndi.gaussian_filter(hu, args.split_sigma_mm / spacing)
    if args.split == "cascade":
        levels = [l for l in args.levels if l > args.level] + [args.level]
        basins = cascade_split(hu, work, markers, levels, fallback)
    elif args.split_level > args.level:
        # split the dense cortex first: at this level the partial-volume bridge across a thin
        # joint (rib head in its socket) is gone, so touching bones fall apart; then the softer
        # shell between --level and --split-level joins whichever core it is nearest to
        core = work & (hu > args.split_level)
        core_basins = segmentation.watershed(elevation, markers=markers, mask=core)
        n_lost = int((core & (core_basins == 0)).sum())
        # the softer shell floods from the cores through bright bone first, so the fronts meet on
        # the darkest path (the joint), not midway through a thin cortex that fell below the level
        basins = segmentation.watershed(-ndi.gaussian_filter(hu, args.split_sigma_mm / spacing),
                                        markers=core_basins, mask=work)
        print(f"  split at {args.split_level:.0f} HU: {int(core.sum())} core voxels, {n_lost} unreachable")
    else:
        basins = segmentation.watershed(elevation, markers=markers, mask=work)
    guide_closed = guide  # the ROI guide, shared by all targets
    for tname in args.targets:
        name = args.name if (args.name and len(args.targets) == 1) else tname
        guide = labels[tname] | coarse[tname]
        if args.guide_close_mm > 0:
            guide = ndi.binary_fill_holes(erode_mm(dilate_mm(guide, args.guide_close_mm, spacing), args.guide_close_mm, spacing))
        target_id = names.index(tname) + 1
        target = basins == target_id
        target, n_components = largest_component(target)
        others = (basins > 0) & (basins != target_id)
        # cortical bone above the iso-level is a porous shell: spongy bone inside connects to the
        # outside through sub-voxel pores, so plain hole filling leaves the bone hollow.  A small
        # closing seals those pores (foramina and the canal are far wider), then the interior fills.
        shell = target.copy()
        if args.solid_close_mm > 0:
            shell_voxels = int(target.sum())
            # spongy bone (above --fill-level) inside the bone's silhouette and nearer to this bone
            # than to any neighbour.  The silhouette is a wide closing of the cortical shell, eroded
            # so that enhanced vessels lying on the surface stay outside.
            near_target = ndi.distance_transform_edt(~target, sampling=spacing)
            near_others = ndi.distance_transform_edt(~others, sampling=spacing) if others.any() else np.inf
            envelope = erode_mm(dilate_mm(target, args.envelope_mm, spacing), args.envelope_mm + 1.5, spacing)
            band = near_target <= 2 * args.solid_close_mm  # thin parts: close to the cortex
            core = ((hu > args.core_level) & envelope) | ((hu > args.fill_level) & band)
            core &= near_target < near_others
            core &= ~others
            # never fill outside this bone's own (closed) guide label + margin, nor inside a
            # neighbour's label: the silhouette can bridge a disc or joint space otherwise
            others_guide = np.zeros(hu.shape, bool)
            for key in names:
                if key != tname:
                    others_guide |= labels[key]
            core &= dilate_mm(guide, args.fill_guide_margin_mm, spacing) & ~others_guide
            solid = target | core
            solid = erode_mm(dilate_mm(solid, args.solid_close_mm, spacing), args.solid_close_mm, spacing) & ~others
            solid &= ~(others_guide & ~shell)  # the closing may not reach into a neighbour's label either
            solid = ndi.binary_fill_holes(solid)
            target, _ = largest_component(solid)
            print(f"solid fill: {shell_voxels} shell voxels -> {int(target.sum())} solid voxels")
            # pockets the closing bridged that are not bone at all (spinal canal, foramen contents)
            low = target & ~shell & (hu < args.carve_level)
            low_lab, n_low = ndi.label(low)
            if n_low:
                sizes = ndi.sum(low, low_lab, range(1, n_low + 1)) * spacing ** 3
                big = np.flatnonzero(sizes > args.carve_min_mm3) + 1
                if big.size:
                    carve = np.isin(low_lab, big)
                    target &= ~carve
                    target, _ = largest_component(target)
                    print(f"carved {big.size} low-density pocket(s), {carve.sum() * spacing**3 / 1000:.1f} cm^3 "
                          f"(< {args.carve_level:.0f} HU, > {args.carve_min_mm3:.0f} mm^3)")
        else:
            target = ndi.binary_fill_holes(target)
        print(f"target {tname}: {int(target.sum())} voxels "
              f"({target.sum() * spacing**3 / 1000:.1f} cm^3), {n_components} component(s) before cleanup")

        # sub-voxel iso-surface of the intensity, restricted to the target bone
        interior = ndi.binary_erosion(target, structure=ball(1.0))
        allowed = ndi.binary_dilation(target, structure=ball(1.0))
        vol = hu.copy()
        vol[interior] = np.maximum(vol[interior], args.level + 100.0)
        vol[~allowed | others] = -1000.0
        zs, ys, xs = np.where(allowed)
        pad = 2
        sl = (slice(max(zs.min() - pad, 0), zs.max() + pad + 1), slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
              slice(max(xs.min() - pad, 0), xs.max() + pad + 1))
        sub = np.pad(vol[sl], 1, constant_values=-1000.0)  # air border: the surface always closes
        verts, faces, _, _ = measure.marching_cubes(sub, level=args.level, spacing=(spacing,) * 3,
                                                    allow_degenerate=False)
        # skimage vertices are (z, y, x) in mm relative to the padded sub-volume corner
        offset_vox = np.array([sl[0].start, sl[1].start, sl[2].start], dtype=float) - 1.0 + crop_offset_vox
        verts_zyx_mm = verts + offset_vox * spacing
        verts_lps = origin + verts_zyx_mm[:, ::-1]  # (x, y, z) LPS mm

        import trimesh
        mesh = trimesh.Trimesh(verts_lps, faces, process=True)
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda m: m.area)
            print(f"kept largest of {len(parts)} surface shells")
        if args.smooth_iters > 0:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=args.smooth_iters)
        mesh.fix_normals()
        bl = mesh.copy()
        bl.apply_scale(MM_TO_BLENDER)

        ply = out_dir / f"{name}.ply"
        stl = out_dir / f"{name}.stl"
        bl.export(ply)
        bl.export(stl)
        lo, hi = mesh.bounds
        info = {"name": name, "target": tname, "level_hu": args.level, "fused": args.fused,
                "labels": args.labels, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                "volume_cm3": float(target.sum() * spacing**3 / 1000), "bbox_lps_mm_min": lo.tolist(),
                "bbox_lps_mm_max": hi.tolist(), "size_mm": (hi - lo).tolist(), "watertight": bool(mesh.is_watertight),
                "coords": "Blender = LPS mm * 0.001 (x=left, y=posterior, z=superior), scanner origin kept",
                "basins_file": f"{name}_basins.nrrd" if args.also_labelmap else None,
                "basin_ids": {str(i): n for i, n in enumerate(names, start=1)}, "target_id": target_id}
        (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1))
        print(f"wrote {ply} and {stl}: {info['vertices']} verts, {info['faces']} faces, "
              f"size mm {np.round(hi - lo, 1).tolist()}, watertight={info['watertight']}")

        if args.also_labelmap:
            basins[target] = target_id  # the final solid bone replaces its porous basin
            lab_img = sitk.GetImageFromArray(basins.astype(np.uint8))
            lab_img.SetSpacing(fused_img.GetSpacing())
            lab_img.SetDirection(fused_img.GetDirection())
            lab_img.SetOrigin((origin + crop_offset_vox[::-1] * spacing).tolist())
            lab_path = out_dir / f"{name}_basins.nrrd"
            sitk.WriteImage(lab_img, str(lab_path), useCompression=True)
            print(f"wrote {lab_path} (ids: {dict(enumerate(names, start=1))})")


if __name__ == "__main__":
    main()
