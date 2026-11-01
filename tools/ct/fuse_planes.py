#!/usr/bin/env python
"""Fuse orthogonal CT reconstructions (axial / coronal / sagittal) into one isotropic volume.

Each Philips reconstruction is sharp in-plane (0.45-0.68 mm) but thick through-plane
(2-3 mm).  All inputs are resampled onto a shared isotropic grid; at every voxel each
input is weighted by how well it resolves the local edge direction (an input whose
through-plane axis is parallel to the local gradient is blurred there and gets a low
weight).  The result is sharper than any single reconstruction.

Usage (venv python):
  python tools/ct/fuse_planes.py --inputs AX.nrrd COR.nrrd SAG.nrrd \
      --roi-from-seg "Sacrum L5.seg.nrrd" sacrum --pad 25 --out fused.nrrd
  python tools/ct/fuse_planes.py --inputs ... --roi -80 90 140 280 -310 -150 --out fused.nrrd
Coordinates are LPS millimetres (the NRRD / ITK convention).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def through_plane_axis(img):
    """Unit vector (LPS) of the image axis with the coarsest spacing."""
    spacing = np.array(img.GetSpacing())
    k = int(np.argmax(spacing))
    direction = np.array(img.GetDirection()).reshape(3, 3)  # column j = axis j
    axis = direction[:, k]
    return axis / np.linalg.norm(axis), spacing


def image_bounds(img):
    size = np.array(img.GetSize())
    corners = []
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                corners.append(img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))))
    corners = np.array(corners)
    return corners.min(0), corners.max(0)


def roi_from_seg(path, segment_name, pad):
    """Bounding box (LPS) of one segment of a Slicer .seg.nrrd, grown by `pad` mm."""
    import nrrd

    header = nrrd.read_header(str(path))
    index = None
    for key, value in header.items():
        match = re.match(r"Segment(\d+)_Name$", key)
        if match and value == segment_name:
            index = match.group(1)
    if index is None:
        names = sorted(v for k, v in header.items() if k.endswith("_Name"))
        sys.exit(f"segment {segment_name!r} not found; available: {names}")
    extent = [int(x) for x in header[f"Segment{index}_Extent"].split()]
    dirs = np.array(header["space directions"][-3:], dtype=float)  # rows = axes
    origin = np.array(header["space origin"], dtype=float)
    corners = []
    for i in extent[0:2]:
        for j in extent[2:4]:
            for k in extent[4:6]:
                corners.append(origin + i * dirs[0] + j * dirs[1] + k * dirs[2])
    corners = np.array(corners)
    return corners.min(0) - pad, corners.max(0) + pad


def roi_from_mask(path, pad, threshold=0.5):
    """Bounding box (LPS) of the non-zero voxels of a label image, grown by `pad` mm."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > threshold
    if not arr.any():
        sys.exit(f"mask {path} is empty")
    zs, ys, xs = np.where(arr)
    corners = []
    for i in (xs.min(), xs.max()):
        for j in (ys.min(), ys.max()):
            for k in (zs.min(), zs.max()):
                corners.append(img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))))
    corners = np.array(corners)
    return corners.min(0) - pad, corners.max(0) + pad


def make_grid(lo, hi, spacing):
    """Axis-aligned isotropic grid covering [lo, hi], snapped to the global lattice (integer
    multiples of `spacing` from the scanner origin) so overlapping ROIs share voxels exactly."""
    lo = np.floor(lo / spacing) * spacing
    size = np.ceil((hi - lo) / spacing).astype(int) + 1
    ref = sitk.Image([int(s) for s in size], sitk.sitkFloat32)
    ref.SetSpacing([float(spacing)] * 3)
    ref.SetOrigin([float(v) for v in lo])
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return ref


def select_inputs(inputs, per_plane_finest=True, max_thickness_ratio=2.5, log=print):
    """Keep the thinnest-sliced series of each plane and drop series much coarser than the finest
    overall: averaging in coarse, soft reconstructions blurs the result more than it denoises."""
    meta = []
    for path in inputs:
        img = sitk.ReadImage(str(path))
        normal, spacing = through_plane_axis(img)
        meta.append((path, tuple(np.round(np.abs(normal)).astype(int)), float(spacing.max())))
    finest = min(t for _, _, t in meta)
    keep = []
    for path, axis, thick in meta:
        best_in_plane = min(t for _, a, t in meta if a == axis)
        if per_plane_finest and thick > best_in_plane + 1e-6:
            log(f"  skip {Path(path).name}: {thick:g} mm, a finer series exists for this plane")
        elif thick > max_thickness_ratio * finest + 1e-6:
            log(f"  skip {Path(path).name}: {thick:g} mm slices, more than {max_thickness_ratio:g}x the finest ({finest:g} mm)")
        else:
            keep.append(path)
    return keep


def fuse(inputs, ref, method="weighted", interpolator="bspline", sigma=0.6, power=2.0,
         floor=0.05, thickness_power=1.0, log=print):
    interp = {"bspline": sitk.sitkBSpline, "linear": sitk.sitkLinear}[interpolator]
    arrays, normals, thick = [], [], []
    for path in inputs:
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)
        normal, spacing = through_plane_axis(img)
        log(f"  {Path(path).name}: spacing={np.round(spacing, 3).tolist()} "
            f"through-plane={np.round(normal, 2).tolist()}")
        res = sitk.Resample(img, ref, sitk.Transform(), interp, -1000.0, sitk.sitkFloat32)
        arrays.append(sitk.GetArrayFromImage(res))  # (z, y, x)
        normals.append(normal)
        thick.append(float(spacing.max()))
    stack = np.stack(arrays)  # (n, z, y, x)
    # thinner slices resolve more: weight each input by (thinnest / its thickness) ** p
    res_w = (min(thick) / np.array(thick, dtype=np.float32)) ** thickness_power
    mean = np.tensordot(res_w / res_w.sum(), stack, axes=1)
    if method == "mean":
        fused = mean
    elif method == "max":
        fused = stack.max(0)
    else:
        mean_img = sitk.GetImageFromArray(mean)
        mean_img.CopyInformation(ref)
        smooth = sitk.SmoothingRecursiveGaussian(mean_img, sigma)
        grad = sitk.GetArrayFromImage(sitk.Gradient(smooth, useImageSpacing=True,
                                                    useImageDirection=True))  # (z,y,x,3) x,y,z
        gnorm = np.linalg.norm(grad, axis=-1) + 1e-6
        weights = np.empty_like(stack)
        for i, normal in enumerate(normals):
            cos = np.abs(grad @ normal.astype(np.float32)) / gnorm
            weights[i] = ((1.0 - cos) ** power + floor) * res_w[i]
        fused = (weights * stack).sum(0) / weights.sum(0)
        del weights, grad
    out = sitk.GetImageFromArray(fused.astype(np.float32))
    out.CopyInformation(ref)
    mean_out = sitk.GetImageFromArray(mean.astype(np.float32))
    mean_out.CopyInformation(ref)
    return out, mean_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="NRRD/NIfTI reconstructions to fuse")
    ap.add_argument("--out", required=True, help="output volume (.nrrd or .nii.gz)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="isotropic output spacing in mm (default: finest spacing among inputs)")
    ap.add_argument("--roi", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="LPS bounding box; default: intersection of all inputs")
    ap.add_argument("--roi-from-seg", nargs=2, metavar=("SEG_NRRD", "SEGMENT_NAME"),
                    help="take the ROI from one segment of a Slicer .seg.nrrd")
    ap.add_argument("--roi-from-mask", metavar="LABEL_IMAGE",
                    help="take the ROI from the non-zero voxels of a label image (e.g. TotalSegmentator)")
    ap.add_argument("--pad", type=float, default=20.0, help="mm added around --roi-from-seg / --roi-from-mask")
    ap.add_argument("--method", choices=["weighted", "mean", "max"], default="weighted")
    ap.add_argument("--interpolator", choices=["bspline", "linear"], default="bspline")
    ap.add_argument("--sigma", type=float, default=0.6, help="mm smoothing before the gradient")
    ap.add_argument("--power", type=float, default=2.0, help="sharpness of the direction weighting")
    ap.add_argument("--finest-only", action="store_true",
                    help="keep only the finest series per plane (sharper, noisier); default fuses every input, "
                         "which is smoother")
    ap.add_argument("--max-thickness-ratio", type=float, default=2.5,
                    help="with --finest-only: drop inputs whose slices are thicker than this times the finest")
    ap.add_argument("--thickness-power", type=float, default=1.0,
                    help="weight inputs by (thinnest slice / slice thickness) ** this (0 = ignore thickness)")
    ap.add_argument("--also-mean", action="store_true", help="also write <out>_mean for comparison")
    args = ap.parse_args(argv)

    if args.finest_only:
        args.inputs = select_inputs(args.inputs, True, args.max_thickness_ratio)
    readers = [sitk.ReadImage(str(p)) for p in args.inputs]
    spacing = args.spacing or float(min(min(r.GetSpacing()) for r in readers))
    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    for r in readers:
        blo, bhi = image_bounds(r)
        lo, hi = np.maximum(lo, blo), np.minimum(hi, bhi)
    if args.roi_from_seg:
        slo, shi = roi_from_seg(args.roi_from_seg[0], args.roi_from_seg[1], args.pad)
        lo, hi = np.maximum(lo, slo), np.minimum(hi, shi)
    if args.roi_from_mask:
        mlo, mhi = roi_from_mask(args.roi_from_mask, args.pad)
        lo, hi = np.maximum(lo, mlo), np.minimum(hi, mhi)
    if args.roi:
        r = np.array(args.roi).reshape(3, 2)
        lo, hi = np.maximum(lo, r[:, 0]), np.minimum(hi, r[:, 1])
    if np.any(hi <= lo):
        sys.exit(f"empty ROI: lo={lo} hi={hi}")
    ref = make_grid(lo, hi, spacing)
    print(f"grid: size={ref.GetSize()} spacing={spacing:.4f} mm  lo={np.round(lo, 1).tolist()} hi={np.round(hi, 1).tolist()}")
    print(f"fusing {len(args.inputs)} inputs with method={args.method}")
    fused, mean = fuse(args.inputs, ref, args.method, args.interpolator, args.sigma, args.power,
                       thickness_power=args.thickness_power)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(fused, str(out), useCompression=True)
    print(f"wrote {out}")
    if args.also_mean:
        mean_path = out.with_name(out.stem.replace(".nii", "") + "_mean" + "".join(out.suffixes))
        sitk.WriteImage(mean, str(mean_path), useCompression=True)
        print(f"wrote {mean_path}")
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "inputs": [str(p) for p in args.inputs], "method": args.method, "spacing_mm": spacing,
        "roi_lps_min": lo.tolist(), "roi_lps_max": hi.tolist(), "size": list(ref.GetSize()),
        "interpolator": args.interpolator, "sigma": args.sigma, "power": args.power,
        "thickness_power": args.thickness_power}, indent=1))


if __name__ == "__main__":
    main()
