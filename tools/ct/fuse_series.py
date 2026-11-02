#!/usr/bin/env python3
"""Fuse the axial, coronal and sagittal series of one CT acquisition into an isotropic volume.

The scanner console built the coronal/sagittal series of the 2021 scan from its thin (0.625 mm)
raw reconstruction, not from the exported 3 mm axial slices (checked empirically: they carry
sub-millimetre detail along the body axis that the axial series does not have).  Each series is
therefore sharp in its two in-plane directions and blurred only across its slice thickness:

    axial    sharp L-R, A-P    blurred S-I (3 mm)
    coronal  sharp L-R, S-I    blurred A-P (3 mm)
    sagittal sharp A-P, S-I    blurred L-R (3 mm)

Fusion is done in the Fourier domain on a common isotropic grid: for every spatial frequency the
series are averaged with weights that fall off with the frequency component along each series'
slice normal, w_i(f) = exp(-2 (f_n t_i)^2) (t_i = slice thickness).  Low frequencies are a plain
average (noise goes down); high frequencies along any axis come from the series that resolve
that axis.  Nothing is sharpened beyond what some series measured; no anatomy is invented.

Used per bone crop by segment_nni.py (whole-volume FFTs would need tens of GB).  Stand-alone:

  python tools/ct/fuse_series.py --run RUN --group Delayed --center-lps X Y Z --size-mm 120 --out crop.nii.gz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import fft as sfft

PAD = 8  # reflect padding (voxels) to soften the crop border before the FFT


class Series:
    """One series: image + which LPS axis its slice normal is + slice thickness (mm)."""

    def __init__(self, path):
        self.path = Path(path)
        self.img = sitk.ReadImage(str(path))
        d = np.array(self.img.GetDirection()).reshape(3, 3)
        assert np.allclose(np.abs(d).sum(axis=0), 1, atol=1e-3), f"{path}: oblique series not supported"
        self.normal_axis = int(np.argmax(np.abs(d[:, 2])))  # LPS axis (0=x L-R, 1=y A-P, 2=z S-I) across the slices
        self.thickness = float(self.img.GetSpacing()[2])
        self.inplane = float(min(self.img.GetSpacing()[:2]))
        self.name = self.path.name.split(".")[0]

    def __repr__(self):
        return f"{self.name}(normal={'xyz'[self.normal_axis]}, {self.thickness} mm)"


def resample(img, origin_lps, spacing_mm, shape_zyx, interpolator=sitk.sitkLinear, default=-5000.0,
             pixel=sitk.sitkFloat32):
    size = [int(shape_zyx[2]), int(shape_zyx[1]), int(shape_zyx[0])]
    out = sitk.Resample(img, size, sitk.Transform(), interpolator, [float(v) for v in origin_lps],
                        [float(spacing_mm)] * 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), default, pixel)
    return sitk.GetArrayFromImage(out)


def fuse_crop(series, origin_lps, spacing_mm, shape_zyx, workers=8, return_parts=False):
    """Fused HU crop (z, y, x) on the isotropic grid given by origin (LPS mm of voxel 0), spacing, shape."""
    # the series (planes) are resampled onto the crop grid in parallel; SimpleITK releases the GIL
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(series), workers)) as ex:
        parts = list(ex.map(lambda s: resample(s.img, origin_lps, spacing_mm, shape_zyx).astype(np.float32), series))
    cover = [p > -4000 for p in parts]
    # where a reformat stack does not reach, use the axial (which defines the group's extent)
    axial = next((p for p, s in zip(parts, series) if s.normal_axis == 2), parts[0])
    for p, c in zip(parts, cover):
        if not c.all():
            p[~c] = axial[~c] if (axial > -4000)[~c].all() else -1000.0
    pads = [(PAD, PAD)] * 3
    F = None
    W = None
    shape_p = tuple(n + 2 * PAD for n in shape_zyx)
    freqs = [sfft.fftfreq(n, d=spacing_mm).astype(np.float32) for n in shape_p[:2]] + \
            [sfft.rfftfreq(shape_p[2], d=spacing_mm).astype(np.float32)]
    fz, fy, fx = np.meshgrid(*freqs, indexing="ij", sparse=True)
    f_lps = {0: fx, 1: fy, 2: fz}
    for p, s in zip(parts, series):
        w = np.exp(-2.0 * (f_lps[s.normal_axis] * s.thickness) ** 2)
        for a in range(3):  # mild preference for the finer in-plane sampling near its Nyquist
            if a != s.normal_axis:
                w = w * np.exp(-0.5 * (f_lps[a] * s.inplane) ** 2)
        w = w.astype(np.float32)
        spec = sfft.rfftn(np.pad(p, pads, mode="reflect"), workers=workers)
        F = spec * w if F is None else F + spec * w
        W = w if W is None else W + w
    fused = sfft.irfftn(F / np.maximum(W, 1e-6), s=shape_p, workers=workers)
    fused = fused[PAD:-PAD, PAD:-PAD, PAD:-PAD].astype(np.float32)
    return (fused, parts) if return_parts else fused


def soft_tissue_noise(hu, lo=0.0, hi=100.0, min_voxels=2000):
    """Noise (HU, robust std of the high-pass residual) inside soft tissue of a crop, or None."""
    from scipy import ndimage as ndi
    smooth = ndi.gaussian_filter(hu, 2.0)  # the mask comes from the smoothed image, so noise cannot break it up
    core = ndi.binary_erosion((smooth > lo) & (smooth < hi), iterations=3)
    if core.sum() < min_voxels:
        return None
    hp = (hu - ndi.gaussian_filter(hu, 3.0))[core]
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def fuse_crop_multi(series_groups, origin_lps, spacing_mm, shape_zyx, blend_mm=20.0, edge_mm=3.0,
                    quality="resolution", workers=8):
    """Fused crop that may span several acquisitions (e.g. the chest and the abdomen scan).

    Each acquisition is fused on its own (fuse_crop) where its axial series covers the crop; where
    acquisitions overlap they are blended with weights that rise linearly with the distance into
    each one's coverage (from edge_mm, where the fill beyond the edge still bleeds in, to
    edge_mm + blend_mm), so the seam is smooth and one acquisition's slice blur never enters
    another's sharp region (a single Fourier pass over all series would do that).

    The weights are scaled by each acquisition's quality so the highest quality source wins where
    both exist and the other one only fills in where the better one has no data:
      quality="resolution"  1 / finest slice thickness  (bones: contrast >> noise, sharpness decides)
      quality="noise"       1 / noise variance measured in the crop's soft tissue (muscles, discs:
                            a sharp-kernel 1 mm series with 50 HU of noise loses to a smooth 3 mm
                            series with 8 HU; falls back to "resolution" where no soft tissue is)"""
    from scipy import ndimage as ndi
    fused, ramps, covers, thick, sigmas = [], [], [], [], []
    for series in series_groups:
        axial = next((s for s in series if s.normal_axis == 2), series[0])
        cov = resample(axial.img, origin_lps, spacing_mm, shape_zyx) > -4000
        if not cov.any():
            continue
        f = fuse_crop(series, origin_lps, spacing_mm, shape_zyx, workers=workers)  # planes in parallel inside
        fused.append(f)
        covers.append(cov)
        thick.append(max(min(s.thickness for s in series), 0.1))
        sigmas.append(soft_tissue_noise(np.where(cov, f, -1000.0)) if quality == "noise" else None)
        ramps.append(np.ones(shape_zyx, dtype=np.float32) if cov.all() else
                     np.clip((ndi.distance_transform_edt(cov, sampling=spacing_mm) - edge_mm) / blend_mm, 0.0, 1.0)
                     .astype(np.float32))
    # one quality scale for all acquisitions: noise variance when it could be measured in every one, else thickness
    if quality == "noise" and fused and all(sg is not None for sg in sigmas):
        qs = [1.0 / max(sg, 1.0) ** 2 for sg in sigmas]
    else:
        qs = [1.0 / t for t in thick]
    weights = [r * q for r, q in zip(ramps, qs)]
    if not fused:
        return np.full(shape_zyx, -1000.0, dtype=np.float32)
    if len(fused) == 1:
        return fused[0]
    W = np.sum(weights, axis=0)
    out = np.sum([f * w for f, w in zip(fused, weights)], axis=0) / np.maximum(W, 1e-9)
    # within edge_mm of every edge no acquisition has weight: take the acquisition that covers there
    hole = W <= 0
    if hole.any():
        for f, cov in zip(fused, covers):
            take = hole & cov
            out[take] = f[take]
            hole &= ~cov
        out[hole] = -1000.0
    return out.astype(np.float32)


def estimate_shift(a, b, spacing_mm, lowpass_mm=4.0):
    """Sub-voxel translation (mm, zyx) that best aligns b to a, from phase correlation on low-pass content."""
    from scipy import ndimage as ndi
    sig = lowpass_mm / spacing_mm
    fa = sfft.fftn(ndi.gaussian_filter(a, sig) - a.mean()); fb = sfft.fftn(ndi.gaussian_filter(b, sig) - b.mean())
    cc = np.real(sfft.ifftn(fa * np.conj(fb) / np.maximum(np.abs(fa * np.conj(fb)), 1e-6)))
    p = np.array(np.unravel_index(int(cc.argmax()), cc.shape), dtype=float)
    # parabolic sub-voxel refinement per axis
    for ax in range(3):
        idx = [int(v) for v in p]
        def at(k):
            j = idx.copy(); j[ax] = (idx[ax] + k) % cc.shape[ax]; return cc[tuple(j)]
        den = at(-1) - 2 * at(0) + at(1)
        if den != 0:
            p[ax] += 0.5 * (at(-1) - at(1)) / den
    p = np.where(p > np.array(cc.shape) / 2, p - np.array(cc.shape), p)
    return p * spacing_mm


def group_series(run, group_name, finest_axial_only=True):
    """Series of a run.json group: finest ORIGINAL axial + every reformat (coronal/sagittal)."""
    manifest = json.loads((Path(run) / "run.json").read_text())
    g = manifest["groups"][group_name]
    by_axis = {}
    for v in g["volumes"]:
        s = Series(v)
        by_axis.setdefault(s.normal_axis, []).append(s)
    out = []
    for axis, ss in sorted(by_axis.items(), key=lambda kv: -kv[0]):  # axial first
        # thinnest first, then the largest stack: the 2021 scan exports the same reformat twice (502 == 602)
        ss.sort(key=lambda s: (s.thickness, -s.img.GetSize()[2]))
        if axis == 2 and not finest_axial_only:
            out += ss
        else:
            out.append(ss[0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--center-lps", nargs=3, type=float, required=True)
    ap.add_argument("--size-mm", type=float, default=120.0)
    ap.add_argument("--spacing-mm", type=float, default=0.6)
    ap.add_argument("--out", required=True, help="fused crop .nii.gz; the per-series resampled crops go next to it")
    ap.add_argument("--check-alignment", action="store_true")
    args = ap.parse_args(argv)
    series = group_series(args.run, args.group)
    print("series:", series)
    n = int(round(args.size_mm / args.spacing_mm))
    origin = np.array(args.center_lps) - args.size_mm / 2
    fused, parts = fuse_crop(series, origin, args.spacing_mm, (n, n, n), return_parts=True)
    for arr, name in [(fused, "fused")] + [(p, s.name) for p, s in zip(parts, series)]:
        img = sitk.GetImageFromArray(arr); img.SetSpacing([args.spacing_mm] * 3); img.SetOrigin(origin.tolist())
        p = Path(args.out) if name == "fused" else Path(args.out).with_name(Path(args.out).name.replace(".nii.gz", f"_{name}.nii.gz"))
        sitk.WriteImage(img, str(p), useCompression=True)
    if args.check_alignment:
        ref = parts[0]
        for p, s in zip(parts[1:], series[1:]):
            print(f"shift of {s.name} relative to {series[0].name} (mm, z y x): {np.round(estimate_shift(ref, p, args.spacing_mm), 2)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
