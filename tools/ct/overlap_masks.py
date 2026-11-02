"""Bone-to-bone overlap of the fused run, measured on the model's masks (before the joint resolution
that makes the labels exclusive): for every pair whose mask extents meet, the shared voxels, their
volume and how deep inside both bones the deepest shared voxel lies.  Resolved meshes cannot overlap
by more than the 0.5 mm smoothing, so this is where the bones really competed."""
import json, sys, time
import numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from pathlib import Path
RUN = Path(sys.argv[1]); OUT = Path(sys.argv[2])
masks = sorted((RUN / "nni_fused" / "masks").glob("*.nii.gz"))
info = {}
for p in masks:
    img = sitk.ReadImage(str(p)); o = np.array(img.GetOrigin()); sp = np.array(img.GetSpacing())
    e = o + (np.array(img.GetSize()) - 1) * sp
    info[p.name[:-7]] = (img, np.minimum(o, e), np.maximum(o, e), sp)
names = sorted(info)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
         if np.all(info[a][1] <= info[b][2] + 1) and np.all(info[b][1] <= info[a][2] + 1)]
print(f"{len(names)} masks, {len(pairs)} pairs with meeting extents", flush=True)
rows = []
t0 = time.time()
for k, (a, b) in enumerate(pairs, 1):
    ia, ib = info[a][0], info[b][0]
    rb = sitk.Resample(ib, ia, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ib.GetPixelID())
    A = sitk.GetArrayFromImage(ia) > 0; B = sitk.GetArrayFromImage(rb) > 0
    I = A & B
    n = int(I.sum()); sp = info[a][3][::-1]
    row = {"a": a, "b": b, "shared_voxels": n, "shared_mm3": float(n * np.prod(sp)), "max_depth_mm": 0.0,
           "frac_of_smaller": float(n / max(min(A.sum(), B.sum()), 1))}
    if n:
        sl = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in ndi.find_objects(I.astype(np.uint8))[0])
        da = ndi.distance_transform_edt(A[sl], sampling=sp); db = ndi.distance_transform_edt(B[sl], sampling=sp)
        row["max_depth_mm"] = float(np.minimum(da, db)[I[sl]].max())
    rows.append(row)
    if n:
        print(f"[{k}/{len(pairs)}] {a:18s} {b:18s} shared {row['shared_mm3']:8.0f} mm3  depth {row['max_depth_mm']:.2f} mm  "
              f"{row['frac_of_smaller']:.1%} of smaller", flush=True)
rows.sort(key=lambda r: -r["max_depth_mm"])
OUT.write_text(json.dumps(rows, indent=1))
print(f"done in {time.time() - t0:.0f} s; pairs sharing voxels: {sum(r['shared_voxels'] > 0 for r in rows)} of {len(rows)}")
print("depth > 2 mm:", sum(r['max_depth_mm'] > 2 for r in rows), "| 1-2 mm:", sum(1 < r['max_depth_mm'] <= 2 for r in rows),
      "| 0.6-1 mm:", sum(0.6 < r['max_depth_mm'] <= 1 for r in rows), "| <= 0.6 mm (1 voxel):", sum(0 < r['max_depth_mm'] <= 0.6 for r in rows))
