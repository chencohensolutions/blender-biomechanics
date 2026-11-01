#!/usr/bin/env python
"""Second pass: make neighbouring bones consistent where their independent extractions disagree.

check_overlaps.py reports pairs whose meshes interpenetrate.  Two causes, both fixed here
with ONE shared decision per voxel instead of two independent runs:
  * overlap: a voxel inside both final masks -> given to the bone whose own body is nearer
    (distance to each bone's mask without the contested voxels; the fitted-atlas label of the
    pair decides first when both atlas labels exist)
  * orphan: bone the first bone's run assigned to the second bone's basin, which the second
    bone's own run did not keep (typically a rib head) -> given to the second bone when it is
    connected to it
  * atlas arbitration (--atlas-arbitrate): bone lying on the partner's fitted atlas and clearly
    off its own is handed to the partner; the atlas decides ownership, never the surface
  * joint zone (--joint-level): where the bones touch, voxels dimmer than cortex are cartilage
    or partial volume, not bone, and are carved out of both so the joint gap reappears where
    the density says it is
Both masks are then re-meshed from their own fused volumes with the partner's adjusted mask
masked out, so the surfaces meet exactly on the shared boundary.  Nothing is moved.

  python tools/ct/resegment_pairs.py RUN_DIR [--overlaps RUN/qa/overlaps.json]
         [--min-penetration-mm 1.0] [--flags BAD CHECK] [--pairs A:B ...] [--blender]

Originals are kept in RUN/meshes/before_resegment/; each adjusted bone's JSON records
"resegmented_with".  Run check_overlaps.py again afterwards.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402
from extract_bone import MM_TO_BLENDER, ball, largest_component  # noqa: E402


class Bone:
    def __init__(self, meshes_dir, fused_dir, name, atlas_labels=None):
        self.name = name
        self.json_path = meshes_dir / f"{name}.json"
        self.info = json.loads(self.json_path.read_text())
        self.basins_path = meshes_dir / self.info["basins_file"]
        self.basins_img = sitk.ReadImage(str(self.basins_path))
        self.basins = sitk.GetArrayFromImage(self.basins_img).astype(np.uint8)
        self.target_id = int(self.info["target_id"])
        self.ids = {v: int(k) for k, v in self.info["basin_ids"].items()}
        self.mask = self.basins == self.target_id
        fused = Path(self.info["fused"])
        if not fused.exists():
            fused = fused_dir / fused.name
        self.hu_img = sitk.ReadImage(str(fused), sitk.sitkFloat32)
        self.spacing = float(self.hu_img.GetSpacing()[0])
        self.atlas = None
        if atlas_labels and (Path(atlas_labels) / f"{name}.nii.gz").exists():
            self.atlas_path = Path(atlas_labels) / f"{name}.nii.gz"
        else:
            self.atlas_path = None
        self.changed = False
        self.carved = np.zeros(self.mask.shape, bool)  # joint-zone voxels given up: never bone again

    def mask_image(self, mask=None):
        img = sitk.GetImageFromArray((self.mask if mask is None else mask).astype(np.uint8))
        img.CopyInformation(self.basins_img)
        return img

    def onto(self, img, interpolator=sitk.sitkNearestNeighbor):
        """Resample a label image onto this bone's basins grid."""
        return sitk.GetArrayFromImage(sitk.Resample(img, self.basins_img, sitk.Transform(), interpolator, 0, sitk.sitkUInt8)) > 0

    def hu_on_grid(self):
        return sitk.GetArrayFromImage(sitk.Resample(self.hu_img, self.basins_img, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32))


def split_overlap(a, b):
    """Give every voxel inside both masks to one bone (atlas label first, else the nearer body)."""
    sp = a.spacing
    mb_on_a = a.onto(b.mask_image())
    overlap = a.mask & mb_on_a
    n = int(overlap.sum())
    if n:
        core_a = a.mask & ~overlap
        core_b = mb_on_a & ~overlap
        da = ndi.distance_transform_edt(~core_a, sampling=sp)
        db = ndi.distance_transform_edt(~core_b, sampling=sp)
        to_a = overlap & (da <= db)
        if a.atlas_path and b.atlas_path:
            at_a = a.onto(sitk.ReadImage(str(a.atlas_path)))
            at_b = a.onto(sitk.ReadImage(str(b.atlas_path)))
            to_a = overlap & np.where(at_a & ~at_b, True, np.where(at_b & ~at_a, False, da <= db))
        to_b = overlap & ~to_a
        a.mask = a.mask & ~to_b
        b.mask = b.mask & ~b.onto(a.mask_image(to_a))
        a.changed = b.changed = True
    return n


def carve_joint(a, b, joint_level, band_mm):
    """Where the two bones touch, voxels dimmer than cortex (< joint_level HU) within band_mm of the
    partner are cartilage/partial volume, not bone: remove them from both masks so the joint gap
    reappears where the density says it is.  Returns voxels removed from (a, b)."""
    sp = a.spacing
    mb_on_a = a.onto(b.mask_image())
    if not (a.mask.any() and mb_on_a.any()):
        return 0, 0
    d_b = ndi.distance_transform_edt(~mb_on_a, sampling=sp)
    d_a = ndi.distance_transform_edt(~a.mask, sampling=sp)
    zone = (d_b <= band_mm) & (d_a <= band_mm)  # the contact zone, both sides
    if not zone.any():
        return 0, 0
    hu = a.hu_on_grid()
    dim = zone & (hu < joint_level)
    cut_a = a.mask & dim
    cut_b_on_a = mb_on_a & dim
    a.mask &= ~cut_a
    a.carved |= cut_a
    cut_b = b.onto(a.mask_image(cut_b_on_a))
    b.mask &= ~cut_b
    b.carved |= cut_b
    if cut_a.any() or cut_b.any():
        a.changed = b.changed = True
    return int(cut_a.sum()), int(cut_b.sum())


def atlas_reassign(a, b, off_own_mm, on_other_mm):
    """Bone of `a` that lies on/inside b's fitted atlas and clearly off a's own fitted atlas belongs to
    b (and vice versa).  The atlas decides ownership only; the surface still comes from the CT.
    Returns voxels moved (a->b, b->a)."""
    if not (a.atlas_path and b.atlas_path):
        return 0, 0
    sp = a.spacing
    at_a = a.onto(sitk.ReadImage(str(a.atlas_path)))
    at_b = a.onto(sitk.ReadImage(str(b.atlas_path)))
    if not (at_a.any() and at_b.any()):
        return 0, 0
    d_a = ndi.distance_transform_edt(~at_a, sampling=sp)
    d_b = ndi.distance_transform_edt(~at_b, sampling=sp)
    mb_on_a = a.onto(b.mask_image())
    # a's bone that the atlases say is b's
    to_b = a.mask & (d_b <= on_other_mm) & (d_a > off_own_mm)
    # b's bone (seen on a's grid) that the atlases say is a's
    to_a_on_a = mb_on_a & (d_a <= on_other_mm) & (d_b > off_own_mm)
    n_ab = n_ba = 0
    if to_b.any():
        # only pieces that touch b's bone move (no floating islands)
        lab, _ = ndi.label(to_b | mb_on_a)
        keep = np.unique(lab[mb_on_a]); keep = keep[keep > 0]
        to_b &= np.isin(lab, keep)
        if to_b.any():
            a.mask &= ~to_b
            b.mask |= b.onto(a.mask_image(to_b))
            n_ab = int(to_b.sum()); a.changed = b.changed = True
    if to_a_on_a.any():
        lab, _ = ndi.label(to_a_on_a | a.mask)
        keep = np.unique(lab[a.mask]); keep = keep[keep > 0]
        to_a_on_a &= np.isin(lab, keep)
        if to_a_on_a.any():
            a.mask |= to_a_on_a
            b.mask &= ~b.onto(a.mask_image(to_a_on_a))
            n_ba = int(to_a_on_a.sum()); a.changed = b.changed = True
    return n_ab, n_ba


def arbitrate(a, b, level):
    """Adjust a.mask / b.mask so they do not share voxels; returns (overlap_vox, orphans_to_b, orphans_to_a)."""
    sp = a.spacing
    n_overlap = split_overlap(a, b)
    mb_on_a = a.onto(b.mask_image())
    overlap = a.mask & mb_on_a
    # orphans: a's run gave them to b's basin but b did not keep them (and vice versa)
    orph_b = np.zeros_like(overlap)
    if b.name in a.ids:
        orph_b = (a.basins == a.ids[b.name]) & ~a.mask & ~mb_on_a & ~a.carved & ~a.onto(b.mask_image(b.carved))
    ma_on_b = b.onto(a.mask_image())
    orph_a = np.zeros(b.mask.shape, bool)
    if a.name in b.ids:
        orph_a = (b.basins == b.ids[a.name]) & ~b.mask & ~ma_on_b & ~b.carved & ~b.onto(a.mask_image(a.carved))
    # orphans join the partner when connected to it
    n_ob = n_oa = 0
    if getattr(arbitrate, "no_orphans", False):
        orph_b[:] = False; orph_a[:] = False
    if orph_b.any():
        cand = mb_on_a | orph_b
        lab, n = ndi.label(cand)
        keep = np.unique(lab[mb_on_a]); keep = keep[keep > 0]
        gained = np.isin(lab, keep) & orph_b
        if gained.any():
            b.mask |= b.onto(a.mask_image(gained)); n_ob = int(gained.sum()); b.changed = True
    if orph_a.any():
        cand = ma_on_b | orph_a
        lab, n = ndi.label(cand)
        keep = np.unique(lab[ma_on_b]); keep = keep[keep > 0]
        gained = np.isin(lab, keep) & orph_a
        if gained.any():
            a.mask |= a.onto(b.mask_image(gained)); n_oa = int(gained.sum()); a.changed = True
    if n_ob or n_oa:
        n_overlap += split_overlap(a, b)
    return n_overlap, n_ob, n_oa


def remesh(bone, partners, level, smooth_iters, out_dir):
    """Rebuild the bone's mesh from its fused volume with the partners' adjusted masks cut out."""
    sp = bone.spacing
    hu = bone.hu_on_grid()
    mask, _ = largest_component(bone.mask)
    bone.mask = mask
    others = np.zeros(mask.shape, bool)
    for p in partners:
        others |= bone.onto(p.mask_image())
    others |= (bone.basins > 0) & (bone.basins != bone.target_id) & ~mask
    interior = ndi.binary_erosion(mask, structure=ball(1.0))
    allowed = ndi.binary_dilation(mask, structure=ball(1.0))
    vol = hu.copy()
    vol[interior] = np.maximum(vol[interior], level + 100.0)
    vol[~allowed | others | bone.carved] = -1000.0  # carved joint space is not bone, whatever its HU
    zs, ys, xs = np.where(allowed)
    pad = 2
    sl = (slice(max(zs.min() - pad, 0), zs.max() + pad + 1), slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
          slice(max(xs.min() - pad, 0), xs.max() + pad + 1))
    sub = np.pad(vol[sl], 1, constant_values=-1000.0)
    verts, faces, _, _ = measure.marching_cubes(sub, level=level, spacing=(sp,) * 3, allow_degenerate=False)
    origin = np.array(bone.basins_img.GetOrigin())
    offset_vox = np.array([sl[0].start, sl[1].start, sl[2].start], dtype=float) - 1.0
    verts_lps = origin + (verts + offset_vox * sp)[:, ::-1]
    mesh = trimesh.Trimesh(verts_lps, faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    mesh.fix_normals()
    bl = mesh.copy(); bl.apply_scale(MM_TO_BLENDER)
    ply = out_dir / f"{bone.name}.ply"
    bl.export(ply); bl.export(ply.with_suffix(".stl"))
    # label map: the adjusted solid replaces the old one
    basins = bone.basins.copy()
    basins[basins == bone.target_id] = 0
    basins[mask] = bone.target_id
    basins[bone.carved] = 0  # so a later pass does not treat the joint space as an orphan
    lab_img = sitk.GetImageFromArray(basins.astype(np.uint8)); lab_img.CopyInformation(bone.basins_img)
    sitk.WriteImage(lab_img, str(bone.basins_path), useCompression=True)
    lo, hi = mesh.bounds
    bone.info.update({"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                      "volume_cm3": float(mask.sum() * sp**3 / 1000), "bbox_lps_mm_min": lo.tolist(),
                      "bbox_lps_mm_max": hi.tolist(), "size_mm": (hi - lo).tolist(),
                      "watertight": bool(mesh.is_watertight)})
    bone.info.setdefault("resegmented_with", [])
    bone.info["resegmented_with"] = sorted(set(bone.info["resegmented_with"]) | {p.name for p in partners})
    bone.json_path.write_text(json.dumps(bone.info, indent=1))
    return ply


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--overlaps", default=None, help="overlaps.json from check_overlaps.py (default RUN/qa/overlaps.json)")
    ap.add_argument("--flags", nargs="+", default=["BAD"], help="flags to treat (BAD, CHECK)")
    ap.add_argument("--min-penetration-mm", type=float, default=1.0, help="mesh penetration below this is left alone")
    ap.add_argument("--pairs", nargs="*", default=None, help="explicit pairs A:B instead of the report")
    ap.add_argument("--atlas-labels", default=None, help="fitted-atlas labels: decide contested voxels first")
    ap.add_argument("--atlas-arbitrate", action="store_true",
                    help="before the boundary rules: bone lying on the partner's fitted atlas and clearly off its "
                         "own (--atlas-on-other-mm / --atlas-off-own-mm) is handed to the partner")
    ap.add_argument("--atlas-on-other-mm", type=float, default=1.0)
    ap.add_argument("--atlas-off-own-mm", type=float, default=2.5)
    ap.add_argument("--no-orphans", action="store_true",
                    help="skip the orphan rule (it moved mid-thoracic laminae the wrong way on the 2021 scan)")
    ap.add_argument("--level", type=float, default=250.0)
    ap.add_argument("--joint-level", type=float, default=0.0,
                    help="if > 0: along shared boundaries, voxels below this HU within --joint-band-mm of the "
                         "partner are carved out of both bones as joint space (cortex is > 500 HU; 450 is a "
                         "good value); 0 = off")
    ap.add_argument("--joint-band-mm", type=float, default=1.5, help="width of the contact zone examined")
    ap.add_argument("--joint-skip", nargs="*", default=["hip_left:sacrum", "hip_right:sacrum"],
                    help="pairs that are arbitrated but never carved (the sacroiliac joint is wide, oblique and "
                         "thin-walled: carving ate its surfaces at 3 mm slices)")
    ap.add_argument("--smooth-iters", type=int, default=5)
    ap.add_argument("--blender", action="store_true")
    ap.add_argument("--collection", default="bones")
    args = ap.parse_args(argv)

    run = Path(args.run)
    arbitrate.no_orphans = args.no_orphans
    meshes_dir, fused_dir = run / "meshes", run / "fused"
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        rep = json.loads(Path(args.overlaps or run / "qa" / "overlaps.json").read_text())
        pairs = [(p["A"], p["B"]) for p in rep["pairs"]
                 if p["flag"] in args.flags and (p["mesh"].get("penetration_mm") or 0) >= args.min_penetration_mm]
    print(f"{len(pairs)} pair(s) to resegment")
    backup = meshes_dir / "before_resegment"
    backup.mkdir(exist_ok=True)
    bones = {}

    def get(name):
        if name not in bones:
            bones[name] = Bone(meshes_dir, fused_dir, name, args.atlas_labels)
            for ext in (".ply", ".stl", ".json"):
                src = meshes_dir / f"{name}{ext}"
                if src.exists() and not (backup / src.name).exists():
                    shutil.copy2(src, backup / src.name)
            if not (backup / bones[name].basins_path.name).exists():
                shutil.copy2(bones[name].basins_path, backup / bones[name].basins_path.name)
        return bones[name]

    partners = {}
    for a_name, b_name in pairs:
        a, b = get(a_name), get(b_name)
        msg = f"  {a_name} / {b_name}:"
        if args.atlas_arbitrate:
            n_ab, n_ba = atlas_reassign(a, b, args.atlas_off_own_mm, args.atlas_on_other_mm)
            msg += f" atlas moved {n_ab} voxels -> {b_name}, {n_ba} -> {a_name};"
        n_overlap, n_ob, n_oa = arbitrate(a, b, args.level)
        partners.setdefault(a_name, set()).add(b_name)
        partners.setdefault(b_name, set()).add(a_name)
        msg += f" {n_overlap} overlapping voxels split, orphans -> {b_name}: {n_ob}, -> {a_name}: {n_oa}"
        skip = {tuple(sorted(x.split(":"))) for x in (args.joint_skip or [])}
        if args.joint_level > 0 and tuple(sorted((a_name, b_name))) not in skip:
            ca, cb = carve_joint(a, b, args.joint_level, args.joint_band_mm)
            msg += f"; joint zone carved {ca} + {cb} voxels"
        print(msg)
    # a later pair can hand a bone voxels that collide with a third bone settled earlier:
    # sweep all pairs again (overlap split + joint carving) until nothing moves
    for sweep in range(3):
        moved = 0
        for a_name, b_name in pairs:
            a, b = bones[a_name], bones[b_name]
            moved += split_overlap(a, b)
            if args.joint_level > 0 and tuple(sorted((a_name, b_name))) not in skip:
                ca, cb = carve_joint(a, b, args.joint_level, args.joint_band_mm)
                moved += ca + cb
        print(f"  sweep {sweep + 1}: {moved} voxels reassigned or carved")
        if moved == 0:
            break
    changed = [n for n, bn in bones.items() if bn.changed]
    print(f"re-meshing {len(changed)} bone(s)")
    manifest = json.loads((run / "run.json").read_text())
    for name in changed:
        ply = remesh(bones[name], [bones[p] for p in partners.get(name, ())], args.level, args.smooth_iters, meshes_dir)
        info = bones[name].info
        print(f"  {name}: {info['faces']} faces, {info['volume_cm3']:.1f} cm^3")
        if args.blender:
            try:
                collection = "cartilage" if name.startswith("costal_cartilage_") else args.collection
                blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                    "path": str(ply), "name": name, "collection": collection, "decimate": 0.0,
                    "props": {"ct_group": manifest["outputs"].get(name, {}).get("group", ""),
                              "ct_run": manifest.get("run_tag", ""), "ct_resegmented": ",".join(sorted(partners.get(name, ())))}})
            except OSError as exc:
                print(f"!! Blender not reachable ({exc})")
    (run / "qa").mkdir(exist_ok=True)
    (run / "qa" / "resegment.json").write_text(json.dumps({"pairs": pairs, "changed": changed}, indent=1))


if __name__ == "__main__":
    main()
