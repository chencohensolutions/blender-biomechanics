#!/usr/bin/env python
"""Segment every intervertebral disc of a CT run and measure it for a disc force model.

A disc is the soft tissue that fills the gap between two vertebral bodies.  For every pair of
consecutive vertebrae (pipeline.VERTEBRAE order, C7-T1 ... L5-sacrum on ct2021) whose meshes
exist, the disc is found from the two final bone masks extract_bone.py left in the run:

  <run>/meshes/<bone>.json          basins_file, target_id (+ bbox, ...)
  <run>/meshes/<bone>_basins.nrrd   uint8 labels; voxels == target_id are the final solid bone
  <run>/fused/<bone>_fused.nrrd     fused CT (HU, LPS, 0.4508 mm), the bone + 20 mm

Method per level (upper bone U, lower bone L), on U's fused grid
  1. L's final mask is resampled onto U's grid (nearest, SimpleITK physical coordinates);
     dU, dL = Euclidean distance (mm) to the U and L masks
  2. "between" voxels: soft tissue (HU in [--hu-min, --hu-max] = [40, 250] after a 0.5 mm
     Gaussian; excludes CSF, fat and bone) outside both masks with dU, dL <= 20 mm whose
     directions to the nearest U point and to the nearest L point are opposed (unit gradients
     of dU and dL, dot < -0.3).  Where the masks touch (thin thoracic discs that the bone
     extraction closed over) the soft voxels inside either mask within 2.5 mm of the other one
     are added (HU ceiling 350: such a disc is thinner than the scan resolution).  Two guards
     follow, both needed on ct2021 (thoracic bone masks are partly hollow, so the spec's region
     leaked into spongy bone and the costovertebral joints):
       corridor  inside both vertebral bodies' outlines (+3 mm) seen along the body-to-body axis.
                 A body = the bone mask opened by a 5 mm ball (pedicles, arch and processes go),
                 largest piece, regrown 4 mm back into the mask
       trough    bone (>= 250 HU, and >= 100 HU brighter than the voxel) within 8 mm both above
                 and below along that axis: a disc is a dark band between two endplates
     The largest connected component (pieces < 1 mm apart count as connected, a thin disc
     breaks up) is the raw disc; the other components >= 5 mm3 are counted as dropped
  3. disc axis: PCA of the raw disc, normal = smallest-variance axis oriented from L to U (the
     body-to-body axis if the disc is not pancake-shaped), midplane through the centroid.
     Footprint: the endplate voxels of each body (body bone within 4 mm of the raw disc whose
     nearest point on the other bone lies along the normal, |cos| >= 0.7) are projected on the
     midplane; the convex hulls of U's and of L's, each shrunk by 1 mm, are intersected (the
     disc lies on both endplates) and clip the raw disc (anterior soft tissue, canal, rib heads).
     The axis is refitted on the clipped disc and the clip repeated once: the final disc mask
  4. parameters (LPS mm; *_m = Blender metres = LPS mm * 0.001)
       center              centroid of the final mask
       normal              unit disc axis, from the lower to the upper vertebra (local +Z)
       major_axis / minor_axis   in-plane PCA axes of the footprint (local +X / +Y; major
                           points to the patient's left, minor = normal x major, right-handed)
       height_mean/min/max_mm  thickness along the normal per 1 mm in-plane column (voxel
                           count * voxel volume / 1 mm^2) over the inner columns (> 1.5 mm
                           inside the footprint edge); min/max = 5th/95th percentile.
                           height_anterior/posterior_mm: mean over the anterior/posterior
                           third of the footprint (along the in-plane direction nearest -Y/+Y)
       area_mm2            area of the midplane footprint (rasterised at 0.25 mm)
       coverage            disc columns / footprint area: < ~0.8 means the disc mask is patchy
                           (thin or partly undetected disc), so the heights are less reliable
       flags               thin (mean height < 3 mm), patchy (coverage < 0.8), one_endplate
                           (only one bone gave an outline), steep_wedge (> 15 deg), small_area
                           (< 300 mm2): a flagged level should not be trusted as measured
       radius_mm           equivalent radius sqrt(area / pi)
       major/minor_radius_mm   footprint semi-axes 2*sqrt(PCA variance) (exact for an ellipse)
       volume_cm3          final mask volume
       gap_mean/min/max_mm           bone-to-bone distance along the normal per column (lowest U
                                     bone voxel minus highest L bone voxel): the disc height as the
                                     bone masks see it, independent of the disc HU window; use this
                                     for a force model, the voxel heights above are lower bounds
       endplate_distance_mm          distance between the two endplate centroids (biased high by
                                     the 3 mm bone layer they average over)
       upper/lower_endplate_center   centroid of the U / L bone voxels within 3 mm of the disc
       wedge_deg           angle between the two endplate planes: atan of the gradient of a
                           plane fitted to the column heights (inner columns, trimmed), i.e. how
                           fast the disc thickens across it; wedge_sagittal_deg its sagittal
                           part, > 0 = taller anteriorly (lordotic); upper/lower_endplate_normal =
                           the disc normal tilted by -/+ half of that gradient
       hu_rings            mean HU of the disc in 5 rings of normalized in-plane radius
                           (0-0.2 ... 0.8-1.0; 1 = footprint edge along that ray, measured from
                           the footprint centroid); null in the JSON (-1 in Blender) when empty
       nucleus_radius_ratio  0.5, a stated default, not a measurement: CT does not separate the
                           nucleus pulposus from the annulus (both ~70-120 HU; that needs MRI T2),
                           so a force model should treat the central 50 % of the radius as the
                           firmer, pressurised core by assumption.  hu_rings shows only density
  5. mesh: marching cubes of the final mask smoothed by a 0.6 mm Gaussian at 0.5, Taubin
     smoothing (10 iterations), PLY + STL in Blender metres with the scanner's LPS axes and
     origin, exactly like extract_bone.py, so discs sit between the bone meshes

CT limits: heights are what the scan resolves (the 250 HU partial-volume boundary of 0.45 mm
voxels from thicker source slices); discs thinner than ~3 mm come out patchy and thin.  The
nucleus/annulus split is not visible at all (see nucleus_radius_ratio).

Outputs (default --out <RUN_DIR>/discs/): <level>.ply/.stl/.json (level names disc_T12_L1,
disc_L5_sacrum ...; a failed level's stale files are removed), discs.json (all levels, failed
ones with "error"), and a printed table.

--blender imports every disc into the running Blender (BlenderMCP socket, localhost:9876)
into --collection (default discs), plus an Empty "<level>_center" (ARROWS, size = equivalent
radius) at the centre whose local Z is the disc normal and local X the major axis, carrying
the parameters as custom properties (lengths in mm unless named *_m / endplate centres in m).
--blender-only skips the extraction and pushes the JSONs already in --out.

  python tools/ct/extract_discs.py --run ".../runs/ct2021" --meshes-dir ".../runs/ct2021/meshes_pass5"
  python tools/ct/extract_discs.py --run ".../runs/ct2021" --blender-only --blender --collection discs
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402
from pipeline import VERTEBRAE  # noqa: E402

MM_TO_BLENDER = 0.001
NUCLEUS_RADIUS_RATIO = 0.5
RINGS = 5

DISC_EMPTY = r'''
import bpy
from mathutils import Matrix
name, col_name, props = %(name)r, %(collection)r, %(props)r
center, normal, major, size = %(center)r, %(normal)r, %(major)r, %(size)r
n = [float(c) for c in normal]
x = [float(c) for c in major]
y = [n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0]]
if name in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
obj = bpy.data.objects.new(name, None)
obj.empty_display_type = "ARROWS"
obj.empty_display_size = size
col = bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
if col.name not in [c.name for c in bpy.context.scene.collection.children]:
    bpy.context.scene.collection.children.link(col)
col.objects.link(obj)
obj.matrix_world = Matrix(((x[0], y[0], n[0], center[0]),
                           (x[1], y[1], n[1], center[1]),
                           (x[2], y[2], n[2], center[2]),
                           (0.0, 0.0, 0.0, 1.0)))
for k, v in props.items():
    obj[k] = v
if bpy.data.filepath:
    bpy.ops.wm.save_mainfile()
print("empty", obj.name, "at", [round(c, 4) for c in obj.matrix_world.translation],
      "z", [round(c, 3) for c in obj.matrix_world.col[2][:3]])
'''


def level_name(upper, lower):
    short = lambda b: b.replace("vertebrae_", "")  # noqa: E731
    return f"disc_{short(upper)}_{short(lower)}"


def unit(v):
    return v / np.linalg.norm(v)


def unit_rows(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def vertebral_body(mask, radius_mm, spacing, regrow_mm=4.0):
    """Morphological opening by a ball of radius_mm, largest piece, regrown regrow_mm back into the
    mask (an irregular or porous endplate layer that the opening ate comes back): the body
    survives, the thinner pedicles, laminae and processes are left as short stumps at most
    (falls back to the whole mask if nothing survives)."""
    if radius_mm <= 0:
        return mask
    core = ndi.distance_transform_edt(mask, sampling=spacing) > radius_mm
    if not core.any():
        return mask
    opened = (ndi.distance_transform_edt(~core, sampling=spacing) <= radius_mm) & mask
    lab, n = ndi.label(opened)
    if n > 1:
        opened = lab == int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    return mask & (ndi.distance_transform_edt(~opened, sampling=spacing) <= regrow_mm)


def pca(points):
    """(centroid, eigenvalues descending, eigenvectors as rows in that order)."""
    c = points.mean(0)
    w, v = np.linalg.eigh(np.cov((points - c).T))
    order = np.argsort(w)[::-1]
    return c, w[order], v[:, order].T


def largest_component(mask, voxel_mm3, min_mm3=5.0, tol_mm=0.0, spacing=1.0):
    """Largest connected piece (pieces closer than tol_mm count as connected), the number of other
    pieces >= min_mm3 and their total volume."""
    if not mask.any():
        return mask, 0, 0.0
    linked = mask if tol_mm <= 0 else ndi.distance_transform_edt(~mask, sampling=spacing) <= tol_mm
    lab, n = ndi.label(linked)
    sizes = np.bincount(lab[mask], minlength=n + 1)[1:]
    big = int(np.argmax(sizes))
    others = np.delete(sizes, big) * voxel_mm3
    return mask & (lab == big + 1), int((others >= min_mm3).sum()), float(others.sum())


class Hull2D:
    """Convex polygon as half-planes a.x + b <= 0, with an inward offset and a gauge (normalized
    radius: 0 at `center`, 1 on the boundary along every ray)."""

    def __init__(self, pts2d, shrink=0.0):
        hull = ConvexHull(pts2d)
        self.eq = hull.equations.copy()  # unit outward normals
        self.eq[:, 2] += shrink
        self.area = float(hull.volume)  # of the unshrunk polygon

    @classmethod
    def from_equations(cls, eq):
        obj = cls.__new__(cls)
        obj.eq = np.asarray(eq, float).copy()
        return obj

    def shrunk(self, mm):
        return Hull2D.from_equations(self.eq + np.array([0.0, 0.0, mm]))

    def raster(self, cell_mm):
        """Points of a cell_mm grid inside the polygon, and the cell area (for area and moments)."""
        a, b = self.eq[:, :2], self.eq[:, 2]
        # bounding box from the pairwise edge-line intersections that lie inside
        verts = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                m = np.array([a[i], a[j]])
                if abs(np.linalg.det(m)) > 1e-9:
                    verts.append(np.linalg.solve(m, -np.array([b[i], b[j]])))
        verts = np.array(verts)
        verts = verts[self.contains(verts, tol=1e-6)]
        if len(verts) < 3:
            return np.zeros((0, 2)), cell_mm ** 2
        lo, hi = verts.min(0), verts.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + cell_mm, cell_mm), np.arange(lo[1], hi[1] + cell_mm, cell_mm))
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return pts[self.contains(pts)], cell_mm ** 2

    def contains(self, pts2d, tol=0.0):
        return np.all(pts2d @ self.eq[:, :2].T + self.eq[:, 2] <= tol, axis=1)

    def gauge(self, pts2d, center):
        a, b = self.eq[:, :2], self.eq[:, 2]
        reach = -(a @ center + b)  # distance from the centre to each edge line (> 0 inside)
        return np.max(((pts2d - center) @ a.T) / reach, axis=1)


class Grid:
    """Index (z, y, x) <-> LPS mm for one SimpleITK image."""

    def __init__(self, img):
        self.spacing = np.array(img.GetSpacing())
        self.origin = np.array(img.GetOrigin())
        self.direction = np.array(img.GetDirection()).reshape(3, 3)

    def to_lps(self, zyx):
        return self.origin + (np.asarray(zyx, float)[..., ::-1] * self.spacing) @ self.direction.T


def read_mask(meshes_dir, bone, ref_img):
    info = json.loads((meshes_dir / f"{bone}.json").read_text())
    lab = sitk.ReadImage(str(meshes_dir / info["basins_file"]))
    res = sitk.Resample(lab, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(res) == int(info["target_id"])


def available(meshes_dir, fused_dir, bone, need_fused):
    j = meshes_dir / f"{bone}.json"
    if not j.exists():
        return False
    basins = json.loads(j.read_text()).get("basins_file")
    if not basins or not (meshes_dir / basins).exists():
        return False
    return not need_fused or (fused_dir / f"{bone}_fused.nrrd").exists()


def extract_level(upper, lower, run, meshes_dir, out_dir, args, log=print, debug=None):
    """Segment, measure and mesh the disc between `upper` and `lower`; returns the JSON dict.
    `debug` (a dict) receives the intermediate masks on the cropped grid for QA images."""
    fused_img = sitk.ReadImage(str(run / "fused" / f"{upper}_fused.nrrd"), sitk.sitkFloat32)
    grid = Grid(fused_img)
    sp = float(grid.spacing[0])
    if not np.allclose(grid.spacing, sp):
        raise ValueError(f"{upper}: anisotropic fused grid {grid.spacing}")
    vox_mm3 = sp ** 3
    hu_full = sitk.GetArrayFromImage(fused_img)
    mu_full = read_mask(meshes_dir, upper, fused_img)
    ml_full = read_mask(meshes_dir, lower, fused_img)
    if not mu_full.any() or not ml_full.any():
        raise ValueError("a bone mask does not reach the upper bone's fused grid")

    # 1. distance fields; everything below works in the crop where both bones are within reach
    du_full = ndi.distance_transform_edt(~mu_full, sampling=sp)
    dl_full = ndi.distance_transform_edt(~ml_full, sampling=sp)
    reach = (du_full <= args.reach_mm) & (dl_full <= args.reach_mm)
    if not reach.any():
        raise ValueError(f"the bones are more than {2 * args.reach_mm} mm apart")
    pad = int(np.ceil(6.0 / sp))
    zs, ys, xs = np.where(reach)
    sl = tuple(slice(max(int(a.min()) - pad, 0), int(a.max()) + pad + 1) for a in (zs, ys, xs))
    off = np.array([s.start for s in sl])
    hu, mu, ml, du, dl = (a[sl] for a in (hu_full, mu_full, ml_full, du_full, dl_full))
    del hu_full, du_full, dl_full, reach
    hu_s = ndi.gaussian_filter(hu, args.hu_sigma_mm / sp)

    # 2. between region: soft tissue whose nearest-U and nearest-L directions are opposed
    gu = np.stack(np.gradient(du, sp))
    gl = np.stack(np.gradient(dl, sp))
    gu /= np.maximum(np.linalg.norm(gu, axis=0), 1e-6)
    gl /= np.maximum(np.linalg.norm(gl, axis=0), 1e-6)
    opposed = np.einsum("i...,i...->...", gu, gl) < args.dot_max
    del gu
    soft = (hu_s >= args.hu_min) & (hu_s <= args.hu_max)
    between = opposed & (du > 0) & (dl > 0) & (du <= args.reach_mm) & (dl <= args.reach_mm) & soft
    # where the two bone masks touch (thin discs the bone extraction closed over), the disc is the
    # low-density trough inside the masks along their interface; a disc thinner than the scan's
    # resolution never gets below 250 HU there, so only the trough test below bounds it
    contact = (((mu & (dl <= args.contact_band_mm)) | (ml & (du <= args.contact_band_mm)))
               & (hu_s >= args.hu_min) & (hu_s <= args.contact_hu_max))
    n_contact = int(contact.sum())
    between |= contact

    idx = lambda m: np.argwhere(m) + off  # noqa: E731  (full-grid indices)
    body_u = vertebral_body(mu, args.body_open_mm, sp, args.body_regrow_mm)
    body_l = vertebral_body(ml, args.body_open_mm, sp, args.body_regrow_mm)
    pts_bu, pts_bl = grid.to_lps(idx(body_u)), grid.to_lps(idx(body_l))
    axis0 = unit(pts_bu.mean(0) - pts_bl.mean(0))  # provisional spine axis, lower -> upper body
    cand_idx = np.argwhere(between)
    cand_pts = grid.to_lps(cand_idx + off)
    # corridor: inside both vertebral bodies' outlines seen along the axis (drops costovertebral,
    # foraminal and paravertebral soft tissue before the largest component is chosen)
    e1 = unit(np.cross(axis0, [1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else np.cross(axis0, [0.0, 1.0, 0.0]))
    e2 = np.cross(axis0, e1)
    flat = lambda p: np.stack([p @ e1, p @ e2], axis=1)  # noqa: E731
    keep = (Hull2D(flat(pts_bu), shrink=-args.corridor_margin_mm).contains(flat(cand_pts))
            & Hull2D(flat(pts_bl), shrink=-args.corridor_margin_mm).contains(flat(cand_pts)))
    # trough: darker than bone on both sides along the axis within --trough-mm (a disc is a thin
    # dark band between two endplates; spongy bone inside an incompletely masked body is not)
    step_idx = (axis0 @ grid.direction)[::-1] / sp  # one mm along the axis in (z, y, x) index units
    base = hu_s[tuple(cand_idx.T)]
    sides = []
    for sign in (1.0, -1.0):
        best = np.full(len(cand_idx), -np.inf, dtype=np.float32)
        for a in np.arange(0.5, args.trough_mm + 1e-6, 0.5):
            q = (cand_idx + sign * a * step_idx).T
            best = np.maximum(best, ndi.map_coordinates(hu_s, q, order=1, mode="nearest"))
        sides.append(best)
    side = np.minimum(*sides)
    keep &= (side >= args.hu_max) & (side - base >= args.trough_depth_hu)
    between = np.zeros_like(between)
    between[tuple(cand_idx[keep].T)] = True
    del cand_idx, cand_pts
    raw, n_dropped, dropped_mm3 = largest_component(between, vox_mm3, tol_mm=args.link_mm, spacing=sp)
    log(f"  between {int(between.sum())} vox ({n_contact} soft voxels at the masks' contact), raw disc "
        f"{raw.sum() * vox_mm3 / 1000:.2f} cm3, {n_dropped} component(s) >= 5 mm3 dropped ({dropped_mm3:.0f} mm3)")
    if debug is not None:
        debug.update(hu=hu, mu=mu, ml=ml, between=between, raw=raw, disc=raw, off=off, grid=grid,
                     cand_u=np.zeros_like(raw), cand_l=np.zeros_like(raw))
    if raw.sum() * vox_mm3 < args.min_disc_mm3:
        raise ValueError(f"no disc found (largest between-component {raw.sum() * vox_mm3:.0f} mm3)")
    pu_all, pl_all = pts_bu.mean(0), pts_bl.mean(0)
    del pts_bu, pts_bl

    # 3. axis + footprint.  Endplate voxels: vertebral-body bone (pedicles, arch and processes
    # opened away) within --endplate-mm of the raw disc whose nearest point on the other bone lies
    # along the normal (|cos| >= --facing-cos)
    raw_idx = np.argwhere(raw)
    gap = float(np.percentile(du[raw] + dl[raw], 95))
    near_raw = ndi.distance_transform_edt(~raw, sampling=sp) <= args.endplate_mm
    cand_u = body_u & near_raw
    cand_l = body_l & near_raw
    del near_raw
    to_l = unit_rows(gl[:, cand_u].T[:, ::-1] @ grid.direction.T)  # (n, 3) LPS, direction to nearest L
    del gl
    to_u = unit_rows(np.stack(np.gradient(du, sp))[:, cand_l].T[:, ::-1] @ grid.direction.T)
    pts_u = grid.to_lps(idx(cand_u))
    pts_l = grid.to_lps(idx(cand_l))
    raw_pts = grid.to_lps(raw_idx + off)

    def disc_axis(points, fallback):
        c, w, v = pca(points)
        n = v[2] if np.sqrt(w[1] / max(w[2], 1e-9)) >= 2.5 else fallback  # not a pancake: keep the old axis
        n = n if np.dot(n, pu_all - pl_all) > 0 else -n
        return c, unit(n)

    def footprint(c0, normal):
        """Half-planes (in the in-plane frame e1, e2 at c0) of the intersection of both endplate hulls."""
        e1 = unit(np.cross(normal, [0.0, 1.0, 0.0]))  # ~ +X (left) for a near-vertical normal
        e2 = np.cross(normal, e1)
        plane = lambda p: np.stack([(p - c0) @ e1, (p - c0) @ e2], axis=1)  # noqa: E731
        eqs = []
        for bone, pts_b, dirs in ((upper, pts_u, to_l), (lower, pts_l, to_u)):
            ep = pts_b[np.abs(dirs @ normal) >= args.facing_cos]
            if len(ep) >= 50:
                eqs.append(Hull2D(plane(ep), shrink=args.shrink_mm).eq)
            else:
                log(f"  ({bone}: only {len(ep)} endplate voxels, its outline is not used)")
                one_sided.add(bone)
        if not eqs:
            raise ValueError("too few endplate voxels on both bones for a footprint")
        return e1, e2, plane, Hull2D.from_equations(np.vstack(eqs))

    one_sided = set()
    disc = raw
    normal = axis0
    for _ in range(2):
        c0, normal = disc_axis(grid.to_lps(idx(disc)), normal)
        e1, e2, plane, foot = footprint(c0, normal)
        keep = foot.contains(plane(raw_pts))
        disc = np.zeros_like(raw)
        disc[tuple(raw_idx[keep].T)] = True
        disc, _, _ = largest_component(disc, vox_mm3, tol_mm=args.link_mm, spacing=sp)
    if disc.sum() * vox_mm3 < args.min_disc_mm3:
        raise ValueError(f"disc vanished after the footprint clip ({disc.sum() * vox_mm3:.0f} mm3 left)")

    # 4. parameters on the final mask, in the frame (major, minor, normal) of its footprint
    one_sided.clear()
    pts = grid.to_lps(idx(disc))
    center, normal = disc_axis(pts, normal)
    e1, e2, plane, foot = footprint(center, normal)
    fp_pts, fp_cell = foot.raster(args.raster_mm)
    area = len(fp_pts) * fp_cell
    fp_c, fp_w, fp_v = pca(fp_pts)
    major = unit(fp_v[0][0] * e1 + fp_v[0][1] * e2)
    if major[0] < 0:  # point the major axis to the patient's left (+X) for a stable frame
        major = -major
    minor = np.cross(normal, major)
    rot = np.array([[major @ e1, minor @ e1], [major @ e2, minor @ e2]])  # (e1, e2) -> (major, minor)
    foot = Hull2D.from_equations(np.column_stack([foot.eq[:, :2] @ rot, foot.eq[:, 2]]))
    fp_c = fp_c @ rot
    major_r, minor_r = (2.0 * np.sqrt(w) for w in fp_w)
    uv = np.stack([(pts - center) @ major, (pts - center) @ minor], axis=1)

    # per-column thickness on a 1 mm in-plane grid
    cols = np.floor(uv / args.column_mm).astype(np.int64)
    key, counts = np.unique(cols, axis=0, return_counts=True)
    thick = counts * vox_mm3 / args.column_mm ** 2
    col_centres = (key + 0.5) * args.column_mm
    inner = foot.shrunk(args.inner_mm).contains(col_centres)
    coverage = float(len(key) * args.column_mm ** 2 / area)
    if inner.sum() < 10:
        inner = np.ones(len(thick), bool)
    t = thick[inner]
    ant_dir = np.array([np.dot(major, [0, -1, 0]), np.dot(minor, [0, -1, 0])])  # -Y = anterior
    ant_dir = ant_dir / max(np.linalg.norm(ant_dir), 1e-9)
    along = col_centres[inner] @ ant_dir
    lo3, hi3 = np.quantile(along, [1 / 3, 2 / 3])

    # endplate centres: bone within 3 mm of the final disc
    near = ndi.distance_transform_edt(~disc, sampling=sp) <= args.contact_mm
    ep_u = grid.to_lps(idx(mu & near))
    ep_l = grid.to_lps(idx(ml & near))
    if len(ep_u) < 10 or len(ep_l) < 10:
        raise ValueError("no endplate bone next to the disc")

    # bone-to-bone gap per column: lowest U-bone voxel minus highest L-bone voxel along the
    # normal, independent of the disc HU window (only the bone masks' partial-volume bias remains)
    def col_extreme(points, reducer, fill):
        uvp = np.stack([(points - center) @ major, (points - center) @ minor], axis=1)
        ckey, inv = np.unique(np.floor(uvp / args.column_mm).astype(np.int64), axis=0, return_inverse=True)
        h = (points - center) @ normal
        out = np.full(len(ckey), fill)
        reducer.at(out, inv.ravel(), h)
        return {tuple(k): v for k, v in zip(ckey, out)}
    u_low = col_extreme(ep_u, np.minimum, np.inf)
    l_high = col_extreme(ep_l, np.maximum, -np.inf)
    gaps = np.array([u_low[k] - l_high[k] for k in map(tuple, key[inner]) if k in u_low and k in l_high])
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) < 5:
        gaps = np.array([np.nan])
    # wedge: plane fit of the column thickness h(u, v) = g . (u, v) + h0 over the inner columns
    # (trimmed at 2.5 sigma); the endplates are the midplane tilted by +-g/2, so the angle between
    # them is atan(|g|) and the sagittal part is atan(g . anterior)
    design = np.column_stack([col_centres[inner], np.ones(int(inner.sum()))])
    ok = np.ones(len(t), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(design[ok], t[ok], rcond=None)[0]
        res = t - design @ coef
        ok = np.abs(res) <= 2.5 * max(res[ok].std(), 1e-6)
    grad = coef[:2]
    nu = unit(normal - 0.5 * (grad[0] * major + grad[1] * minor))
    nl = unit(normal + 0.5 * (grad[0] * major + grad[1] * minor))
    wedge = float(np.degrees(np.arctan(np.linalg.norm(grad))))
    wedge_sag = float(np.degrees(np.arctan(grad @ ant_dir)))  # > 0: taller anteriorly (lordotic)

    # radial HU rings
    hu_d = hu[disc]
    r = np.clip(foot.gauge(uv, fp_c), 0, 1 - 1e-9)
    ring = (r * RINGS).astype(int)
    hu_rings = [float(hu_d[ring == k].mean()) if np.any(ring == k) else None for k in range(RINGS)]

    # 5. mesh
    zs, ys, xs = np.where(disc)
    pad = 4
    msl = tuple(slice(max(int(a.min()) - pad, 0), int(a.max()) + pad + 1) for a in (zs, ys, xs))
    sub = np.pad(ndi.gaussian_filter(disc[msl].astype(np.float32), args.mesh_sigma_mm / sp), 1)
    verts, faces, _, _ = measure.marching_cubes(sub, level=0.5, allow_degenerate=False)
    verts_idx = verts + np.array([s.start for s in msl]) - 1.0 + off
    import trimesh
    mesh = trimesh.Trimesh(grid.to_lps(verts_idx), faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    if args.smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=args.smooth_iters)
    mesh.fix_normals()
    bl = mesh.copy()
    bl.apply_scale(MM_TO_BLENDER)
    name = level_name(upper, lower)
    ply, stl = out_dir / f"{name}.ply", out_dir / f"{name}.stl"
    bl.export(ply)
    bl.export(stl)

    flags = []
    if t.mean() < 3.0:
        flags.append("thin")  # below what the scan resolves well: heights are lower bounds
    if coverage < 0.8:
        flags.append("patchy")  # the disc mask fills < 80 % of the footprint
    if one_sided:
        flags.append("one_endplate")  # the footprint comes from one bone's outline only
    if wedge > 15.0:
        flags.append("steep_wedge")
    if area < 300.0:
        flags.append("small_area")
    m = lambda v: [float(c) * MM_TO_BLENDER for c in v]  # noqa: E731
    f = lambda v: [float(c) for c in v]  # noqa: E731
    cu, cl = ep_u.mean(0), ep_l.mean(0)
    info = {
        "level": name, "upper_bone": upper, "lower_bone": lower,
        "center_mm": f(center), "center_m": m(center),
        "normal": f(normal), "major_axis": f(major), "minor_axis": f(minor),
        "height_mean_mm": float(t.mean()), "height_min_mm": float(np.quantile(t, 0.05)),
        "height_max_mm": float(np.quantile(t, 0.95)), "height_median_mm": float(np.median(t)),
        "height_anterior_mm": float(t[along >= hi3].mean()), "height_posterior_mm": float(t[along <= lo3].mean()),
        "area_mm2": float(area), "radius_mm": float(np.sqrt(area / np.pi)),
        "major_radius_mm": float(major_r), "minor_radius_mm": float(minor_r),
        "width_major_mm": float(np.ptp(uv[:, 0])), "width_minor_mm": float(np.ptp(uv[:, 1])),
        "volume_cm3": float(disc.sum() * vox_mm3 / 1000),
        "upper_endplate_center_mm": f(cu), "lower_endplate_center_mm": f(cl),
        "upper_endplate_center_m": m(cu), "lower_endplate_center_m": m(cl),
        "endplate_distance_mm": float(np.linalg.norm(cu - cl)),
        "gap_mean_mm": float(np.nanmean(gaps)), "gap_min_mm": float(np.nanquantile(gaps, 0.05)),
        "gap_max_mm": float(np.nanquantile(gaps, 0.95)), "gap_columns": int(np.isfinite(gaps).sum()),
        "upper_endplate_normal": f(nu), "lower_endplate_normal": f(nl),
        "wedge_deg": wedge, "wedge_sagittal_deg": wedge_sag,
        "hu_rings": hu_rings, "hu_mean": float(hu_d.mean()),
        "nucleus_radius_ratio": NUCLEUS_RADIUS_RATIO,
        "nucleus_note": "default, not measured: CT does not separate nucleus from annulus",
        "flags": flags,
        "components_dropped": n_dropped, "dropped_mm3": dropped_mm3,
        "raw_volume_cm3": float(raw.sum() * vox_mm3 / 1000),
        "inside_mask_fraction": float((disc & (mu | ml)).sum() / disc.sum()),
        "columns_used": int(inner.sum()), "coverage": coverage, "gap_p95_mm": gap,
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "ply": str(ply), "stl": str(stl), "fused": str(run / "fused" / f"{upper}_fused.nrrd"),
        "coords": "LPS mm (x=left, y=posterior, z=superior); *_m = Blender metres = LPS mm * 0.001, "
                  "scanner origin kept",
        "params": {k: getattr(args, k) for k in ("reach_mm", "dot_max", "contact_band_mm", "contact_hu_max", "corridor_margin_mm", "trough_mm", "trough_depth_hu", "hu_min", "hu_max", "hu_sigma_mm",
                                                 "link_mm", "min_disc_mm3", "endplate_mm", "body_open_mm", "body_regrow_mm", "facing_cos", "raster_mm", "shrink_mm", "column_mm",
                                                 "inner_mm", "contact_mm", "mesh_sigma_mm", "smooth_iters")},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1))
    if debug is not None:
        debug.update(disc=disc, cand_u=cand_u, cand_l=cand_l)
    return info


def fmt(v):
    return "-" if v is None else f"{v:.0f}"


def print_table(results):
    print(f"\n{'level':16s} {'h_mean':>6s} {'h_min':>6s} {'h_max':>6s} {'gap':>5s} {'ant':>5s} {'post':>5s} {'area':>6s} "
          f"{'2a x 2b mm':>11s} {'cov':>4s} {'vol':>5s} {'wedge':>5s} {'w_sag':>6s} {'drop':>4s} {'HU c':>5s} {'HU rim':>6s}  flags")
    for r in results:
        if "error" in r:
            print(f"{r['level']:16s} FAILED: {r['error']}")
            continue
        print(f"{r['level']:16s} {r['height_mean_mm']:6.1f} {r['height_min_mm']:6.1f} {r['height_max_mm']:6.1f} "
              f"{r.get('gap_mean_mm', float('nan')):5.1f} {r['height_anterior_mm']:5.1f} {r['height_posterior_mm']:5.1f} {r['area_mm2']:6.0f} "
              f"{2 * r['major_radius_mm']:5.0f}x{2 * r['minor_radius_mm']:<5.0f} {r['coverage']:4.2f} {r['volume_cm3']:5.1f} "
              f"{r['wedge_deg']:5.1f} {r['wedge_sagittal_deg']:6.1f} {r['components_dropped']:4d} "
              f"{fmt(r['hu_rings'][0]):>5s} {fmt(r['hu_rings'][-1]):>6s}  {','.join(r.get('flags', []))}")
    print("heights mm from disc voxels (inner columns, min/max = 5th/95th pct); gap = bone-to-bone distance "
          "along the normal per column (mean), independent of the disc HU window;\narea mm2 of the endplate footprint, cov = disc "
          "columns / footprint area,\nvol cm3, wedge deg (w_sag > 0 = taller anteriorly), drop = dropped "
          "components >= 5 mm3, HU centre ring / rim ring\nflags: thin = mean height < 3 mm (below what the scan "
          "resolves; heights are lower bounds), patchy = coverage < 0.8, one_endplate = footprint from one\n"
          "bone only, steep_wedge > 15 deg, small_area < 300 mm2: treat flagged levels as unreliable")


def push(results, args):
    try:
        for r in results:
            if "error" in r:
                continue
            props = {"level": r["level"], "height_mm": r["height_mean_mm"], "area_mm2": r["area_mm2"],
                     "wedge_deg": r["wedge_deg"]}
            print("  " + str(blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                "path": r["ply"], "name": r["level"], "collection": args.collection, "decimate": 0.0,
                "props": props})).strip())
            print("  " + str(blender_bridge.run_code(empty_code(r, args.collection))).strip())
    except (ConnectionRefusedError, OSError) as exc:
        print(f"!! Blender not reachable on localhost:9876 ({exc}); discs stay on disk")


def empty_code(r, collection):
    props = {"level": r["level"], "upper_bone": r["upper_bone"], "lower_bone": r["lower_bone"],
             "normal": r["normal"], "height_mean_mm": r["height_mean_mm"], "height_min_mm": r["height_min_mm"],
             "height_max_mm": r["height_max_mm"], "gap_mean_mm": r.get("gap_mean_mm", -1.0),
             "gap_min_mm": r.get("gap_min_mm", -1.0), "gap_max_mm": r.get("gap_max_mm", -1.0),
             "endplate_distance_mm": r.get("endplate_distance_mm", -1.0),
             "area_mm2": r["area_mm2"], "radius_mm": r["radius_mm"],
             "major_radius_mm": r["major_radius_mm"], "minor_radius_mm": r["minor_radius_mm"],
             "wedge_deg": r["wedge_deg"], "nucleus_radius_ratio": r["nucleus_radius_ratio"],
             "upper_endplate_center": r["upper_endplate_center_m"],
             "lower_endplate_center": r["lower_endplate_center_m"],
             "hu_rings": [-1.0 if v is None else v for v in r["hu_rings"]], "flags": ",".join(r.get("flags", []))}
    return DISC_EMPTY % {"name": r["level"] + "_center", "collection": collection, "props": props,
                         "center": r["center_m"], "normal": r["normal"], "major": r["major_axis"],
                         "size": r["radius_mm"] * MM_TO_BLENDER}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (pipeline.py --out)")
    ap.add_argument("--meshes-dir", default=None, help="bone meshes + basins (default <run>/meshes)")
    ap.add_argument("--out", default=None, help="output directory (default <run>/discs)")
    ap.add_argument("--levels", nargs="*", default=None, help="only these levels, e.g. disc_L4_L5 L4_L5")
    ap.add_argument("--reach-mm", type=float, default=20.0, help="max distance of a disc voxel to either bone")
    ap.add_argument("--dot-max", type=float, default=-0.3, help="max dot of the unit directions to U and L")
    ap.add_argument("--hu-min", type=float, default=40.0, help="soft-tissue window low end (CSF, fat below)")
    ap.add_argument("--hu-max", type=float, default=250.0, help="soft-tissue window high end (bone above)")
    ap.add_argument("--hu-sigma-mm", type=float, default=0.5, help="Gaussian on HU before the window")
    ap.add_argument("--contact-band-mm", type=float, default=2.5,
                    help="also take soft voxels inside a bone mask within this of the other mask (touching masks)")
    ap.add_argument("--contact-hu-max", type=float, default=350.0,
                    help="HU ceiling for the trough voxels inside touching masks (partial volume)")
    ap.add_argument("--corridor-margin-mm", type=float, default=3.0,
                    help="candidates must lie inside both vertebral bodies' outlines along the axis + this")
    ap.add_argument("--trough-mm", type=float, default=8.0,
                    help="bone must be found within this distance above and below a disc voxel")
    ap.add_argument("--trough-depth-hu", type=float, default=100.0,
                    help="that bone must be at least this much brighter than the disc voxel")
    ap.add_argument("--link-mm", type=float, default=1.0,
                    help="disc pieces closer than this count as one component (thin discs break up)")
    ap.add_argument("--min-disc-mm3", type=float, default=50.0, help="smaller discs count as not found")
    ap.add_argument("--endplate-mm", type=float, default=4.0,
                    help="endplate voxels: vertebral-body bone within this of the raw disc")
    ap.add_argument("--raster-mm", type=float, default=0.25, help="raster step for the footprint area")
    ap.add_argument("--body-open-mm", type=float, default=5.0,
                    help="opening radius that strips the vertebral arch off the body for the footprint")
    ap.add_argument("--body-regrow-mm", type=float, default=4.0,
                    help="the opened body is regrown this far back into the bone mask")
    ap.add_argument("--facing-cos", type=float, default=0.7,
                    help="endplate voxels face the other bone along the normal (|cos| >= this)")
    ap.add_argument("--shrink-mm", type=float, default=1.0, help="inward offset of the endplate hull")
    ap.add_argument("--column-mm", type=float, default=1.0, help="in-plane column size for the heights")
    ap.add_argument("--inner-mm", type=float, default=1.5, help="heights use columns this far inside the edge")
    ap.add_argument("--contact-mm", type=float, default=3.0, help="endplate slab: bone within this of the disc")
    ap.add_argument("--mesh-sigma-mm", type=float, default=0.6, help="Gaussian on the mask before marching cubes")
    ap.add_argument("--smooth-iters", type=int, default=10, help="Taubin smoothing iterations (0 = off)")
    ap.add_argument("--blender", action="store_true", help="push discs + centre empties into the running Blender")
    ap.add_argument("--blender-only", action="store_true", help="skip the extraction, push the JSONs in --out")
    ap.add_argument("--collection", default="discs", help="Blender collection")
    args = ap.parse_args(argv)

    run = Path(args.run)
    meshes_dir = Path(args.meshes_dir) if args.meshes_dir else run / "meshes"
    out_dir = Path(args.out) if args.out else run / "discs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = None
    if args.levels:
        wanted = {lv if lv.startswith("disc_") else "disc_" + lv for lv in args.levels}

    t_all = time.time()
    results = []
    if args.blender_only:
        for p in sorted(out_dir.glob("disc_*.json")):
            r = json.loads(p.read_text())
            if wanted is None or r["level"] in wanted:
                results.append(r)
        order = {level_name(a, b): i for i, (a, b) in enumerate(zip(VERTEBRAE, VERTEBRAE[1:]))}
        results.sort(key=lambda r: order.get(r["level"], 99))
    else:
        for upper, lower in zip(VERTEBRAE, VERTEBRAE[1:]):
            name = level_name(upper, lower)
            if wanted is not None and name not in wanted:
                continue
            if not (available(meshes_dir, run / "fused", upper, True) and available(meshes_dir, run / "fused", lower, False)):
                continue
            t0 = time.time()
            print(f"{name}: {upper} over {lower}")
            try:
                info = extract_level(upper, lower, run, meshes_dir, out_dir, args)
                results.append(info)
                print(f"  height {info['height_mean_mm']:.1f} mm, area {info['area_mm2']:.0f} mm2, "
                      f"wedge {info['wedge_deg']:.1f} deg ({time.time() - t0:.0f}s)")
            except (ValueError, RuntimeError, MemoryError) as exc:
                print(f"  !! {name} failed: {exc}")
                results.append({"level": name, "upper_bone": upper, "lower_bone": lower, "error": str(exc)})
                for ext in (".json", ".ply", ".stl"):
                    (out_dir / f"{name}{ext}").unlink(missing_ok=True)
        summary = {"run": str(run), "meshes_dir": str(meshes_dir), "levels": results,
                   "nucleus_radius_ratio": NUCLEUS_RADIUS_RATIO,
                   "coords": "LPS mm; *_m = Blender metres = LPS mm * 0.001, scanner origin kept"}
        (out_dir / "discs.json").write_text(json.dumps(summary, indent=1))
    print_table(results)
    print(f"total {time.time() - t_all:.0f}s, outputs in {out_dir}")
    if args.blender:
        push(results, args)


if __name__ == "__main__":
    main()
