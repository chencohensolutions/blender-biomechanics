"""Signed distance field grids for rigid bone meshes (bpy-free).

A field stores, on a regular grid in the mesh's local frame, the signed distance
to the mesh surface: negative inside the bone, positive outside. Because bones are
rigid, a field is computed once per mesh and reused for every pose; a query
transforms points into the bone's local frame and interpolates trilinearly.

Unsigned distances come from the caller (a BVH nearest-surface query in Blender).
The sign is recovered by flood-filling the exterior from the grid boundary through
cells farther than a fraction of a voxel from the surface, which seals the small
holes in segmented bone meshes, and from the surface normal only inside that band.
"""

import numpy as np


DEFAULT_VOXEL_MM = 1.0
DEFAULT_PADDING_MM = 8.0
MAX_CELLS = 2_500_000
# Cells closer than this many voxels to the surface take their sign from the normal.
BAND_VOXELS = 0.75


def grid_spec(bounds_min, bounds_max, voxel, padding, max_cells=MAX_CELLS):
    """Return (origin, voxel, shape) covering the padded bounds with at most max_cells."""
    bounds_min = np.asarray(bounds_min, dtype=np.float64) - padding
    bounds_max = np.asarray(bounds_max, dtype=np.float64) + padding
    extent = bounds_max - bounds_min
    voxel = float(voxel)
    while True:
        shape = np.maximum(np.ceil(extent / voxel).astype(np.int64) + 1, 2)
        if int(np.prod(shape)) <= max_cells:
            break
        voxel *= 1.1
    return bounds_min, voxel, tuple(int(value) for value in shape)


def grid_points(origin, voxel, shape):
    """All cell centers as an (N, 3) array in x-major (C) order matching values.ravel()."""
    axes = [origin[axis] + voxel * np.arange(shape[axis]) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _dilate(mask):
    grown = mask.copy()
    for axis in range(3):
        # np.roll wraps around; wrapped cells are grid-boundary cells, which only
        # ever grow the exterior, so the wrap is harmless.
        grown |= np.roll(mask, 1, axis=axis) | np.roll(mask, -1, axis=axis)
    return grown


def sign_distances(unsigned, normal_dot, voxel, band_voxels=BAND_VOXELS, seal_voxels=1):
    """Apply a sign to an unsigned distance grid.

    unsigned: (nx, ny, nz) distances; normal_dot: same shape, dot(point - nearest,
    normal) from the nearest surface point. Cells in the surface band use the
    normal's sign. The exterior is flood-filled from the grid boundary through cells
    farther than band + seal voxels from the surface, then grown back over the
    sealing ring, so mesh holes narrower than about 2 * seal voxels stay closed.
    """
    band = unsigned <= band_voxels * voxel
    core = unsigned > (band_voxels + seal_voxels) * voxel
    outside = np.zeros(unsigned.shape, dtype=bool)
    for axis in range(3):
        index = [slice(None)] * 3
        for edge in (0, -1):
            index[axis] = edge
            outside[tuple(index)] = core[tuple(index)]
    while True:
        grown = _dilate(outside) & core
        if np.array_equal(grown, outside):
            break
        outside = grown
    ring = ~band & ~core
    for _step in range(seal_voxels + 1):
        outside |= _dilate(outside) & ring
    sign = np.where(outside, 1.0, -1.0)
    sign = np.where(band, np.where(normal_dot >= 0.0, 1.0, -1.0), sign)
    return (unsigned * sign).astype(np.float32)


class DistanceField:
    """Trilinearly interpolated signed distance grid in a mesh's local frame."""

    def __init__(self, origin, voxel, values):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel = float(voxel)
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = np.array(self.values.shape, dtype=np.int64)
        self.extent_max = self.origin + self.voxel * (self.shape - 1)
        self._flat = self.values.ravel()
        self._stride = np.array([self.shape[1] * self.shape[2], self.shape[2], 1], dtype=np.int64)
        self._upper = self.shape - 2

    def sample(self, points):
        """Signed distances at local points (N, 3); outside the grid the distance grows."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        coords = (points - self.origin) * (1.0 / self.voxel)
        clipped = np.clip(coords, 0.0, self._upper + 1.0)
        outside = coords - clipped
        beyond = np.sqrt(np.einsum("ij,ij->i", outside, outside)) * self.voxel
        base = np.minimum(clipped.astype(np.int64), self._upper)
        frac = clipped - base
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        flat = base @ self._stride
        sx, sy, sz = self._stride
        take = self._flat.take
        c00 = take(flat) * (1 - fx) + take(flat + sx) * fx
        c01 = take(flat + sz) * (1 - fx) + take(flat + sx + sz) * fx
        c10 = take(flat + sy) * (1 - fx) + take(flat + sx + sy) * fx
        c11 = take(flat + sy + sz) * (1 - fx) + take(flat + sx + sy + sz) * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz + beyond

    def sample_with_gradient(self, points):
        """Return (distances, unit gradients) using central differences of one voxel."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        count = len(points)
        h = self.voxel
        offsets = np.array([[0, 0, 0], [h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]])
        # One batched query for the center and the six stencil points.
        stacked = (points[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        values = self.sample(stacked).reshape(7, count)
        distances = values[0]
        gradient = np.stack(((values[1] - values[2]), (values[3] - values[4]), (values[5] - values[6])), axis=1) / (2.0 * h)
        norms = np.linalg.norm(gradient, axis=1, keepdims=True)
        gradient = np.where(norms > 1e-9, gradient / np.maximum(norms, 1e-9), 0.0)
        return distances, gradient

    def to_npz_dict(self):
        return {"origin": self.origin, "voxel": np.array(self.voxel), "values": self.values}

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_npz_dict(cls, data):
        return cls(data["origin"], float(data["voxel"]), data["values"])


def box_sdf(points, half_extent):
    """Exact signed distance of points to an axis-aligned box at the origin (test helper)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = np.abs(points) - np.asarray(half_extent, dtype=np.float64)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside
