#!/usr/bin/env python
"""Export every real bone of the Z-Anatomy skeleton FBX as one PLY per bone (headless Blender).

  blender -b --python tools/ct/export_atlas_meshes.py -- --fbx SkeletalSystem100.fbx --out DIR

The FBX is only loadable by Blender, so this step runs inside Blender; register_atlas.py
then reads the PLYs with trimesh (--atlas-meshes).

Selection
  MESH objects of the import whose name has no Z-Anatomy label/helper suffix
  (BONE_SUFFIX_RE: .i .j .el .er .ol .or .t .e1l ...) and no Blender duplicate suffix
  (.001).  Same filter as register_atlas.py --blender.

Outputs
  DIR/<name>.ply   binary little-endian PLY, evaluated mesh (modifiers applied) in world
                   coordinates (object transform applied), metres, Blender axes as imported
                   (+x = patient left, +y = posterior, +z = up); faces triangulated, vertex
                   order = mesh.vertices (vertex count matches the Blender mesh)
  DIR/index.json   {atlas object name: file name}
  File names: characters other than letters, digits, '.', '-', '_' become '_'.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np

BONE_SUFFIX_RE = re.compile(r"\.(i|j|el|er|ol|or|t|[eo]\d*[lr])$")
NUMERIC_SUFFIX_RE = re.compile(r"\.\d{3,}$")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def write_ply(path, verts, tris):
    verts = np.ascontiguousarray(verts, dtype="<f4")
    faces = np.empty(len(tris), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    faces["n"] = 3
    faces["i"] = tris
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(verts.tobytes())
        fh.write(faces.tobytes())


def mesh_arrays(obj, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        co = np.empty(nv * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        tri = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        M = np.array(ev.matrix_world, dtype=np.float64)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        return co, tri.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fbx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.fbx)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    keep = sorted((o for o in bpy.data.objects if o.type == "MESH"
                   and not BONE_SUFFIX_RE.search(o.name) and not NUMERIC_SUFFIX_RE.search(o.name)),
                  key=lambda o: o.name)
    index, used = {}, {}
    for obj in keep:
        fname = sanitize(obj.name) + ".ply"
        if fname in used:
            raise SystemExit(f"file name clash: {obj.name!r} and {used[fname]!r} -> {fname}")
        used[fname] = obj.name
        verts, tris = mesh_arrays(obj, depsgraph)
        if len(verts) == 0 or len(tris) == 0:
            print(f"skip empty mesh {obj.name!r}")
            continue
        write_ply(out / fname, verts, tris)
        index[obj.name] = fname
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    n_mesh = sum(o.type == "MESH" for o in bpy.data.objects)
    print(f"exported {len(index)} bones of {n_mesh} mesh objects to {out}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
