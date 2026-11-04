#!/usr/bin/env python
"""Export the CT scene open in Blender as glTF (GLB), through the BlenderMCP socket.

Every mesh object of the view layer is exported at full resolution with its viewport (object)
colour baked into a Principled material, so a glTF viewer shows the same colouring as Blender's
Workbench renders. The meshes are already in Blender metres at the scanner position, so every
file shares one frame and the files can be loaded side by side.

  python tools/ct/export_gltf.py OUT_DIR                       # ct2021_all.glb (uncompressed) + ct2021_all_draco.glb + 3 group files
  python tools/ct/export_gltf.py OUT_DIR --groups skeleton     # one group only
  python tools/ct/export_gltf.py OUT_DIR --no-single           # the group files only
  python tools/ct/export_gltf.py OUT_DIR --collections heart   # any collections into one file

Collections excluded from the view layer are included for the export and excluded again after it
(the scene keeps `thorax` excluded because the organ run's one-body `heart` is superseded by the
12-part heart; that object is skipped). Hidden objects are unhidden for the export and hidden again.
Draco compression is lossless in topology and keeps positions to 16 bits of the bounding box
(about 0.01 mm on this scene); viewers need KHR_draco_mesh_compression (Blender, three.js, Babylon).
The group files stay under the 30 MiB chat upload limit; the single uncompressed file is ~370 MiB.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402

GROUPS = {
    "skeleton": ("bones", "discs", "cartilage"),
    "muscles": ("muscles",),
    "organs": ("abdomen", "thorax", "vessels", "nervous", "heart"),
}
SUPERSEDED = {"thorax": ("heart",)}  # collection -> base names replaced by a later, finer stage

CODE = r'''
import bpy, os, time
OUT, PREFIX, JOBS, SUPERSEDED, QUANT, LEVEL = %(out)r, %(prefix)r, %(jobs)r, %(superseded)r, %(quant)r, %(level)r
vl = bpy.context.view_layer
def walk(l, f):
    for ch in l.children:
        f(ch); walk(ch, f)
excluded = []
def note(ch):
    if ch.exclude:
        excluded.append(ch.name); ch.exclude = False
walk(vl.layer_collection, note)
vl.update()
def base(o): return o.name.rsplit(".", 1)[0]
def coll_of(o): return o.users_collection[0].name if o.users_collection else ""
skip = {o.name for c, bases in SUPERSEDED.items() if c in bpy.data.collections
        for o in bpy.data.collections[c].objects if base(o) in bases}
meshes = [o for o in vl.objects if o.type == "MESH" and not o.hide_viewport and o.name not in skip]
hidden = [o for o in meshes if o.hide_get()]
for o in hidden: o.hide_set(False)
print("included excluded collections:", excluded, "| unhidden:", [o.name for o in hidden], "| skipped superseded:", sorted(skip))
for o in meshes:
    name = "glb_" + o.name
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    col = tuple(o.color)
    if bsdf: bsdf.inputs["Base Color"].default_value = col
    mat.diffuse_color = col
    if o.data.materials: o.data.materials[0] = mat
    else: o.data.materials.append(mat)
def export(path, objs, draco):
    if not objs:
        print(os.path.basename(path), "nothing to export"); return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    vl.objects.active = objs[0]
    kw = dict(export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=LEVEL,
              export_draco_position_quantization=QUANT) if draco else {}
    t = time.time()
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_materials="EXPORT", export_yup=True, **kw)
    print(os.path.basename(path), len(objs), "meshes", sum(len(o.data.polygons) for o in objs), "faces",
          round(os.path.getsize(path) / 2**20, 1), "MiB", round(time.time() - t), "s")
for name, colls, draco in JOBS:
    objs = meshes if colls is None else [o for o in meshes if coll_of(o) in colls]
    export(os.path.join(OUT, PREFIX + name + ".glb"), objs, draco)
for o in hidden: o.hide_set(True)
walk(vl.layer_collection, lambda ch: setattr(ch, "exclude", True) if ch.name in excluded else None)
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output folder (created)")
    ap.add_argument("--prefix", default="ct2021_", help="file name prefix (default ct2021_)")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS), help="group files to write (default all three)")
    ap.add_argument("--collections", nargs="*", help="instead of the groups: these collections into one Draco file named after the first")
    ap.add_argument("--no-single", action="store_true", help="skip the two whole-scene files")
    ap.add_argument("--no-draco", action="store_true", help="write the group files uncompressed too")
    ap.add_argument("--quantization", type=int, default=16, help="Draco position bits (default 16)")
    ap.add_argument("--level", type=int, default=6, help="Draco compression level 0-10 (default 6)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    draco = not args.no_draco
    jobs = []
    if args.collections:
        jobs.append((args.collections[0] + ("_draco" if draco else ""), tuple(args.collections), draco))
    else:
        if not args.no_single:
            jobs += [("all", None, False), ("all_draco", None, True)]
        jobs += [(g + ("_draco" if draco else ""), GROUPS[g], draco) for g in args.groups]
    print(blender_bridge.run_code(CODE % {"out": str(out), "prefix": args.prefix, "jobs": jobs, "superseded": SUPERSEDED,
                                          "quant": args.quantization, "level": args.level}, timeout=args.timeout))


if __name__ == "__main__":
    main()
