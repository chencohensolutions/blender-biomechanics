#!/usr/bin/env python
"""Push CT-derived meshes into a running Blender through the BlenderMCP socket (localhost:9876).

The meshes written by extract_bone.py are already in Blender world coordinates
(LPS millimetres * 0.001 -> metres, scanner origin preserved), so every bone shares one
transform and inter-bone gaps are exactly those of the scan.

  python tools/ct/blender_bridge.py new-scene "/path/ct_bones.blend"
  python tools/ct/blender_bridge.py import sacrum.ply --name sacrum --collection bones [--decimate 0.3]
  python tools/ct/blender_bridge.py exec "print(len(bpy.data.objects))"
"""
import argparse
import json
import socket
import sys

HOST, PORT = "localhost", 9876


def run_code(code, timeout=600.0):
    with socket.create_connection((HOST, PORT), timeout=timeout) as sock:
        sock.sendall(json.dumps({"type": "execute_code", "params": {"code": code}}).encode())
        chunks = []
        while True:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                reply = json.loads(b"".join(chunks).decode())
                break
            except json.JSONDecodeError:
                continue
    if reply.get("status") != "success":
        raise RuntimeError(reply.get("message") or reply)
    return reply.get("result", {}).get("result") or reply.get("result")


NEW_SCENE = r'''
import bpy
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for block in list(coll):
        if block.users == 0:
            coll.remove(block)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
scene.unit_settings.length_unit = "MILLIMETERS"
for name in ("bones", "muscles", "atlas"):
    if name not in bpy.data.collections:
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.clip_end = 100.0
                space.shading.type = "SOLID"
bpy.ops.wm.save_as_mainfile(filepath=%(path)r)
print("saved", bpy.data.filepath, "collections", [c.name for c in bpy.data.collections])
'''

IMPORT_MESH = r'''
import bpy, os
path, name, col_name, decimate, props = %(path)r, %(name)r, %(collection)r, %(decimate)r, %(props)r
before = set(bpy.data.objects)
if path.lower().endswith(".ply"):
    bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z", global_scale=1.0)
else:
    bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z", global_scale=1.0)
new = [o for o in bpy.data.objects if o not in before]
obj = new[0]
if name in bpy.data.objects and bpy.data.objects[name] is not obj:
    old = bpy.data.objects[name]
    bpy.data.objects.remove(old, do_unlink=True)
obj.name = name
obj.data.name = name
for k, v in (props or {}).items():
    obj[k] = v
col = bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
if col.name not in [c.name for c in bpy.context.scene.collection.children]:
    bpy.context.scene.collection.children.link(col)
for c in list(obj.users_collection):
    c.objects.unlink(obj)
col.objects.link(obj)
if decimate and 0 < decimate < 1:
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.ratio = decimate
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
obj.data.shade_smooth() if hasattr(obj.data, "shade_smooth") else None
bb = [obj.matrix_world @ __import__("mathutils").Vector(c) for c in obj.bound_box]
lo = [min(v[i] for v in bb) for i in range(3)]; hi = [max(v[i] for v in bb) for i in range(3)]
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        with bpy.context.temp_override(area=area, region=[r for r in area.regions if r.type == "WINDOW"][0]):
            bpy.ops.view3d.view_all(center=False)
if bpy.data.filepath:
    bpy.ops.wm.save_mainfile()
print("imported", obj.name, "verts", len(obj.data.vertices), "faces", len(obj.data.polygons),
      "bbox_m", [round(v, 4) for v in lo], [round(v, 4) for v in hi])
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new-scene"); p.add_argument("path")
    p = sub.add_parser("import"); p.add_argument("mesh"); p.add_argument("--name", required=True)
    p.add_argument("--collection", default="bones"); p.add_argument("--decimate", type=float, default=0.0)
    p = sub.add_parser("exec"); p.add_argument("code")
    args = ap.parse_args(argv)
    if args.cmd == "new-scene":
        print(run_code(NEW_SCENE % {"path": args.path}))
    elif args.cmd == "import":
        print(run_code(IMPORT_MESH % {"path": args.mesh, "name": args.name,
                                      "collection": args.collection, "decimate": args.decimate, "props": {}}))
    else:
        print(run_code(args.code))


if __name__ == "__main__":
    main()
