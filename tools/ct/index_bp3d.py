#!/usr/bin/env python
"""Index the BodyParts3D 4.0 (IS-A tree, 99 % polygon reduction) skeleton as an alternative
atlas for register_atlas.py: one PLY per bone-like part (metres, BodyParts3D frame, no
reorientation) and atlas_bones.json in the Z-Anatomy atlas schema.

  python -I tools/ct/index_bp3d.py ATLAS_DIR [--download DIR] [--obj DIR]

  ATLAS_DIR/download/isa_parts_list_e.txt    concept id \\t representation id \\t en   (header line)
  ATLAS_DIR/download/isa_element_parts.txt   concept id \\t name \\t element file id (header line):
                                             every concept (element or compound) -> its FJ OBJ ids
  ATLAS_DIR/obj/**/FJ*.obj                   element meshes (unzipped isa_BP3D_4.0_obj_99.zip)

Outputs
  ATLAS_DIR/atlas_bones.json   [{name, vertices, bbox_min, bbox_max, centroid, fma, elements}]
                               metres; vertices = vertex count, centroid = vertex mean (as in the
                               Z-Anatomy atlas_bones.json)
  ATLAS_DIR/meshes/<name>.ply  metres, + index.json {name: file} (register_atlas.py --atlas-meshes)
  stdout                       units decision (sacrum bbox), frame (up / anterior / left axes)

Part names are the exact BodyParts3D English names (lower case, e.g. "fifth lumbar vertebra",
"right hip bone"); BodyParts3D left/right is the patient's side.  BodyParts3D has no "sternum"
or "skull" concept, so two compounds are synthesised from elements: "sternum" (manubrium, body
of sternum, xiphoid process) and "skull" (SKULL_PARTS, no teeth or hyoid, as the Z-Anatomy
composite).  BodyParts3D has no coccyx.

The downloaded data is untrusted: this script only parses the text tables and the OBJ files.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

BONE_RE = re.compile(r"\b(vertebra|rib|sacrum|coccyx|hip bone|ilium|ischium|pubis|femur|sternum|manubrium|"
                     r"xiphoid|clavicle|scapula|humerus|skull|cranium|mandible|costal cartilage|atlas|axis)\b")
SKULL_PARTS = ["frontal bone", "occipital bone", "sphenoid bone", "ethmoid", "vomer", "mandible"] + [
    f"{side} {b}" for b in ("temporal bone", "parietal bone", "maxilla", "zygomatic bone", "nasal bone",
                            "lacrimal bone", "palatine bone", "inferior nasal concha")
    for side in ("left", "right")]
SYNTH = {"sternum": ["manubrium", "body of sternum", "xiphoid process"], "skull": SKULL_PARTS}
EXCLUDE_RE = re.compile(r"^zone of |\bdisk\b|intervertebral")   # "zone of sternum" = the manubrium element
SACRUM_MM = (90.0, 160.0)                  # plausible sacrum height range in mm


def read_table(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    bad = [r for r in rows if len(r) != 3]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows without 3 tab-separated columns, e.g. {bad[0]}")
    return rows


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas_dir")
    ap.add_argument("--download", default=None, help="text tables (default ATLAS_DIR/download)")
    ap.add_argument("--obj", default=None, help="unzipped OBJ tree (default ATLAS_DIR/obj)")
    args = ap.parse_args(argv)
    t0 = time.time()
    root = Path(args.atlas_dir)
    dl = Path(args.download) if args.download else root / "download"
    obj_dir = Path(args.obj) if args.obj else root / "obj"

    parts = read_table(dl / "isa_parts_list_e.txt")            # concept, representation, en
    name_of = {c: en.strip() for c, _rep, en in parts}
    elements = {}
    for c, _name, fj in read_table(dl / "isa_element_parts.txt"):
        elements.setdefault(c.strip(), []).append(fj.strip())
    by_name = {}
    for c, n in name_of.items():
        by_name.setdefault(n, []).append(c)
    dup = {n: cs for n, cs in by_name.items() if len(cs) > 1}
    if dup:
        print(f"WARNING: {len(dup)} names with several concept ids (first used): {list(dup.items())[:5]}")
    concept = {n: cs[0] for n, cs in by_name.items()}

    obj_files = {p.stem: p for p in obj_dir.rglob("*.obj")}
    print(f"parts list: {len(name_of)} concepts; element table: {len(elements)} concepts; "
          f"OBJ files: {len(obj_files)}")

    # selection by name (keywords + skull bones, minus disks/zones); BodyParts3D's "bone organ"
    # compound is incomplete (it lacks the sternum), so it is only reported, not used as a filter
    bone_organ = set(elements.get(concept.get("bone organ"), [])) | set(elements.get(concept.get("costal cartilage"), []))
    selected, rejected, outside = {}, [], []
    for n, c in sorted(concept.items()):
        if not (BONE_RE.search(n) or n in SKULL_PARTS):
            continue
        fj = elements.get(c, [])
        if EXCLUDE_RE.search(n) or not fj:
            rejected.append(n)
            continue
        selected[n] = {"fma": c, "elements": sorted(set(fj))}
        if not set(fj) <= bone_organ:
            outside.append(n)
    for n, members in SYNTH.items():
        missing = [m for m in members if m not in selected]
        if missing:
            print(f"WARNING: synthesised '{n}' lacks {missing}")
        fj = sorted({e for m in members if m in selected for e in selected[m]["elements"]})
        selected[n] = {"fma": None, "elements": fj, "synthesised_from": [m for m in members if m in selected]}
    print(f"bone-like parts: {len(selected)} (incl. synthesised {list(SYNTH)}); "
          f"keyword matches excluded (disks, zones): {len(rejected)}")
    print(f"  selected parts with elements outside BodyParts3D 'bone organ' (kept): {outside}")

    cache = {}

    def element(fj):
        if fj not in cache:
            if fj not in obj_files:
                cache[fj] = None
            else:
                m = trimesh.load(obj_files[fj], force="mesh", process=False, skip_materials=True)
                cache[fj] = m if len(m.vertices) else None
        return cache[fj]

    # units from the sacrum
    sac = element(selected["sacrum"]["elements"][0])
    ext = sac.extents
    if SACRUM_MM[0] <= ext.max() <= SACRUM_MM[1]:
        unit, to_m = "mm", 1e-3
    elif SACRUM_MM[0] / 1000 <= ext.max() <= SACRUM_MM[1] / 1000:
        unit, to_m = "m", 1.0
    else:
        raise SystemExit(f"sacrum bbox extents {ext} fit neither mm nor m")
    print(f"sacrum raw bbox min {np.round(sac.bounds[0], 2).tolist()} max {np.round(sac.bounds[1], 2).tolist()} "
          f"extents {np.round(ext, 2).tolist()} -> units {unit}")

    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    entries, index, missing_parts = [], {}, {}
    for n, info in selected.items():
        ms = [element(fj) for fj in info["elements"]]
        miss = [fj for fj, m in zip(info["elements"], ms) if m is None]
        if miss:
            missing_parts[n] = miss
        ms = [m for m in ms if m is not None]
        if not ms:
            continue
        mesh = ms[0].copy() if len(ms) == 1 else trimesh.util.concatenate(ms)
        mesh.apply_scale(to_m)
        f = f"{sanitize(n)}.ply"
        mesh.export(mesh_dir / f)
        index[n] = f
        v = np.asarray(mesh.vertices)
        entries.append({"name": n, "vertices": int(len(v)), "bbox_min": v.min(0).tolist(),
                        "bbox_max": v.max(0).tolist(), "centroid": v.mean(0).tolist(),
                        "fma": info["fma"], "elements": info["elements"],
                        **({"synthesised_from": info["synthesised_from"]} if "synthesised_from" in info else {})})
    if missing_parts:
        print(f"WARNING: {len(missing_parts)} parts with missing OBJ elements: {list(missing_parts.items())[:8]}")
    (mesh_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    (root / "atlas_bones.json").write_text(json.dumps(entries, indent=1))

    # frame
    E = {e["name"]: e for e in entries}
    c = lambda n: np.asarray(E[n]["centroid"])  # noqa: E731
    axes = "xyz"
    up_vec = c("skull") - c("sacrum")
    vert = [n for n in E if re.fullmatch(r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
                                         r"eleventh|twelfth) thoracic vertebra", n)]
    ant_vec = c("sternum") - np.mean([c(n) for n in vert], axis=0)
    left_vec = c("left hip bone") - c("right hip bone")

    def dom(v):
        i = int(np.argmax(np.abs(v)))
        return f"{'+' if v[i] > 0 else '-'}{axes[i]}"
    print(f"frame: up {dom(up_vec)} (sacrum->skull {np.round(up_vec * 1000, 1).tolist()} mm), "
          f"anterior {dom(ant_vec)} (T-vertebrae->sternum {np.round(ant_vec * 1000, 1).tolist()} mm), "
          f"patient left {dom(left_vec)} (right->left hip {np.round(left_vec * 1000, 1).tolist()} mm)")
    print(f"skull top {np.round(np.asarray(E['skull']['bbox_max']) * 1000, 1).tolist()} mm; "
          f"sacrum size {np.round((np.asarray(E['sacrum']['bbox_max']) - E['sacrum']['bbox_min']) * 1000, 1).tolist()} mm; "
          f"right femur size {np.round((np.asarray(E['right femur']['bbox_max']) - E['right femur']['bbox_min']) * 1000, 1).tolist()} mm")
    print(f"wrote {len(entries)} parts: {root / 'atlas_bones.json'}, {mesh_dir}/ ({len(cache)} OBJ elements loaded) "
          f"in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
