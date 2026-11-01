#!/usr/bin/env python
"""DICOM folder -> three-plane fusion -> TotalSegmentator guides -> per-bone meshes -> Blender.

One command reproduces the whole bone extraction for a scan, so a second scan (or a
second parameter set) can be run the same way and compared with compare_runs.py.

  python tools/ct/pipeline.py --dicom "/Users/me/Downloads/CT 2021" --match Delayed --match Lung \
      --bones all --out ".../spine/segmentation/runs/ct2021" \
      --blender append --collection bones_2021

--match may be given several times: each selects one acquisition group, i.e. every series
whose description contains one of its '|'-separated substrings (all planes, several
reconstructions per plane allowed; they must be co-registered, i.e. one acquisition).  Groups are tried in the given order and each bone
is taken from the first group whose axial volume contains the bone's guide label entirely,
so list the highest-resolution acquisition first; align_groups.py later brings the other
groups into the first group's frame through the vertebrae they share.  Bones from different acquisitions can
disagree by a few mm (breathing, table move); each Blender object records its group in the
custom property "ct_group".  --bones all extracts every bone class that has a label.

Steps (each cached in the run directory; delete a sub-folder to redo that step):
  1. series/    discover every DICOM series, pick the axial / coronal / sagittal
                reconstructions of each acquisition group (--match substring or --series
                numbers, most slices per plane wins), convert to NRRD + NIfTI
  2. totalseg/  TotalSegmentator on each group's axial series for the requested bones and
                their neighbours (coarse 1.5 mm anatomical labels used only as guides)
  3. fused/     fuse_planes.py over each bone's neighbourhood (finest in-plane spacing)
  4. meshes/    extract_bone.py: watershed split at the joints, sub-voxel iso-surface,
                PLY + STL in Blender metres with the scanner's LPS axes
  5. Blender    optional: push every mesh into the running Blender (blender_bridge.py)
  run.json      manifest with the chosen series, parameters and outputs

Bone names are TotalSegmentator class names (sacrum, vertebrae_L5, hip_left, femur_right,
rib_left_7 ...); L1-L5 / T1-T12 / C1-C7 are accepted as shorthand.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_bridge  # noqa: E402
import extract_bone  # noqa: E402
import fuse_planes  # noqa: E402

RIBS = [f"rib_{side}_{i}" for side in ("left", "right") for i in range(1, 13)]
ALL_BONES = None  # filled below
VERTEBRAE = ([f"vertebrae_C{i}" for i in range(1, 8)] + [f"vertebrae_T{i}" for i in range(1, 13)]
             + [f"vertebrae_L{i}" for i in range(1, 6)] + ["sacrum"])

CARTILAGES = [f"costal_cartilage_{side}_{i}" for side in ("left", "right") for i in range(1, 11)]
ALL_BONES = (VERTEBRAE + ["hip_left", "hip_right", "femur_left", "femur_right", "sternum", "clavicula_left",
                          "clavicula_right", "scapula_left", "scapula_right", "humerus_left", "humerus_right",
                          "skull", "costal_cartilages"] + RIBS)


def canonical(bone):
    b = bone.strip()
    if len(b) in (2, 3) and b[0].upper() in "CTL" and b[1:].isdigit():
        return f"vertebrae_{b[0].upper()}{int(b[1:])}"
    b = b.replace(" ", "_")
    by_lower = {name.lower(): name for name in ALL_BONES}
    return by_lower.get(b.lower(), b)  # keep TotalSegmentator's own casing (vertebrae_L1)


def neighbours(bone):
    """Bones that may touch `bone` through a joint and must be split from it."""
    if bone == "sacrum":
        return ["vertebrae_L5", "hip_left", "hip_right"]
    if bone.startswith("hip_"):
        side = bone.split("_")[1]
        return ["sacrum", f"femur_{side}"]
    if bone.startswith("femur_"):
        side = bone.split("_")[1]
        return [f"hip_{side}"]
    if bone.startswith("vertebrae_"):
        i = VERTEBRAE.index(bone)
        out = [v for v in (VERTEBRAE[i - 1] if i > 0 else None, VERTEBRAE[i + 1]) if v]
        if bone[10] == "T":  # rib n articulates with T(n-1) and T(n)
            n = int(bone[11:])
            out += [f"rib_{side}_{k}" for side in ("left", "right") for k in (n, n + 1) if 1 <= k <= 12]
        if bone == "vertebrae_C1":
            out.append("skull")
        return out
    if bone.startswith("rib_"):
        side, n = bone.split("_")[1], int(bone.split("_")[2])
        out = [f"vertebrae_T{k}" for k in (n - 1, n) if 1 <= k <= 12] + ["costal_cartilages",
                                                                          f"costal_cartilage_{side}_{n}"]
        if n == 1:
            out.append("clavicula_" + side)
        return out
    if bone == "sternum":
        return ["costal_cartilages", "clavicula_left", "clavicula_right"] + [c for c in CARTILAGES if int(c.split("_")[-1]) <= 7]
    if bone == "costal_cartilages":
        return ["sternum"] + RIBS
    if bone.startswith("costal_cartilage_"):
        side, n = bone.split("_")[2], int(bone.split("_")[3])
        return [f"rib_{side}_{n}", "sternum"] + [f"costal_cartilage_{side}_{k}" for k in (n - 1, n + 1) if 1 <= k <= 10]
    if bone.startswith("clavicula_"):
        side = bone.split("_")[1]
        return ["sternum", f"scapula_{side}", f"rib_{side}_1"]
    if bone.startswith("scapula_"):
        side = bone.split("_")[1]
        return [f"humerus_{side}", f"clavicula_{side}"] + [f"rib_{side}_{k}" for k in range(2, 8)]
    if bone.startswith("humerus_"):
        side = bone.split("_")[1]
        return [f"scapula_{side}"]
    if bone == "skull":
        return ["vertebrae_C1"]
    return []


def plane_from_iop(iop):
    if iop is None or len(iop) != 6:
        return "unknown"
    normal = np.cross(iop[:3], iop[3:])
    return {0: "sagittal", 1: "coronal", 2: "axial"}[int(np.argmax(np.abs(normal)))]


def discover_series(root, min_files=10):
    reader = sitk.ImageSeriesReader()
    found = []
    for dirpath, _dirs, filenames in os.walk(root):
        if not filenames:
            continue
        for sid in reader.GetGDCMSeriesIDs(dirpath):
            files = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if len(files) < min_files:
                continue
            fr = sitk.ImageFileReader()
            fr.SetFileName(files[0])
            fr.LoadPrivateTagsOff()
            fr.ReadImageInformation()

            def tag(key, default=""):
                return fr.GetMetaData(key).strip() if fr.HasMetaDataKey(key) else default

            iop_raw = tag("0020|0037")
            iop = [float(v) for v in iop_raw.split("\\")] if iop_raw else None
            found.append({
                "uid": sid, "number": int(tag("0020|0011", "0") or 0), "description": tag("0008|103e"),
                "plane": plane_from_iop(iop), "n": len(files), "files": files,
                "thickness_mm": tag("0018|0050"), "pixel_spacing_mm": tag("0028|0030"),
                "acquisition": tag("0020|0012"), "modality": tag("0008|0060")})
    found.sort(key=lambda s: s["number"])
    return found


def select_series(series, match=None, numbers=None):
    """All reconstructions of one acquisition group: every series whose description contains one
    of the '|'-separated substrings (or whose number is listed).  Returns {plane: [series...]},
    each plane's list sorted finest slices first."""
    if numbers:
        picked = [s for s in series if s["number"] in numbers]
    else:
        subs = [m.lower() for m in (match or "").split("|") if m]
        picked = [s for s in series if not subs or any(m in s["description"].lower() for m in subs)]
    chosen = {}
    for s in picked:
        if s["plane"] == "unknown":
            continue
        chosen.setdefault(s["plane"], []).append(s)
    for plane in chosen:
        chosen[plane].sort(key=lambda s: (float(s["thickness_mm"] or 99), -s["n"], -s["number"]))
    return chosen


def convert_series(s, out_dir):
    base = out_dir / f"{s['number']:03d}_{s['plane']}"
    nrrd, nii = base.with_suffix(".nrrd"), Path(str(base) + ".nii.gz")
    if not nrrd.exists():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(s["files"])
        img = reader.Execute()
        sitk.WriteImage(img, str(nrrd), useCompression=True)
        if s["plane"] == "axial":
            sitk.WriteImage(img, str(nii), useCompression=True)
        print(f"  wrote {nrrd.name}: size={img.GetSize()} spacing={np.round(img.GetSpacing(), 3).tolist()}")
    return nrrd, nii


def run_totalsegmentator(axial_nii, out_dir, classes, device):
    missing = [c for c in classes if not (out_dir / f"{c}.nii.gz").exists()]
    if not missing:
        print(f"  totalseg: all {len(classes)} labels cached")
        return
    exe = Path(sys.executable).parent / "TotalSegmentator"
    cmd = [str(exe), "-i", str(axial_nii), "-o", str(out_dir), "--roi_subset", *classes, "--device", device]
    print("  " + " ".join(cmd[:5]) + f" ... ({len(classes)} classes, device {device})")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and device != "cpu":
        print(f"  totalseg failed on {device} ({res.stderr.strip().splitlines()[-1] if res.stderr else ''}); retrying on cpu")
        cmd[-1] = "cpu"
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("TotalSegmentator failed:\n" + res.stdout[-2000:] + res.stderr[-2000:])
    print(f"  totalseg done in {time.time() - t0:.0f}s")


def label_path(bone, group, atlas_labels=None):
    """Guide label for `bone` in `group`: the TotalSegmentator label, else the fitted-atlas label."""
    ts = group["totalseg"] / f"{bone}.nii.gz"
    if ts.exists():
        return ts
    if atlas_labels:
        alt = Path(atlas_labels) / f"{bone}.nii.gz"
        if alt.exists():
            return alt
    return None


def label_bounds(path):
    """(lo, hi) LPS bounds of the non-zero voxels, or None if the label is empty."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    if not arr.any():
        return None
    zs, ys, xs = np.where(arr)
    pts = [img.TransformIndexToPhysicalPoint((int(i), int(j), int(k)))
           for i in (xs.min(), xs.max()) for j in (ys.min(), ys.max()) for k in (zs.min(), zs.max())]
    pts = np.array(pts)
    return pts.min(0), pts.max(0), int(arr.sum())


def choose_group(bone, groups, margin=5.0, atlas_labels=None):
    """First group whose axial volume contains the bone label (with margin); else the one with the
    largest label.  Returns (group, bounds) or (None, None)."""
    best = None
    for g in groups:
        path = label_path(bone, g, atlas_labels)
        if path is None:
            continue
        b = label_bounds(path)
        if b is None:
            continue
        lo, hi, count = b
        if np.all(lo - margin >= g["bounds"][0]) and np.all(hi + margin <= g["bounds"][1]):
            return g, b
        if best is None or count > best[1][2]:
            best = (g, b)
    return best if best else (None, None)


def process_bone(bone, group, fused, out, args):
    """Fusion + extraction of one bone (runs in a worker process). Returns (bone, outputs|None, error)."""
    try:
        if fused.exists():
            print(f"  cached {fused.name}")
        else:
            fargs = ["--inputs", *map(str, group["volumes"]), "--roi-from-mask", str(label_path(bone, group, args.atlas_labels)),
                     "--pad", str(args.pad), "--method", args.method, "--out", str(fused)]
            if args.spacing:
                fargs += ["--spacing", str(args.spacing)]
            if args.finest_only:
                fargs += ["--finest-only"]
            fuse_planes.main(fargs)
        guide_files = [str(label_path(c, group, args.atlas_labels)) for c in [bone, *neighbours(bone)]
                       if label_path(c, group, args.atlas_labels) is not None]
        extra = list(args.extract_args)
        if args.atlas_labels:
            extra += ["--atlas-labels", str(args.atlas_labels)]
        if args.denoise_mm == "auto":
            finest = min(max(sitk.ReadImage(str(v)).GetSpacing()) for v in group["volumes"])
            extra += ["--denoise-mm", "0.6" if finest <= 1.5 else "0"]
        else:
            extra += ["--denoise-mm", str(args.denoise_mm)]
        extract_bone.main(["--fused", str(fused), "--target", bone, "--labels", *guide_files,
                           "--out-dir", str(out / "meshes"), "--level", str(args.level),
                           "--smooth-iters", str(args.smooth_iters), "--also-labelmap", *extra])
    except (SystemExit, ValueError, RuntimeError, MemoryError) as exc:
        return bone, None, str(exc)
    return bone, {"group": group["name"], "fused": str(fused), "ply": str(out / "meshes" / f"{bone}.ply"),
                  "stl": str(out / "meshes" / f"{bone}.stl"), "json": str(out / "meshes" / f"{bone}.json")}, None


REMOVE_OTHER_RUNS = r'''
import bpy
col = bpy.data.collections.get(%(collection)r)
removed = []
if col:
    for obj in list(col.objects):
        if obj.get("ct_run") != %(run_tag)r:
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    if bpy.data.filepath:
        bpy.ops.wm.save_mainfile()
print("removed", len(removed), "objects from other runs:", removed)
'''


def push_to_blender(args, items, first=False):
    try:
        if first and args.blender == "new":
            if not args.blend:
                sys.exit("--blender new needs --blend PATH")
            print("  " + str(blender_bridge.run_code(blender_bridge.NEW_SCENE % {"path": args.blend})))
        for bone, files in items:
            collection = "cartilage" if bone.startswith("costal_cartilage_") else args.collection
            print("  " + str(blender_bridge.run_code(blender_bridge.IMPORT_MESH % {
                "path": files["ply"], "name": bone + args.suffix, "collection": collection,
                "decimate": args.decimate, "props": {"ct_group": files["group"], "ct_level_hu": args.level,
                                                    "ct_run": args.run_tag}})).strip())
    except (ConnectionRefusedError, OSError) as exc:
        print(f"!! Blender not reachable on localhost:9876 ({exc}); meshes stay on disk")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", help="folder containing the DICOM files (searched recursively)")
    src.add_argument("--volumes", nargs="+", help="already converted NRRD/NIfTI reconstructions instead of DICOM")
    ap.add_argument("--match", action="append", help="SeriesDescription substring selecting an acquisition group; "
                    "repeat in priority order, e.g. --match Delayed --match Lung")
    ap.add_argument("--series", nargs="+", type=int, help="explicit series numbers to use (axial, coronal, sagittal)")
    ap.add_argument("--bones", nargs="+", required=True,
                    help="bones to extract (TotalSegmentator names, L5/T12 shorthand, or 'all')")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--level", type=float, default=250.0, help="HU iso-value for the bone surface")
    ap.add_argument("--spacing", type=float, default=None, help="fused grid spacing (default: finest input spacing)")
    ap.add_argument("--pad", type=float, default=20.0, help="mm around each bone's guide label for the fusion")
    ap.add_argument("--method", default="weighted", choices=["weighted", "mean", "max"])
    ap.add_argument("--smooth-iters", type=int, default=5, help="Taubin smoothing iterations of the mesh")
    ap.add_argument("--denoise-mm", default="auto",
                    help="Gaussian before thresholding: 'auto' = 0.6 mm for groups with slices <= 1.5 mm "
                         "(sharp kernels), 0 for thick-slice soft groups; or a number for all")
    ap.add_argument("--atlas-labels", default=None,
                    help="fitted-atlas label directory from register_atlas.py (anatomical prior for the split)")
    ap.add_argument("--finest-only", action="store_true",
                    help="fuse only the finest series per plane (sharper, noisier); default fuses all")
    ap.add_argument("--extract-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this flag is passed to extract_bone.py, e.g. "
                         "--extract-args --split hu --split-level 0")
    ap.add_argument("--device", default="mps", help="TotalSegmentator device: mps, cpu or gpu")
    ap.add_argument("--jobs", type=int, default=1, help="bones processed in parallel (each needs ~2-4 GB RAM)")
    ap.add_argument("--run-tag", default=None, help="stored on each Blender object as ct_run (default: <out name>@<start time>)")
    ap.add_argument("--joint-level", type=float, default=650.0,
                    help="consistency pass: voxels below this HU where two bones touch are joint space, not bone "
                         "(cortex is 900-1200 HU in these scans); 0 = off")
    ap.add_argument("--resegment", action="store_true",
                    help="after extraction: overlap report, joint re-cut of all vertebra pairs and of flagged "
                         "pairs (resegment_pairs.py), overlap report again")
    ap.add_argument("--keep-old", action="store_true",
                    help="keep Blender objects from earlier runs in the collection (default: remove them at the end)")
    ap.add_argument("--blender", choices=["none", "append", "new"], default="none",
                    help="push meshes into the running Blender; 'new' first clears it and saves --blend")
    ap.add_argument("--blend", help="path for --blender new")
    ap.add_argument("--collection", default="bones", help="Blender collection for the meshes")
    ap.add_argument("--suffix", default="", help="suffix for Blender object names, e.g. _2021")
    ap.add_argument("--decimate", type=float, default=0.0, help="Blender decimate ratio after import (0 = keep full)")
    args = ap.parse_args(argv)

    t_start = time.time()
    out = Path(args.out)
    for sub in ("series", "totalseg", "fused", "meshes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    want_all = [b.lower() for b in args.bones] == ["all"]
    bones = list(ALL_BONES) if want_all else [canonical(b) for b in args.bones]
    if want_all and args.atlas_labels:
        bones += [c for c in CARTILAGES if (Path(args.atlas_labels) / f"{c}.nii.gz").exists()]
    args.run_tag = args.run_tag or f"{out.name}@{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "run_tag": args.run_tag, "args": vars(args),
                "groups": {}, "bones": {}, "outputs": {}}

    # 1. series -> acquisition groups
    groups = []
    if args.dicom:
        print(f"[1/5] scanning DICOM in {args.dicom}")
        series = discover_series(args.dicom)
        for s in series:
            print(f"  {s['number']:4d} {s['plane']:8s} n={s['n']:4d} thk={s['thickness_mm']:>4s} "
                  f"px={s['pixel_spacing_mm']:<22s} {s['description']}")
        selections = [(m, select_series(series, m, None)) for m in args.match] if args.match else []
        if args.series:
            selections.append(("series_" + "_".join(map(str, args.series)), select_series(series, None, args.series)))
        if not selections:
            sys.exit("give --match (repeatable) and/or --series")
        for name, chosen in selections:
            if "axial" not in chosen:
                print(f"!! group {name!r}: no axial series, skipped")
                continue
            volumes = []
            for plane in ("axial", "coronal", "sagittal"):
                for k, s in enumerate(chosen.get(plane, [])):
                    nrrd, nii = convert_series(s, out / "series")
                    volumes.append(nrrd)
                    if plane == "axial" and k == 0:
                        axial_nii = nii  # finest axial series guides TotalSegmentator
            groups.append({"name": name.replace("|", "+"), "volumes": volumes, "axial_nii": axial_nii,
                           "series": {p: [s["number"] for s in chosen[p]] for p in chosen}})
            print(f"  group {name}: " + ", ".join(f"{p}={[s['number'] for s in chosen[p]]}" for p in chosen))
    else:
        print("[1/5] using given volumes as one group")
        volumes = [Path(v) for v in args.volumes]
        axial_nii = out / "series" / "axial.nii.gz"
        if not axial_nii.exists():
            sitk.WriteImage(sitk.ReadImage(str(volumes[0])), str(axial_nii), useCompression=True)
        groups.append({"name": "volumes", "volumes": volumes, "axial_nii": axial_nii, "series": {}})
    for g in groups:
        g["bounds"] = fuse_planes.image_bounds(sitk.ReadImage(str(g["volumes"][0])))
        g["totalseg"] = out / "totalseg" / g["name"]
        g["totalseg"].mkdir(parents=True, exist_ok=True)
        manifest["groups"][g["name"]] = {"series": g["series"], "bounds_lps": [b.tolist() for b in g["bounds"]],
                                         "volumes": [str(v) for v in g["volumes"]]}

    # 2. guides per group
    classes = sorted({c for b in bones for c in [b, *neighbours(b)]} & set(ALL_BONES))  # TS classes only
    print(f"[2/5] TotalSegmentator guides: {len(classes)} classes x {len(groups)} group(s)")
    for g in groups:
        run_totalsegmentator(g["axial_nii"], g["totalseg"], classes, args.device)

    # 3 + 4. per bone (fusion + extraction), --jobs in parallel; Blender import in this process
    jobs = []
    for bone in bones:
        g, bounds = choose_group(bone, groups, atlas_labels=args.atlas_labels)
        if g is None:
            if not want_all:
                print(f"!! no guide label for {bone} in any group; skipping")
            continue
        lo, hi, count = bounds
        manifest["bones"][bone] = {"group": g["name"], "label_voxels": count,
                                   "label_bounds_lps": [lo.tolist(), hi.tolist()]}
        jobs.append((bone, g, out / "fused" / f"{bone}_fused.nrrd"))
    print(f"[3/5+4/5] {len(jobs)} bones, {args.jobs} parallel job(s)")
    pushed_first = False

    def finish(bone, outputs, error):
        nonlocal pushed_first
        if error:
            print(f"!! {bone} failed: {error}")
            manifest["bones"][bone]["error"] = error
            return
        manifest["outputs"][bone] = outputs
        info = json.loads(Path(outputs["json"]).read_text())
        print(f"  {bone} <- {outputs['group']}: {info['faces']} faces, {info['volume_cm3']:.1f} cm^3, "
              f"size {[round(v) for v in info['size_mm']]} mm")
        (out / "run.json").write_text(json.dumps(manifest, indent=1, default=str))
        if args.blender != "none":
            push_to_blender(args, [(bone, outputs)], first=not pushed_first)
            pushed_first = True

    if args.jobs <= 1:
        for bone, g, fused in jobs:
            print(f"[3/5] {bone} <- group {g['name']}: fusing {len(g['volumes'])} volumes; [4/5] extracting")
            finish(*process_bone(bone, g, fused, out, args))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_bone, bone, g, fused, out, args): bone for bone, g, fused in jobs}
            for fut in as_completed(futures):
                finish(*fut.result())

    if args.resegment and manifest["outputs"]:
        # consistency pass: every consecutive vertebra pair plus every pair the overlap report
        # flags, cut once with a shared decision (resegment_pairs.py), then the report again
        import subprocess
        py = sys.executable
        qa = out / "qa"
        subprocess.run([py, str(HERE / "check_overlaps.py"), str(out), "--out", str(qa)], check=False)
        present = [v for v in VERTEBRAE if v in manifest["outputs"]]
        pairs = [f"{a}:{b}" for a, b in zip(present, present[1:])]
        rep = qa / "overlaps.json"
        if rep.exists():
            for pr in json.loads(rep.read_text())["pairs"]:
                if pr["flag"] == "BAD" and (pr["mesh"].get("penetration_mm") or 0) >= 1.0:
                    pairs.append(f"{pr['A']}:{pr['B']}")
        # rib heads and the sacroiliac joints are contested too, not only the vertebra pairs
        for b in manifest["outputs"]:
            if b.startswith("rib_"):
                side, n = b.split("_")[1], int(b.split("_")[2])
                pairs += [f"{b}:vertebrae_T{k}" for k in (n - 1, n) if 1 <= k <= 12 and f"vertebrae_T{k}" in manifest["outputs"]]
        for h in ("hip_left", "hip_right"):  # arbitrated only; resegment_pairs skips carving them
            if h in manifest["outputs"] and "sacrum" in manifest["outputs"]:
                pairs.append(f"{h}:sacrum")
        cmd = [py, str(HERE / "resegment_pairs.py"), str(out), "--pairs", *dict.fromkeys(pairs),
               "--smooth-iters", str(args.smooth_iters), "--joint-level", str(args.joint_level)]
        if args.atlas_labels:
            cmd += ["--atlas-labels", str(args.atlas_labels)]
        if args.blender != "none":
            cmd += ["--blender", "--collection", args.collection]
        print(f"[5/5] consistency pass over {len(dict.fromkeys(pairs))} pairs")
        subprocess.run(cmd, check=False)
        subprocess.run([py, str(HERE / "check_overlaps.py"), str(out), "--out", str(qa)], check=False)

    if args.blender != "none" and not args.keep_old and manifest["outputs"]:
        try:
            for collection in {args.collection, "cartilage"}:
                print("  " + str(blender_bridge.run_code(REMOVE_OTHER_RUNS % {"collection": collection,
                                                                              "run_tag": args.run_tag})).strip())
        except (ConnectionRefusedError, OSError) as exc:
            print(f"!! Blender not reachable for cleanup ({exc})")

    manifest["seconds"] = round(time.time() - t_start, 1)
    (out / "run.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"done in {manifest['seconds']}s; manifest {out / 'run.json'}")


if __name__ == "__main__":
    main()
