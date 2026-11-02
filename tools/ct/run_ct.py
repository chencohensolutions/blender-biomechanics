#!/usr/bin/env python3
"""One command for the whole CT pipeline: guide labels -> bones -> muscles -> Blender, one structure at a time.

Stages (each cached, so a rerun only redoes what is missing):

  1. guides   TotalSegmentator on each acquisition's finest ORIGINAL axial series: the bone classes
              (task total, --roi_subset) and the muscle tasks total (gluteals, iliopsoas),
              abdominal_muscles and headneck_muscles, at full resolution (--fast-guides trades
              guide quality for a tenfold faster seed; the model refines either way).
  2. bones    segment_nni.py --structures bones: nnInteractive v1.0 on 0.6 mm isotropic crops fused
              in the Fourier domain from the axial + coronal + sagittal series of the acquisition
              (fuse_series.py); joint resolution per acquisition, cross-acquisition carve, contact
              report (measured vs inferred boundaries), Gaussian + Taubin meshes.
  3. muscles  segment_nni.py --structures muscles: the same on ONE merged 0.8 mm volume over all
              acquisitions (blended across their seam); the other muscles and the saved bone masks
              are the negative prompts; bones are carved out of the muscle labels afterwards.
  4. organs   segment_nni.py --structures organs: lungs, heart, airway, oesophagus, thyroid, the
              abdominal organs, the great vessels and the spinal cord likewise, with the bone and
              muscle masks as negative prompts and carved out.
  5. blender  meshes imported into the running Blender (BlenderMCP, localhost:9876): bones into
              collection `bones`, muscles into `muscles`, organs into `thorax` / `abdomen` /
              `vessels` / `nervous`, replacing earlier runs' objects.

Console: one line per structure ([k/N] name: seconds, cm3, median HU, neighbours); the model's own
output goes to OUT/qa/segment.log.  --verbose shows everything.

  python tools/ct/run_ct.py --run RUN --blender                      # everything
  python tools/ct/run_ct.py --run RUN --structures muscles --blender # muscles only
  python tools/ct/run_ct.py --run RUN --structures organs --blender  # organs + vessels only
  python tools/ct/run_ct.py --run RUN --bones T8 sacrum --structures bones
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402
import segment_nni  # noqa: E402

MUSCLE_TASKS = {"total": segment_nni.MUSCLES_TOTAL, "abdominal_muscles": None, "headneck_muscles": None}


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def task_classes(task):
    from totalsegmentator.map_to_binary import class_map
    return list(class_map[task].values())


def run_guides(groups, structures, device, fast, log):
    """TotalSegmentator guides for every acquisition; cached per label file."""
    exe = Path(sys.executable).parent / "TotalSegmentator"
    for g in groups:
        axial = Path(g["volume"])
        ts_dir = Path(g["totalseg"])
        ts_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        if "bones" in structures:
            jobs.append(("total", [b for b in pipeline.ALL_BONES if b != "costal_cartilages"] + ["costal_cartilages"], False))
        if "muscles" in structures:
            jobs.append(("total", segment_nni.MUSCLES_TOTAL, fast))
            jobs.append(("abdominal_muscles", None, fast))
            jobs.append(("headneck_muscles", None, fast))
        if "organs" in structures:
            jobs.append(("total", segment_nni.ORGANS, False))
        for task, classes, use_fast in jobs:
            wanted = classes if classes else task_classes(task)
            missing = [c for c in wanted if not (ts_dir / f"{c}.nii.gz").exists()]
            if not missing:
                log(f"  guides {g['name']}/{task}: {len(wanted)} labels cached")
                continue
            t0 = time.time()
            if classes:  # roi subset of the big model
                out_dir = ts_dir
                cmd = [str(exe), "-i", str(axial), "-o", str(out_dir), "--roi_subset", *missing, "--device", device]
            else:        # a whole task: write to its own folder, then copy the labels next to the others
                out_dir = ts_dir.parent / f"{g['name']}_{task}"
                cmd = [str(exe), "-i", str(axial), "-o", str(out_dir), "--task", task, "--device", device]
            if use_fast:
                cmd.append("--fast")
            log(f"  guides {g['name']}/{task}: {len(missing)} labels to compute{' (fast 3 mm)' if use_fast else ''} ...")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                log(f"!! TotalSegmentator {task} on {g['name']} failed: {(res.stderr or res.stdout).strip().splitlines()[-1]}")
                continue
            if out_dir != ts_dir:
                for f in out_dir.glob("*.nii.gz"):
                    if not (ts_dir / f.name).exists():  # trapezius exists in two tasks: first one wins
                        (ts_dir / f.name).write_bytes(f.read_bytes())
            log(f"  guides {g['name']}/{task}: done in {hms(time.time() - t0)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pipeline run folder (run.json, series/, totalseg/)")
    ap.add_argument("--structures", nargs="+", choices=["bones", "muscles", "organs"], default=["bones", "muscles", "organs"])
    ap.add_argument("--bones", nargs="+", default=["all"], help="bone names or 'all'")
    ap.add_argument("--muscles", nargs="+", default=["all"], help="muscle names or 'all'")
    ap.add_argument("--organs", nargs="+", default=["all"], help="organ / vessel names or 'all'")
    ap.add_argument("--atlas-labels", help="fitted atlas labels used where TotalSegmentator has no bone label "
                                           "(default RUN/atlas_bp3d/labels when it exists)")
    ap.add_argument("--group-override", nargs="*", default=["rib_right_11=Lung"], metavar="BONE=GROUP")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-guides", action="store_true", help="do not run TotalSegmentator (use cached labels only)")
    ap.add_argument("--fast-guides", action="store_true", help="muscle guide tasks in TotalSegmentator's 3 mm fast mode")
    ap.add_argument("--force", action="store_true", help="re-segment structures that already have a mask")
    ap.add_argument("--skip-segment", action="store_true", help="only resolve, mesh and import existing masks")
    ap.add_argument("--blender", action="store_true", help="import into the running Blender")
    ap.add_argument("--no-replace", action="store_true", help="keep other runs' objects in the Blender collections")
    ap.add_argument("--verbose", action="store_true", help="show the model's output on the console too")
    ap.add_argument("--run-tag", default="ct2021-nni-fused", help="Blender ct_run tag for bones (muscles / organs get -muscles / -organs)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    t_all = time.time()

    def log(msg):
        print(msg, flush=True)

    atlas = args.atlas_labels or (str(run / "atlas_bp3d" / "labels") if (run / "atlas_bp3d" / "labels").exists() else None)
    groups = segment_nni.load_groups(run)
    log(f"run: {run}")
    log("acquisitions: " + "; ".join(f"{g['name']} <- " + "+".join(s.name for s in g["series"]) for g in groups))

    # 1. guides
    if not args.skip_guides:
        log("\n[1/5] guide labels (TotalSegmentator)")
        run_guides(groups, args.structures, args.device, args.fast_guides, log)
    else:
        log("\n[1/5] guide labels: skipped (cached labels only)")

    summary = {}
    common = ["--run", str(run), "--device", args.device] + (["--quiet"] if not args.verbose else []) + \
             (["--force"] if args.force else []) + (["--skip-segment"] if args.skip_segment else []) + \
             (["--blender"] if args.blender else []) + ([] if args.no_replace else ["--replace"])

    # 2. bones
    if "bones" in args.structures:
        log("\n[2/5] bones: nnInteractive on fused axial+coronal+sagittal crops (0.6 mm)")
        t0 = time.time()
        argv_b = common + ["--structures", "bones", "--bones", *args.bones, "--run-tag", args.run_tag]
        if atlas:
            argv_b += ["--atlas-labels", atlas]
        if args.group_override:
            argv_b += ["--group-override", *args.group_override]
        summary["bones"] = (segment_nni.main(argv_b) or {}, time.time() - t0)
    else:
        log("\n[2/5] bones: skipped")

    # 3. muscles
    if "muscles" in args.structures:
        log("\n[3/5] muscles: nnInteractive on the merged fused volume (0.8 mm), bones carved out")
        t0 = time.time()
        argv_m = common + ["--structures", "muscles", "--bones", *args.muscles, "--run-tag", args.run_tag + "-muscles"]
        summary["muscles"] = (segment_nni.main(argv_m) or {}, time.time() - t0)
    else:
        log("\n[3/5] muscles: skipped")

    # 4. organs
    if "organs" in args.structures:
        log("\n[4/5] organs: nnInteractive on the merged fused volume (0.8 mm), bones and muscles carved out")
        t0 = time.time()
        argv_o = common + ["--structures", "organs", "--bones", *args.organs, "--run-tag", args.run_tag + "-organs"]
        summary["organs"] = (segment_nni.main(argv_o) or {}, time.time() - t0)
    else:
        log("\n[4/5] organs: skipped")

    # 5. summary
    log(f"\n[5/5] summary ({hms(time.time() - t_all)} total)")
    for kind, (results, sec) in summary.items():
        out = run / ("nni_fused" if kind == "bones" else f"nni_fused_{kind}")
        contacts_path = out / "qa" / "contacts.json"
        contacts = json.loads(contacts_path.read_text()) if contacts_path.exists() else {}
        measured = sum(1 for v in contacts.values() if v["boundary"] == "measured")
        vol = sum(v["volume_cm3"] for v in results.values())
        log(f"  {kind}: {len(results)} meshes, {vol:.0f} cm^3, {len(contacts)} contacts "
            f"({measured} measured / {len(contacts) - measured} inferred), {hms(sec)}  -> {out / 'meshes'}")
    if args.blender:
        log("  Blender: objects replaced in collections bones / muscles / thorax, abdomen, vessels, nervous; "
            "the .blend was saved by the import")
    else:
        log("  Blender: not touched (add --blender)")
    return summary


if __name__ == "__main__":
    main()
