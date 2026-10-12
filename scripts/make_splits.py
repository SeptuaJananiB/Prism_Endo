"""Scan CholecSeg8k-style data and propose a VIDEO-level train/val/test split.

Prints a table per video and per clip (frames, and in what share of the
scanned frames each target class is present, read from the dense
*_endo_watershed_mask.png maps, every Nth frame), then proposes a split made
of whole videos (default ~70/15/15 by video count, seeded) that keeps every
target class in train and, where possible, in test and val. The proposal is
written as a YAML snippet (data.group_by: video + data.splits).

Each Cholec80 video is a separate procedure; patient identity is not
released, so video-level grouping is the closest available proxy for a
patient-level split.

With ONE video no patient-level split exists: the script refuses, explains
why, prints the current clip-block (temporal) split instead and writes no file.

  .venv/Scripts/python scripts/make_splits.py
  .venv/Scripts/python scripts/make_splits.py "data.root=[video01,data/cholecseg8k/video09,data/cholecseg8k/video12]"
  .venv/Scripts/python scripts/make_splits.py --every 5 --seed 1 --out outputs/splits/video_splits.yaml
"""

import argparse
import itertools
import logging
import math
import random
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.segmentation_data import Sample, discover_from_config, split_samples  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

SPLITS = ("train", "val", "test")
ABBR = {"grasper": "grasp", "l_hook": "lhook", "abdominal_wall": "abdw", "liver": "liver",
        "gastrointestinal_tract": "gi", "fat": "fat", "connective_tissue": "conn", "blood": "blood",
        "cystic_duct": "cduct", "gallbladder": "gallb", "hepatic_vein": "hepv", "liver_ligament": "ligam"}
PROXY_NOTE = ("Each Cholec80 video is a separate procedure; patient identity is not released, so "
              "video-level grouping is the closest available proxy for a patient-level split.")


# ---------------------------------------------------------------------------
# Targets and scanning
# ---------------------------------------------------------------------------

def target_classes(class_lists: list[list[dict]]) -> list[tuple[str, list[int]]]:
    """Foreground classes (index > 0) of all task configs, deduplicated by raw
    values; a class that is just the union of others (e.g. the tissue task's
    merged 'instrument' = grasper + L-hook) is dropped."""
    seen: dict[tuple, str] = {}
    for classes in class_lists:
        for c in classes[1:]:
            vals = tuple(sorted(int(v) for v in c.get("values") or []))
            if vals and vals not in seen:
                seen[vals] = c["name"]
    out = []
    for vals, name in seen.items():
        others = set().union(*[set(v) for v in seen if v != vals])
        if len(vals) > 1 and set(vals) <= others:
            continue
        out.append((name, list(vals)))
    return out


def _frame_number(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def scan(samples: list[Sample], targets: list[tuple[str, list[int]]], every: int = 10,
         min_area: int = 200) -> dict[str, dict]:
    """Per clip: video, frames, scanned frames, frames with each target class
    (>= min_area px) and pixels per class. `samples` must be grouped by clip."""
    by_clip: dict[str, list[Sample]] = {}
    for s in samples:
        by_clip.setdefault(s.group, []).append(s)
    stats = {}
    for clip, items in by_clip.items():
        items = sorted(items, key=lambda s: (_frame_number(s.image), s.image.name))
        present = np.zeros(len(targets), np.int64)
        pixels = np.zeros(len(targets), np.int64)
        scanned = items[::max(1, every)]
        for s in scanned:
            raw = cv2.imread(str(s.label), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise IOError(f"Could not read label: {s.label}")
            if raw.ndim == 3:
                raw = raw[..., 0]
            counts = np.bincount(raw.ravel(), minlength=256)
            px = np.array([counts[v].sum() for _, v in targets], np.int64)
            pixels += px
            present += px >= min_area
        stats[clip] = {"video": items[0].video, "frames": len(items), "scanned": len(scanned),
                       "present": present, "pixels": pixels}
    return stats


def per_video(clip_stats: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for clip, st in clip_stats.items():
        v = out.setdefault(st["video"], {"clips": [], "frames": 0, "scanned": 0,
                                         "present": np.zeros_like(st["present"]),
                                         "pixels": np.zeros_like(st["pixels"])})
        v["clips"].append(clip)
        for k in ("frames", "scanned", "present", "pixels"):
            v[k] = v[k] + st[k]
    return dict(sorted(out.items()))


def print_table(title: str, rows: dict[str, dict], targets, key_name: str, show_video: bool = False):
    abbr = [ABBR.get(n, n[:5]) for n, _ in targets]
    head = f"{key_name:16s} " + (f"{'video':8s} " if show_video else "") + f"{'frames':>6s} {'scan':>4s} "
    print(f"\n{title}")
    print(head + " ".join(f"{a:>5s}" for a in abbr))
    for name, r in rows.items():
        cells = []
        for p in r["present"]:
            cells.append(f"{round(100 * p / max(r['scanned'], 1)):>5d}" if p else f"{'-':>5s}")
        line = f"{name:16s} " + (f"{r['video']:8s} " if show_video else "")
        print(line + f"{r['frames']:6d} {r['scanned']:4d} " + " ".join(cells))


# ---------------------------------------------------------------------------
# Video-level split proposal
# ---------------------------------------------------------------------------

def split_sizes(n: int, fractions=(0.7, 0.15, 0.15)) -> tuple[int, int, int]:
    """(n_train, n_val, n_test) videos. Needs >= 2 videos; with exactly 2 there
    is no patient-disjoint validation video (val = 0)."""
    if n < 2:
        raise ValueError("A video-level split needs at least 2 videos.")
    if n == 2:
        return 1, 0, 1
    n_test = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    while n - n_test - n_val < 1:
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return n - n_test - n_val, n_val, n_test


def propose_video_split(video_stats: dict[str, dict], fractions=(0.7, 0.15, 0.15),
                        seed: int = 0, max_enumerate: int = 200_000, trials: int = 50_000) -> dict:
    """Choose whole videos for train/val/test. Score (lexicographic, higher is
    better): target classes present in train, then in test, then in val,
    then closeness of the frame shares to `fractions`. Exhaustive when the
    number of assignments is small, else `trials` random ones (seeded); the
    seed only breaks ties."""
    videos = sorted(video_stats)
    n_train, n_val, n_test = split_sizes(len(videos), fractions)
    rng = random.Random(seed)
    order = videos[:]
    rng.shuffle(order)
    masks = {v: sum(1 << i for i, p in enumerate(video_stats[v]["present"]) if p) for v in videos}
    frames = {v: int(video_stats[v]["frames"]) for v in videos}
    total = sum(frames.values())
    anywhere = 0
    for m in masks.values():
        anywhere |= m

    def score(assign):
        cov, dev = {}, 0.0
        for name, frac in zip(SPLITS, fractions):
            m = 0
            for v in assign[name]:
                m |= masks[v]
            cov[name] = bin(m & anywhere).count("1")
            dev += abs(sum(frames[v] for v in assign[name]) / total - frac)
        return cov["train"], cov["test"], cov["val"], -round(dev, 9)

    def candidates():
        n_comb = math.comb(len(order), n_test) * math.comb(len(order) - n_test, n_val)
        if n_comb <= max_enumerate:
            for test in itertools.combinations(order, n_test):
                left = [v for v in order if v not in test]
                for val in itertools.combinations(left, n_val):
                    yield {"test": list(test), "val": list(val), "train": [v for v in left if v not in val]}
        else:
            for _ in range(trials):
                perm = order[:]
                rng.shuffle(perm)
                yield {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}

    best, best_score = None, None
    for cand in candidates():
        sc = score(cand)
        if best_score is None or sc > best_score:
            best, best_score = cand, sc
    best = {k: sorted(best[k]) for k in SPLITS}
    return {"splits": best, "score": best_score, "sizes": (n_train, n_val, n_test),
            "frames": {k: sum(frames[v] for v in best[k]) for k in SPLITS}, "total_frames": total}


def split_coverage(members: list[str], stats: dict[str, dict], targets) -> tuple[list[str], list[str]]:
    present = np.zeros(len(targets), np.int64)
    for m in members:
        present += stats[m]["present"]
    names = [n for n, _ in targets]
    return [n for n, p in zip(names, present) if p], [n for n, p in zip(names, present) if not p]


def snippet(root, proposal: dict, args) -> str:
    header = [f"# Proposed by scripts/make_splits.py on {date.today().isoformat()} "
              f"(seed={args.seed}, every {args.every}th frame scanned, class present = >= {args.min_area} px).",
              "# Video-level split. " + PROXY_NOTE,
              "# Merge into a task config (or a copy of configs/seg_multivideo_example.yaml).",
              "# Explicit lists (no 'rest'): videos added later never join a split silently."]
    if not proposal["splits"]["val"]:
        header.append("# WARNING: only 2 videos -> no patient-disjoint validation video (val is empty; "
                      "best.pt is then chosen on training loss).")
    body = yaml.safe_dump({"data": {"root": root, "group_by": "video", "splits": proposal["splits"]}},
                          sort_keys=False, default_flow_style=None)
    return "\n".join(header) + "\n" + body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=["configs/instrument_seg.yaml", "configs/tissue_seg.yaml"],
                   help="task configs: data section from the first, target classes from all")
    p.add_argument("--every", type=int, default=10, help="scan every Nth frame of each clip")
    p.add_argument("--min-area", type=int, default=200, help="px; a class counts as present above this")
    p.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/splits/video_splits.yaml", help="where to write the YAML snippet")
    p.add_argument("overrides", nargs="*", help="key.sub=value config overrides, e.g. data.root=[a,b]")
    args = p.parse_args(argv)

    cfgs = [load_config(c, args.overrides) for c in args.configs]
    data_cfg = cfgs[0]["data"]
    targets = target_classes([c["classes"] for c in cfgs if "classes" in c])
    samples = discover_from_config({**data_cfg, "group_by": "clip"})
    roots = data_cfg["root"] if isinstance(data_cfg["root"], (list, tuple)) else [data_cfg["root"]]
    print(f"Scanned data.root = {roots} (layout {data_cfg.get('layout', 'cholecseg8k')}), "
          f"every {args.every}th frame, class present if >= {args.min_area} px")
    print("Columns = % of scanned frames containing the class ('-' = never): "
          + ", ".join(f"{ABBR.get(n, n[:5])}={n}{v}" for n, v in targets))

    clip_stats = scan(samples, targets, args.every, args.min_area)
    vid_stats = per_video(clip_stats)
    print_table("Per video", vid_stats, targets, "video")
    print_table("Per clip", dict(sorted(clip_stats.items())), targets, "clip", show_video=True)
    absent = split_coverage(list(vid_stats), vid_stats, targets)[1]
    print(f"\n{len(vid_stats)} video(s), {len(clip_stats)} clip(s), {len(samples)} frames. "
          f"Target classes never seen in the scan: {absent or 'none'}")

    if len(vid_stats) < 2:
        only = next(iter(vid_stats))
        print(f"\nREFUSED: no patient-level split can be proposed. All data comes from ONE video ({only}), "
              "i.e. one procedure/patient. Any train/val/test split of it is within-patient: test frames "
              "share the patient, anatomy, surgeon, camera and lighting with training frames, so test scores "
              "measure generalisation across time within one procedure, not to new patients. "
              "Add >= 2 more Cholec80/CholecSeg8k videos (see docs/DATASETS.md) so that test (and val) can be "
              f"different videos. {PROXY_NOTE}")
        gb = data_cfg.get("group_by", "clip")
        print(f"\nCurrent configured split (data.group_by: {gb}) - temporal clip blocks, WITHIN-PATIENT ONLY:")
        regrouped = [Sample(s.image, s.label, s.video if gb == "video" else s.group, s.video) for s in samples]
        clip_of = {s.image: s.group for s in samples}
        data_log = logging.getLogger("prism.data")
        level = data_log.level
        data_log.setLevel(logging.ERROR)  # the within-patient caveat is printed above instead
        try:
            current = split_samples(regrouped, data_cfg["splits"])
        except ValueError as e:
            print(f"  configured splits are invalid for this data: {e}\nNo file written.")
            return 0
        finally:
            data_log.setLevel(level)
        for name, subset in current.items():
            clips = sorted({clip_of[s.image] for s in subset})
            have, miss = split_coverage(clips, clip_stats, targets)
            print(f"  {name:5s}: {len(clips):2d} clips, {len(subset):5d} frames "
                  f"({100 * len(subset) / len(samples):.1f}%) {clips}")
            print(f"         classes present: {have}")
            print(f"         classes absent:  {miss}")
        print("No file written.")
        return 0

    proposal = propose_video_split(vid_stats, tuple(args.fractions), args.seed)
    n_train, n_val, n_test = proposal["sizes"]
    print(f"\nProposed VIDEO-level split ({n_train}/{n_val}/{n_test} videos, seed {args.seed}). {PROXY_NOTE}")
    for name in ("test", "val", "train"):
        vids = proposal["splits"][name]
        have, miss = split_coverage(vids, vid_stats, targets)
        miss = [m for m in miss if m not in absent]
        fr = proposal["frames"][name]
        print(f"  {name:5s}: {len(vids):2d} videos, {fr:5d} frames ({100 * fr / proposal['total_frames']:.1f}%) {vids}")
        print(f"         target classes missing (but present elsewhere in the data): {miss or 'none'}")
    if n_val == 0:
        print("WARNING: with 2 videos there is no patient-disjoint validation video; val is empty and "
              "train_segmentation.py then keeps best.pt by training loss. Add a third video.")
    text = snippet(data_cfg["root"], proposal, args)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n{text}\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
