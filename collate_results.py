#!/usr/bin/env python3
"""
collate_results.py
==================
Load all sweep results, flatten to rows, and save one parquet file per
(hierarchy, task) pair to keep memory usage manageable:

    data/ti/{hier}__{task}.parquet         — pure TI hierarchies
    data/exception/{hier}__{task}.parquet  — exception hierarchies

Sources
-------
    results/sweep3/          LoRA sweep   (model/task/hier/lora_r…/seed…)
    results/sweep3_probe/    Probe sweep  (model/task/hier/l2_…/seed…)
    results/sweep_full/      Full FT      (model/task/hier/seed…)

Usage
-----
    python collate_results.py [--roots LORA PROBE FULL] [--out-dir data]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ── Hierarchy sets ─────────────────────────────────────────────────────────────
TI_TAGS = {"ti_n9"}
EXC_TAGS = {
    "exc_mid_n9", "exc_mid_n11",
    "exc_far_n11",
}
ALL_TAGS = TI_TAGS | EXC_TAGS

# ── Path regexes ───────────────────────────────────────────────────────────────
_LORA_RE = re.compile(
    r"(?P<model>[^/]+)/(?P<task>[^/]+)/(?P<hier>[^/]+)"
    r"/lora_r(?P<lora_r>\d+)_alpha(?P<lora_alpha>\d+)"
    r"/seed(?P<seed>\d+)_(?P<cond>fwd|rev)"
)
_PROBE_RE = re.compile(
    r"(?P<model>[^/]+)/(?P<task>[^/]+)/(?P<hier>[^/]+)"
    r"/l2_(?P<l2>[^/]+)"
    r"/seed(?P<seed>\d+)_(?P<cond>fwd|rev)"
)
_FULL_RE = re.compile(
    r"(?P<model>[^/]+)/(?P<task>[^/]+)/(?P<hier>[^/]+)"
    r"/seed(?P<seed>\d+)_(?P<cond>fwd|rev)"
)


def _unslug_l2(slug: str) -> float:
    return float(slug.replace("p", "."))


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load(root: Path, pattern: re.Pattern, glob_depth: str,
          extra_fn, setting_name: str, keep_tags: set) -> list[dict]:
    runs = []
    for p in sorted(root.glob(f"{glob_depth}/results.json")):
        rel = p.parent.relative_to(root)
        m = pattern.search(str(rel))
        if m is None or m["hier"] not in keep_tags:
            continue
        with open(p) as f:
            data = json.load(f)
        run = {
            "setting": setting_name,
            "model":   m["model"],
            "task":    m["task"],
            "hier":    m["hier"],
            "seed":    int(m["seed"]),
            "reverse": m["cond"] == "rev",
            "data":    data,
        }
        run.update(extra_fn(m))
        runs.append(run)
    return runs


def load_all(lora_root: Path, probe_root: Path, full_root: Path,
             keep_tags: set) -> list[dict]:
    runs = []
    # LoRA:  model/task/hier/lora_r{R}_alpha{A}/seed{S}_{cond}  → 5 levels
    runs += _load(lora_root, _LORA_RE, "*/*/*/*/*",
                  lambda m: {"lora_r": int(m["lora_r"])},
                  "lora", keep_tags)
    # Probe: model/task/hier/l2_{slug}/seed{S}_{cond}           → 5 levels
    runs += _load(probe_root, _PROBE_RE, "*/*/*/*/*",
                  lambda m: {"l2_reg": _unslug_l2(m["l2"]), "l2_slug": m["l2"]},
                  "probe", keep_tags)
    # Full:  model/task/hier/seed{S}_{cond}                     → 4 levels
    runs += _load(full_root, _FULL_RE, "*/*/*/*",
                  lambda m: {},
                  "full", keep_tags)
    return runs


# ── Flatten runs → rows ────────────────────────────────────────────────────────

def _paired_correct(items: list[dict]) -> dict:
    by_key: dict = {}
    for it in items:
        if it.get("no_gt", False):
            continue
        key = (min(it["rank_a"], it["rank_b"]), max(it["rank_a"], it["rank_b"]))
        by_key.setdefault(key, {})[it["direction"]] = it
    return {
        k: pair["forward"]["prob_yes"] > pair["backward"]["prob_yes"]
        for k, pair in by_key.items()
        if "forward" in pair and "backward" in pair
    }


def run_to_rows(run: dict) -> list[dict]:
    rows = []
    for phase in ("pre", "post"):
        items = run["data"][f"{phase}_finetune"]
        pc = _paired_correct(items)
        for it in items:
            no_gt = it.get("no_gt", False)
            key = (min(it["rank_a"], it["rank_b"]), max(it["rank_a"], it["rank_b"]))
            gt     = (1 if it["label"] == "Yes" else -1) if not no_gt else float("nan")
            margin = it["logit_diff"] * gt             if not no_gt else float("nan")
            correct = float(
                (it["prob_yes"] > 0.5) == (it["label"] == "Yes")
            ) if not no_gt else float("nan")
            rows.append({
                "setting":        run["setting"],
                "model":          run["model"],
                "task":           run["task"],
                "hier":           run["hier"],
                "seed":           run["seed"],
                "reverse":        run["reverse"],
                "phase":          phase,
                "rank_a":         it["rank_a"],
                "rank_b":         it["rank_b"],
                "rank_higher":    min(it["rank_a"], it["rank_b"]),
                "rank_lower":     max(it["rank_a"], it["rank_b"]),
                "gap":            it["gap"],
                "is_train":       it["is_train"],
                "is_exception":   it.get("is_exception", False),
                "no_gt":          no_gt,
                "direction":      "forward" if it["rank_a"] < it["rank_b"] else "backward",
                "label":          it.get("label"),
                "logit_diff":     it["logit_diff"],
                "prob_yes":       it["prob_yes"],
                "ground_truth":   gt,
                "margin":         margin,
                "correct":        correct,
                "paired_correct": float(pc.get(key, float("nan"))),
                # hyperparams — NaN when not applicable
                "lora_r":         float(run.get("lora_r", float("nan"))),
                "l2_reg":         float(run.get("l2_reg", float("nan"))),
            })
    return rows


def runs_to_df(runs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([row for run in runs for row in run_to_rows(run)])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-root",  default="results/sweep3",       type=Path)
    parser.add_argument("--probe-root", default="results/sweep3_probe", type=Path)
    parser.add_argument("--full-root",  default="results/sweep_full",   type=Path)
    parser.add_argument("--out-dir",    default="data",                  type=Path)
    args = parser.parse_args()

    # All (hier, task) pairs present across sweeps — discovered dynamically
    # by scanning directory names, then processed one at a time.
    roots_patterns = [
        (args.lora_root,  _LORA_RE,  "*/*/*/*/*"),
        (args.probe_root, _PROBE_RE, "*/*/*/*/*"),
        (args.full_root,  _FULL_RE,  "*/*/*/*"),
    ]

    print("Discovering (hier, task) pairs …")
    pairs: set[tuple[str, str]] = set()
    for root, pattern, glob in roots_patterns:
        for p in root.glob(f"{glob}/results.json"):
            rel = p.parent.relative_to(root)
            m = pattern.search(str(rel))
            if m and m["hier"] in ALL_TAGS:
                pairs.add((m["hier"], m["task"]))

    if not pairs:
        print("No results found. Check that the results directories exist.")
        return

    # Group into ti / exception sub-dirs for tidiness
    ti_dir  = args.out_dir / "ti"
    exc_dir = args.out_dir / "exception"
    ti_dir.mkdir(parents=True, exist_ok=True)
    exc_dir.mkdir(parents=True, exist_ok=True)

    for hier, task in sorted(pairs):
        tags = {hier}   # load only this single hier
        runs = load_all(args.lora_root, args.probe_root, args.full_root, tags)
        # keep only the matching task
        runs = [r for r in runs if r["task"] == task]
        if not runs:
            continue

        n_lora  = sum(1 for r in runs if r["setting"] == "lora")
        n_probe = sum(1 for r in runs if r["setting"] == "probe")
        n_full  = sum(1 for r in runs if r["setting"] == "full")

        df = runs_to_df(runs)
        subdir = ti_dir if hier in TI_TAGS else exc_dir
        out_path = subdir / f"{hier}__{task}.parquet"
        df.to_parquet(out_path, index=False)
        print(
            f"  {hier:20s}  {task:8s}  "
            f"lora={n_lora:4d} probe={n_probe:4d} full={n_full:3d}  "
            f"{len(df):>8,} rows  →  {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
