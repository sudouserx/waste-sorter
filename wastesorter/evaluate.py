"""Evaluate the full pipeline at the BIN level and tune the fusion.

train.py only scores the supervised classifier. This script answers the questions that decide what the app shows:
  * does CLIP fusion help or hurt bin accuracy, and at which weight / aggregation?
  * how often is the answer "ok" vs "uncertain", and how accurate are the "ok" answers?
  * does the no-item gate fire on real items?
  * (--leakage) how many test images are near-duplicates of training images?

Usage
  # held-out split written by train.py (material labels -> bins; no hazardous / no_item examples exist here)
  python -m wastesorter.evaluate --ckpt ckpt/best.pt --splits ckpt/splits.json --leakage

  # your own phone photos, one sub-folder per bin (recyclable/ organic/ non_recyclable/ hazardous/ no_item/)
  # or per material (plastic/, glass/, ...). This is the number that matters for the app.
  python -m wastesorter.evaluate --ckpt ckpt/best.pt --folder my_photos/
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .data import IMG_EXT
from .pipeline import PipelineConfig, WastePredictor, decide, fuse_bin_probs, load_image
from .taxonomy import BINS, MATERIAL_TO_BIN, MATERIALS, NO_ITEM, label_to_bin

W_GRID = [round(x, 1) for x in np.linspace(0, 1, 11)]


def items_from_splits(path: str, split: str) -> List[Tuple[str, str, str]]:
    d = json.loads(Path(path).read_text())
    return [(s["path"], MATERIAL_TO_BIN[MATERIALS[s["label"]]], s["source"]) for s in d[split]]


def items_from_folder(root: str) -> List[Tuple[str, str, str]]:
    out, skipped = [], set()
    for sub in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        b = label_to_bin(sub.name)
        if b is None:
            skipped.add(sub.name)
            continue
        out += [(str(f), b, "folder") for f in sorted(sub.rglob("*")) if f.suffix.lower() in IMG_EXT]
    if skipped:
        print(f"[warn] ignored folders with unknown labels: {sorted(skipped)}")
    if not out:
        raise SystemExit(f"no labelled images under {root} (expected sub-folders like recyclable/, organic/, ...)")
    return out


def run_models(pred: WastePredictor, paths: List[str], batch: int) -> Dict[str, Optional[np.ndarray]]:
    acc: Dict[str, list] = {"sup_bins": [], "mat_probs": [], "prompt_probs": []}
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(paths), batch):
            imgs = list(ex.map(load_image, paths[i:i + batch]))
            c = pred.components(imgs)
            for k in acc:
                if c[k] is not None:
                    acc[k].append(c[k])
            print(f"\r  {min(i + batch, len(paths))}/{len(paths)} images", end="", flush=True)
    print()
    return {k: (np.concatenate(v) if v else None) for k, v in acc.items()}


def score(fused: np.ndarray, no_item: np.ndarray, y: np.ndarray, cfg: PipelineConfig) -> dict:
    """y: bin index, or -1 for no_item images."""
    item = y >= 0
    res = {"n_items": int(item.sum())}
    if item.any():
        res["bin_accuracy"] = float((fused[item].argmax(1) == y[item]).mean())
    st = [decide(f, float(n), cfg.conf_threshold, cfg.no_item_threshold) for f, n in zip(fused, no_item)]
    status = np.array([s[0] for s in st])
    if item.any():
        ok = (status == "ok") & item
        res["ok_rate"] = float(ok.sum() / item.sum())
        res["uncertain_rate"] = float(((status == "uncertain") & item).sum() / item.sum())
        res["false_no_item_rate"] = float(((status == "no_item") & item).sum() / item.sum())
        res["accuracy_when_ok"] = float((fused[ok].argmax(1) == y[ok]).mean()) if ok.any() else None
    if (~item).any():
        res["no_item_recall"] = float((status[~item] == "no_item").mean())
    return res


def dhash(path: str) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def near_duplicates(test_paths: List[str], train_paths: List[str], max_dist: int = 4) -> np.ndarray:
    """For each test image, the smallest dHash Hamming distance to any training image (<= max_dist ~ near-duplicate)."""
    with ThreadPoolExecutor(8) as ex:
        te = np.array(list(ex.map(dhash, test_paths)), dtype=np.uint64)
        tr = np.array(list(ex.map(dhash, train_paths)), dtype=np.uint64)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    best = np.empty(len(te), dtype=np.int64)
    for i in range(0, len(te), 64):
        x = (te[i:i + 64, None] ^ tr[None, :]).view(np.uint8).reshape(-1, len(tr), 8)
        best[i:i + 64] = popcount[x].sum(-1, dtype=np.int64).min(1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="best.pt (omit for CLIP-only)")
    ap.add_argument("--splits", default=None, help="splits.json from train.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--folder", default=None, help="folder of bin- or material-named sub-folders")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--w", type=float, default=0.6, help="current w_supervised (for the status metrics)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--no-item", type=float, default=0.6)
    ap.add_argument("--leakage", action="store_true", help="near-duplicate check of test vs train (needs --splits)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args()
    if not a.splits and not a.folder:
        raise SystemExit("give --splits and/or --folder")

    cfg = PipelineConfig(checkpoint=a.ckpt, use_clip=not a.no_clip, w_supervised=a.w,
                         conf_threshold=a.conf, no_item_threshold=a.no_item)
    pred = WastePredictor(cfg)
    from .clip_zeroshot import clip_bins

    sets = {}
    if a.splits:
        sets[f"splits:{a.split}"] = items_from_splits(a.splits, a.split)
    if a.folder:
        sets[f"folder:{Path(a.folder).name}"] = items_from_folder(a.folder)

    results = {}
    for name, items in sets.items():
        paths = [p for p, _, _ in items]
        y = np.array([-1 if b == NO_ITEM else BINS.index(b) for _, b, _ in items])
        labels, counts = np.unique([b for _, b, _ in items], return_counts=True)
        print(f"\n=== {name}: {len(items)} images, labels {dict(zip(labels.tolist(), counts.tolist()))}")
        comp = run_models(pred, paths, a.batch)
        n = len(items)
        r: Dict = {"n": n}

        sup = comp["sup_bins"]
        if sup is not None:
            r["supervised_only"] = score(sup, np.zeros(n), y, cfg)
            r["supervised_only"].pop("no_item_recall", None)
        if comp["prompt_probs"] is not None:
            for agg in ("sum", "max"):
                cb, ni = clip_bins(comp["prompt_probs"], pred.clip.prompt_matrix, agg)
                r[f"clip_only_{agg}"] = score(cb, ni, y, cfg)
                if sup is not None:
                    sweep = {}
                    for w in W_GRID:
                        fused = np.stack([fuse_bin_probs(s, c, w) for s, c in zip(sup, cb)])
                        sweep[w] = score(fused, ni, y, cfg)
                    r[f"fused_{agg}"] = sweep

        if a.leakage and name.startswith("splits:"):
            d = json.loads(Path(a.splits).read_text())
            dist = near_duplicates(paths, [s["path"] for s in d["train"]])
            dup = dist <= 4
            srcs = np.array([s for _, _, s in items])
            r["leakage"] = {"near_duplicate_rate": float(dup.mean()),
                            "per_source": {s: float(dup[srcs == s].mean()) for s in sorted(set(srcs.tolist()))}}
            if sup is not None and (~dup).any():
                r["leakage"]["sup_bin_accuracy_clean_only"] = float((sup[~dup].argmax(1) == y[~dup]).mean())
                if dup.any():
                    r["leakage"]["sup_bin_accuracy_duplicates_only"] = float((sup[dup].argmax(1) == y[dup]).mean())
        results[name] = r
        print_summary(r, cfg)
        if name.startswith("splits:"):
            print("note: this split has no hazardous or no_item examples, so CLIP's main benefits are not measured "
                  "here; run with --folder on your own phone photos for that.")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")


def _fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


def print_summary(r: dict, cfg: PipelineConfig):
    cols = ("bin_accuracy", "ok_rate", "accuracy_when_ok", "uncertain_rate", "false_no_item_rate", "no_item_recall")
    print(f"{'setting':<22}" + "".join(f"{c[:12]:>14}" for c in cols))
    rows = []
    if "supervised_only" in r:
        rows.append(("supervised only", r["supervised_only"]))
    for agg in ("sum", "max"):
        if f"clip_only_{agg}" in r:
            rows.append((f"clip only ({agg})", r[f"clip_only_{agg}"]))
        sweep = r.get(f"fused_{agg}")
        if sweep:
            best_w = max(sweep, key=lambda w: sweep[w].get("bin_accuracy", 0))
            cur = round(cfg.w_supervised, 1)
            rows.append((f"fused {agg} w={cur}", sweep.get(cur, {})))
            rows.append((f"fused {agg} best w={best_w}", sweep[best_w]))
    for label, m in rows:
        print(f"{label:<22}" + "".join(f"{_fmt(m.get(c)):>14}" for c in cols))
    if "leakage" in r:
        print("leakage:", json.dumps(r["leakage"]))


if __name__ == "__main__":
    main()
