"""CLI: python -m wastesorter.predict img1.jpg folder/ --ckpt outputs/best.pt"""
import argparse
import json
from pathlib import Path

from .data import IMG_EXT
from .pipeline import PipelineConfig, WastePredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--w", type=float, default=0.6, help="w_supervised")
    ap.add_argument("--agg", default="sum", choices=["sum", "max"])
    a = ap.parse_args()
    pred = WastePredictor(PipelineConfig(checkpoint=a.ckpt, use_clip=not a.no_clip, w_supervised=a.w, clip_agg=a.agg))
    files = []
    for p in map(Path, a.paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(json.dumps({"file": str(p), "error": "not found"}))
    for f in files:
        try:
            print(json.dumps({"file": str(f), **pred.predict(f)}))
        except OSError as e:  # unreadable / corrupt image: report and keep going
            print(json.dumps({"file": str(f), "error": str(e)}))


if __name__ == "__main__":
    main()
