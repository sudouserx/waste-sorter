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
    a = ap.parse_args()
    pred = WastePredictor(PipelineConfig(checkpoint=a.ckpt, use_clip=not a.no_clip))
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    for f in files:
        print(json.dumps({"file": str(f), **pred.predict(f)}))


if __name__ == "__main__":
    main()
