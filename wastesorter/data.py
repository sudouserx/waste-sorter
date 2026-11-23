"""Dataset acquisition (TrashNet via HF Hub, RealWaste via git/Kaggle input), unification, splits."""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .taxonomy import MATERIALS, normalize_label

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REALWASTE_GIT = "https://github.com/sam-single/realwaste"
TRASHNET_HF_REPO = "garythung/trashnet"
TRASHNET_ZIP = "dataset-resized.zip"


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    source: str  # "trashnet" | "realwaste"


def _has_images(p: Path) -> bool:
    return p.exists() and any(f.suffix.lower() in IMG_EXT for f in p.rglob("*") if f.is_file())


def scan_class_folders(root: Path, source: str) -> List[Sample]:
    """Recursively find folders whose name maps to a known material and collect their images."""
    out: List[Sample] = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("__MACOSX", ".git"))  # prune instead of walking .git
        cls = normalize_label(Path(d).name)
        if cls is None:
            continue
        for f in files:
            if f.startswith("._") or Path(f).suffix.lower() not in IMG_EXT:
                continue
            out.append(Sample(str(Path(d) / f), MATERIALS.index(cls), source))
    return sorted(out, key=lambda s: s.path)


def prepare_trashnet(data_dir: Path) -> Path:
    target = data_dir / "trashnet"
    if _has_images(target):
        return target
    from huggingface_hub import hf_hub_download

    zp = hf_hub_download(repo_id=TRASHNET_HF_REPO, filename=TRASHNET_ZIP, repo_type="dataset")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(target)
    return target


def prepare_realwaste(data_dir: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--realwaste-dir not found: {p}")
        return p
    kin = Path("/kaggle/input")  # if you attached a RealWaste Kaggle dataset, use it
    if kin.exists():
        for d in kin.iterdir():
            if "realwaste" in d.name.lower().replace("-", "").replace("_", "") and _has_images(d):
                return d
    target = data_dir / "realwaste"
    if _has_images(target):
        return target
    if shutil.which("git") is None:
        raise RuntimeError("git not found; attach RealWaste as a dataset or pass --realwaste-dir")
    subprocess.run(["git", "clone", "--depth", "1", REALWASTE_GIT, str(target)], check=True)
    return target


def get_samples(data_dir: str, trashnet_dir: Optional[str] = None, realwaste_dir: Optional[str] = None) -> List[Sample]:
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    tn = Path(trashnet_dir) if trashnet_dir else prepare_trashnet(data)
    rw = prepare_realwaste(data, realwaste_dir)
    samples = scan_class_folders(tn, "trashnet") + scan_class_folders(rw, "realwaste")
    if not any(s.source == "trashnet" for s in samples) or not any(s.source == "realwaste" for s in samples):
        raise RuntimeError("Could not find images for both TrashNet and RealWaste; check paths / internet access.")
    counts = np.bincount([s.label for s in samples], minlength=len(MATERIALS))
    for name, c in zip(MATERIALS, counts):
        if c < 20:
            raise RuntimeError(f"Class '{name}' has only {int(c)} images; dataset discovery failed.")
    return samples


def subsample(samples: List[Sample], per_group: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    groups = {}
    for s in samples:
        groups.setdefault((s.source, s.label), []).append(s)
    out: List[Sample] = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:per_group])
    return out


def _natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def split_samples(samples: List[Sample], val_frac=0.1, test_frac=0.1, seed=42, mode="chunked", chunk=25
                  ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split into train/val/test.

    mode="random":  stratified random split (what the first run used). Near-duplicate photos of the same
                    object (TrashNet shoots each object from several angles) can land on both sides.
    mode="chunked": within each (source, class), sort by natural filename order, cut into blocks of ~`chunk`
                    consecutive files and assign whole blocks to a split. HEURISTIC: assumes consecutively
                    numbered files were captured together. Unverified for both datasets, so compare the
                    random vs chunked test accuracy; a big drop means leakage was inflating the numbers.
    """
    if mode == "random":
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(samples))
        strat = np.array([f"{s.source}|{s.label}" for s in samples])
        tv, te = train_test_split(idx, test_size=test_frac, stratify=strat, random_state=seed)
        tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac), stratify=strat[tv], random_state=seed)
        pick = lambda ii: [samples[i] for i in ii]
        return pick(tr), pick(va), pick(te)
    if mode != "chunked":
        raise ValueError(f"unknown split mode: {mode}")

    rng = random.Random(seed)
    strata = {}
    for s in samples:
        strata.setdefault((s.source, s.label), []).append(s)
    tr: List[Sample] = []; va: List[Sample] = []; te: List[Sample] = []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda s: _natural_key(s.path))
        n = len(items)
        size = max(5, min(chunk, n // 10))
        chunks = [items[i:i + size] for i in range(0, n, size)]
        rng.shuffle(chunks)
        n_te, n_va = max(1, round(n * test_frac)), max(1, round(n * val_frac))
        got_te = got_va = 0
        for c in chunks:
            if got_te < n_te:
                te += c; got_te += len(c)
            elif got_va < n_va:
                va += c; got_va += len(c)
            else:
                tr += c
    return tr, va, te


class WasteDataset(Dataset):
    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        with Image.open(s.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(im), s.label