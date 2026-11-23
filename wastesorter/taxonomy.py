"""Single source of truth: material classes, bins, label synonyms, CLIP item prompts.

Edit MATERIAL_TO_BIN / ITEM_PROMPTS to match your local recycling rules.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

# Material classes the supervised model is trained on (TrashNet + RealWaste, merged).
MATERIALS: List[str] = ["cardboard", "glass", "metal", "organic", "paper", "plastic", "textile", "trash"]

# Output bins. "hazardous" has no training data; it comes from zero-shot CLIP only.
BINS: List[str] = ["recyclable", "organic", "non_recyclable", "hazardous"]
NO_ITEM = "no_item"  # CLIP-only gate for empty backgrounds in live camera mode
CLIP_OUTPUTS: List[str] = BINS + [NO_ITEM]

MATERIAL_TO_BIN = {
    "cardboard": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "organic": "organic",
    "textile": "non_recyclable",
    "trash": "non_recyclable",
}

BIN_INFO = {
    "recyclable": {
        "display": "Recyclable bin",
        "tip": "Empty and rinse containers; keep paper and cardboard dry and free of food.",
    },
    "organic": {
        "display": "Organic / compost bin",
        "tip": "Food scraps and plant matter only, no packaging.",
    },
    "non_recyclable": {
        "display": "General waste bin",
        "tip": "Usually not accepted for recycling.",
    },
    "hazardous": {
        "display": "Special drop-off (hazardous / e-waste)",
        "tip": "Do not put in household bins; take to a collection point.",
    },
}

# Folder-name synonyms across datasets (after stripping non-letters, lowercase).
_SYNONYMS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
    "miscellaneoustrash": "trash",
    "foodorganics": "organic",
    "foodorganic": "organic",
    "vegetation": "organic",
    "organic": "organic",
    "textiletrash": "textile",
    "textile": "textile",
}

# Folder names accepted for a bin-labelled evaluation set (evaluate.py --folder), after the same normalisation.
_BIN_SYNONYMS = {
    "recyclable": "recyclable", "recycle": "recyclable", "recycling": "recyclable",
    "compost": "organic", "organicbin": "organic",
    "nonrecyclable": "non_recyclable", "general": "non_recyclable", "generalwaste": "non_recyclable",
    "landfill": "non_recyclable", "residual": "non_recyclable",
    "hazardous": "hazardous", "ewaste": "hazardous", "special": "hazardous",
    "noitem": NO_ITEM, "empty": NO_ITEM, "background": NO_ITEM,
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def normalize_label(folder_name: str) -> Optional[str]:
    """'1-Cardboard' / 'Miscellaneous Trash' / 'Food Organics' -> canonical material, else None."""
    return _SYNONYMS.get(_key(folder_name))


def label_to_bin(folder_name: str) -> Optional[str]:
    """Folder name of a bin-labelled set -> bin (or NO_ITEM). Accepts bin names/aliases and material names."""
    k = _key(folder_name)
    if k in _BIN_SYNONYMS:
        return _BIN_SYNONYMS[k]
    mat = _SYNONYMS.get(k)
    return MATERIAL_TO_BIN[mat] if mat else None


# (description, target) where target is a bin or NO_ITEM. Used for zero-shot CLIP.
# Local rules differ: e.g. many cities take EMPTY aerosol cans with metal recycling, and some refuse plastic bags.
ITEM_PROMPTS: List[Tuple[str, str]] = [
    # recyclable
    ("a plastic bottle", "recyclable"), ("a glass bottle", "recyclable"), ("a glass jar", "recyclable"),
    ("an aluminum drink can", "recyclable"), ("a metal food tin can", "recyclable"),
    ("a cardboard box", "recyclable"), ("a cereal box", "recyclable"), ("a newspaper", "recyclable"),
    ("a magazine", "recyclable"), ("sheets of paper", "recyclable"), ("a paper bag", "recyclable"),
    ("an egg carton", "recyclable"), ("a plastic food container", "recyclable"),
    ("a plastic detergent bottle", "recyclable"), ("a milk jug", "recyclable"),
    # organic
    ("a banana peel", "organic"), ("an apple core", "organic"), ("vegetable scraps", "organic"),
    ("leftover food", "organic"), ("a piece of fruit", "organic"), ("egg shells", "organic"),
    ("coffee grounds", "organic"), ("a tea bag", "organic"), ("fallen leaves", "organic"),
    ("a slice of bread", "organic"),
    # non-recyclable
    ("a chip bag", "non_recyclable"), ("a candy wrapper", "non_recyclable"),
    ("a plastic shopping bag", "non_recyclable"), ("a styrofoam container", "non_recyclable"),
    ("a disposable coffee cup", "non_recyclable"), ("a used tissue", "non_recyclable"),
    ("a diaper", "non_recyclable"), ("a cigarette butt", "non_recyclable"),
    ("a broken ceramic plate", "non_recyclable"), ("a piece of clothing", "non_recyclable"),
    ("a shoe", "non_recyclable"), ("a toothbrush", "non_recyclable"),
    ("a plastic straw", "non_recyclable"), ("plastic cutlery", "non_recyclable"),
    ("a greasy pizza box", "non_recyclable"), ("a face mask", "non_recyclable"),
    ("a rubber glove", "non_recyclable"), ("a kitchen sponge", "non_recyclable"),
    # hazardous / e-waste
    ("a battery", "hazardous"), ("a light bulb", "hazardous"), ("a fluorescent tube", "hazardous"),
    ("a smartphone", "hazardous"), ("a laptop computer", "hazardous"),
    ("an electronic cable", "hazardous"), ("a paint can", "hazardous"),
    ("an aerosol spray can", "hazardous"), ("a syringe", "hazardous"),
    ("a bottle of pills", "hazardous"),
    # no waste item in view (scene prompts; they get scene templates, not "a piece of rubbish" templates)
    ("an empty table", NO_ITEM), ("a blank wall", NO_ITEM), ("an empty floor", NO_ITEM),
    ("a person's face", NO_ITEM), ("an empty room", NO_ITEM),
]


def prompt_matrix() -> Tuple[List[str], np.ndarray]:
    """Returns (descriptions, [P, len(CLIP_OUTPUTS)] one-hot matrix mapping prompt -> bin/no_item)."""
    descs = [d for d, _ in ITEM_PROMPTS]
    m = np.zeros((len(descs), len(CLIP_OUTPUTS)), dtype=np.float32)
    for i, (_, tgt) in enumerate(ITEM_PROMPTS):
        m[i, CLIP_OUTPUTS.index(tgt)] = 1.0
    return descs, m


def class_to_bin_matrix(classes: List[str]) -> np.ndarray:
    m = np.zeros((len(classes), len(BINS)), dtype=np.float32)
    for i, c in enumerate(classes):
        m[i, BINS.index(MATERIAL_TO_BIN[c])] = 1.0
    return m
