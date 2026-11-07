"""Pure-logic tests (no GPU/network): python tests/test_logic.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wastesorter.pipeline import SessionSmoother, fuse_bin_probs  # noqa: E402
from wastesorter.taxonomy import (BINS, CLIP_OUTPUTS, MATERIAL_TO_BIN, MATERIALS, class_to_bin_matrix,  # noqa: E402
                                  normalize_label, prompt_matrix)


def test_labels():
    assert normalize_label("1-Cardboard") == "cardboard"
    assert normalize_label("Miscellaneous Trash") == "trash"
    assert normalize_label("Food Organics") == "organic"
    assert normalize_label("Vegetation") == "organic"
    assert normalize_label("Textile Trash") == "textile"
    assert normalize_label("__MACOSX") is None
    assert set(MATERIAL_TO_BIN) == set(MATERIALS)
    assert set(MATERIAL_TO_BIN.values()) <= set(BINS)


def test_matrices():
    m = class_to_bin_matrix(MATERIALS)
    assert m.shape == (len(MATERIALS), len(BINS)) and (m.sum(1) == 1).all()
    descs, pm = prompt_matrix()
    assert pm.shape == (len(descs), len(CLIP_OUTPUTS)) and (pm.sum(1) == 1).all()
    assert (pm.sum(0) > 0).all(), "every bin (and no_item) needs at least one prompt"


def test_fusion():
    sup = np.array([0.9, 0.05, 0.05, 0.0])
    clip = np.array([0.1, 0.1, 0.1, 0.7])
    f = fuse_bin_probs(sup.copy(), clip, 0.6)
    assert abs(f.sum() - 1) < 1e-6
    assert abs(f[3] - 0.7) < 1e-6, "hazardous prob should equal CLIP's"
    f2 = fuse_bin_probs(sup.copy(), np.array([0.8, 0.1, 0.09, 0.01]), 0.6)
    assert f2.argmax() == 0
    assert fuse_bin_probs(None, clip, 0.6).argmax() == 3
    assert fuse_bin_probs(sup.copy(), None, 0.6).argmax() == 0


def test_smoother():
    s = SessionSmoother(alpha=0.5, ttl=5.0)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert (s.update("x", a, now=0.0) == a).all()
    assert np.allclose(s.update("x", b, now=1.0), [0.5, 0.5])
    assert (s.update("x", b, now=20.0) == b).all(), "expired history must be dropped"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
