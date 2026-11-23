"""Pure-logic tests (no GPU/network/model weights): python tests/test_logic.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wastesorter.clip_zeroshot import aggregate_prompts, clip_bins  # noqa: E402
from wastesorter.data import Sample, split_samples  # noqa: E402
from wastesorter.pipeline import SessionSmoother, decide, fuse_bin_probs, load_image  # noqa: E402
from wastesorter.taxonomy import (BINS, CLIP_OUTPUTS, MATERIAL_TO_BIN, MATERIALS, NO_ITEM,  # noqa: E402
                                  class_to_bin_matrix, label_to_bin, normalize_label, prompt_matrix)


def test_labels():
    assert normalize_label("1-Cardboard") == "cardboard"
    assert normalize_label("Miscellaneous Trash") == "trash"
    assert normalize_label("Food Organics") == "organic"
    assert normalize_label("Vegetation") == "organic"
    assert normalize_label("Textile Trash") == "textile"
    assert normalize_label("__MACOSX") is None
    assert set(MATERIAL_TO_BIN) == set(MATERIALS)
    assert set(MATERIAL_TO_BIN.values()) <= set(BINS)
    assert label_to_bin("Hazardous") == "hazardous" and label_to_bin("e-waste") == "hazardous"
    assert label_to_bin("plastic") == "recyclable" and label_to_bin("no_item") == NO_ITEM
    assert label_to_bin("non_recyclable") == "non_recyclable" and label_to_bin("misc") is None


def test_matrices():
    m = class_to_bin_matrix(MATERIALS)
    assert m.shape == (len(MATERIALS), len(BINS)) and (m.sum(1) == 1).all()
    descs, pm = prompt_matrix()
    assert pm.shape == (len(descs), len(CLIP_OUTPUTS)) and (pm.sum(1) == 1).all()
    assert (pm.sum(0) > 0).all(), "every bin (and no_item) needs at least one prompt"
    assert len(set(descs)) == len(descs), "duplicate prompt descriptions"


def test_clip_aggregation():
    _, pm = prompt_matrix()
    p = np.full((2, pm.shape[0]), 1.0 / pm.shape[0])
    for mode in ("sum", "max"):
        out = aggregate_prompts(p, pm, mode)
        assert np.allclose(out.sum(1), 1)
    # uniform prompt probs: "max" is unbiased, "sum" favours bins that own more prompts
    assert np.allclose(aggregate_prompts(p, pm, "max")[0], 1.0 / len(CLIP_OUTPUTS))
    b, ni = clip_bins(p[0], pm, "sum")
    assert b.shape == (1, len(BINS)) and np.allclose(b.sum(1), 1) and 0 < ni[0] < 1


def test_fusion():
    sup = np.array([0.9, 0.05, 0.05, 0.0])
    clip = np.array([0.1, 0.1, 0.1, 0.7])
    f = fuse_bin_probs(sup, clip, 0.6)
    assert np.allclose(sup, [0.9, 0.05, 0.05, 0.0]), "inputs must not be modified"
    assert abs(f.sum() - 1) < 1e-6
    assert abs(f[3] - 0.7) < 1e-6, "hazardous prob should equal CLIP's"
    f2 = fuse_bin_probs(sup, np.array([0.8, 0.1, 0.09, 0.01]), 0.6)
    assert f2.argmax() == 0
    assert fuse_bin_probs(None, clip, 0.6).argmax() == 3
    assert fuse_bin_probs(sup, None, 0.6).argmax() == 0


def test_decide():
    f = np.array([0.7, 0.1, 0.1, 0.1])
    assert decide(f, 0.1, 0.5, 0.6)[:2] == ("ok", "recyclable")
    assert decide(f, 0.9, 0.5, 0.6)[:2] == ("no_item", None)
    assert decide(np.array([0.4, 0.3, 0.2, 0.1]), 0.0, 0.5, 0.6)[:2] == ("uncertain", "recyclable")


def test_smoother():
    s = SessionSmoother(alpha=0.5, ttl=5.0)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert (s.update("x", a, now=0.0) == a).all()
    assert np.allclose(s.update("x", b, now=1.0), [0.5, 0.5])
    assert (s.update("x", b, now=20.0) == b).all(), "expired history must be dropped"


def test_chunked_split():
    samples = [Sample(f"/d/{src}/{c}/img{i}.jpg", c, src) for src in ("a", "b") for c in range(3) for i in range(1, 101)]
    tr, va, te = split_samples(samples, mode="chunked", seed=0)
    paths = [set(s.path for s in x) for x in (tr, va, te)]
    assert not (paths[0] & paths[1]) and not (paths[0] & paths[2]) and not (paths[1] & paths[2])
    assert sum(map(len, paths)) == len(samples)
    for x in (va, te):  # every (source, class) is represented
        assert {(s.source, s.label) for s in x} == {(s.source, s.label) for s in samples}


def test_load_image_exif():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rot.jpg"
        im = Image.new("RGB", (40, 20), "red")
        exif = im.getexif()
        exif[0x0112] = 6  # orientation: rotate 90 CW
        im.save(p, exif=exif)
        out = load_image(str(p))
        assert out.size == (20, 40) and out.mode == "RGB"
        assert load_image(out).size == (20, 40), "re-loading an already loaded image must not rotate it again"
        big = load_image(Image.new("RGB", (4000, 1000)))
        assert max(big.size) == 1600


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
