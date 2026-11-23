"""Inference pipeline: supervised material classifier + zero-shot CLIP, fused into a bin decision."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from .taxonomy import BIN_INFO, BINS, class_to_bin_matrix

HAZ = BINS.index("hazardous")


@dataclass
class PipelineConfig:
    checkpoint: Optional[str] = None     # path to best.pt; if missing -> CLIP-only mode
    use_clip: bool = True
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "laion2b_s34b_b88k"
    w_supervised: float = 0.6            # weight of the fine-tuned classifier in the fusion (tune with evaluate.py)
    conf_threshold: float = 0.5          # below this -> status "uncertain"
    no_item_threshold: float = 0.6       # CLIP "empty scene" probability above this -> "no_item"
    tta_flip: bool = True
    clip_agg: str = "sum"                # "sum" | "max" (see clip_zeroshot.aggregate_prompts)
    min_temperature: float = 1.0         # never sharpen probabilities beyond the raw model (OOD safety)
    smooth_alpha: float = 0.6            # weight of newest frame in live-feed smoothing
    smooth_ttl: float = 8.0              # seconds before a session's history expires


def load_image(src, max_side: int = 1600) -> Image.Image:
    """Path / bytes / PIL -> upright RGB image no larger than max_side. Idempotent on its own output."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")  # phones store rotation in EXIF; convert() also copies
    img.thumbnail((max_side, max_side))
    return img


def fuse_bin_probs(p_sup: Optional[np.ndarray], p_clip: Optional[np.ndarray], w: float) -> np.ndarray:
    """Fuse bin distributions. The supervised model has no 'hazardous' class, so it defers to CLIP's
    hazardous probability h and spreads the remaining (1-h) over its own bins.
    Consequence: the fused hazardous probability always equals CLIP's, whatever w is."""
    if p_sup is None and p_clip is None:
        raise ValueError("no predictions to fuse")
    if p_sup is None:
        return p_clip / p_clip.sum()
    if p_clip is None:
        return p_sup / p_sup.sum()
    h = float(p_clip[HAZ])
    sup = p_sup * (1.0 - h)   # new array, the caller's p_sup is not modified
    sup[HAZ] = h
    fused = w * sup + (1.0 - w) * p_clip
    return fused / fused.sum()


def decide(fused: np.ndarray, no_item: float, conf_threshold: float, no_item_threshold: float
           ) -> Tuple[str, Optional[str], float]:
    """-> (status, bin or None, confidence of the top bin)."""
    top = int(np.argmax(fused))
    conf = float(fused[top])
    if no_item >= no_item_threshold:
        return "no_item", None, conf
    if conf < conf_threshold:
        return "uncertain", BINS[top], conf
    return "ok", BINS[top], conf


class SessionSmoother:
    """EMA over per-frame probability vectors so a live camera feed doesn't flicker."""

    def __init__(self, alpha: float = 0.6, ttl: float = 8.0, max_sessions: int = 1000):
        self.alpha, self.ttl, self.max_sessions = alpha, ttl, max_sessions
        self._d: Dict[str, Tuple[float, np.ndarray]] = {}

    def update(self, sid: str, vec: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        now = time.monotonic() if now is None else now
        self._d = {k: v for k, v in self._d.items() if now - v[0] <= self.ttl}
        if len(self._d) >= self.max_sessions and sid not in self._d:
            oldest = min(self._d, key=lambda k: self._d[k][0])
            self._d.pop(oldest)
        prev = self._d.get(sid)
        if prev is not None and prev[1].shape == vec.shape:
            vec = self.alpha * vec + (1.0 - self.alpha) * prev[1]
        self._d[sid] = (now, vec)
        return vec


def _load_checkpoint(path: str) -> dict:
    try:  # the checkpoint only holds tensors and plain Python types, so the safe loader works
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu")


class WastePredictor:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.clip = None
        self.classes: List[str] = []
        self.model_info: Dict = {}
        self.smoother = SessionSmoother(self.cfg.smooth_alpha, self.cfg.smooth_ttl)

        ck = self.cfg.checkpoint
        if ck and Path(ck).is_file():
            self._load_classifier(ck)
        elif ck:
            print(f"[warn] checkpoint not found at {ck}; running CLIP-only")
        if self.cfg.use_clip:
            from .clip_zeroshot import ClipZeroShot

            self.clip = ClipZeroShot(self.cfg.clip_model, self.cfg.clip_pretrained)
        if self.model is None and self.clip is None:
            raise RuntimeError("Need a trained checkpoint and/or use_clip=True")
        self.model_info.update({"clip": self.cfg.clip_model if self.clip else None, "w_supervised": self.cfg.w_supervised,
                                "clip_agg": self.cfg.clip_agg, "conf_threshold": self.cfg.conf_threshold,
                                "no_item_threshold": self.cfg.no_item_threshold})

    def _load_classifier(self, path: str):
        import timm
        import torchvision.transforms as T

        ck = _load_checkpoint(path)
        self.classes = list(ck["classes"])
        m = timm.create_model(ck["model_name"], pretrained=False, num_classes=len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.eval().to(self.device).to(memory_format=torch.channels_last)
        fitted = float(ck.get("temperature", 1.0))
        self.temperature = max(self.cfg.min_temperature, fitted)
        self.class_to_bin = class_to_bin_matrix(self.classes)
        s = int(ck["img_size"])
        self.tf = T.Compose([
            T.Resize((s, s), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(ck["mean"], ck["std"]),
        ])
        self.model_info = {"backbone": ck["model_name"], "img_size": s, "temperature": self.temperature,
                           "fitted_temperature": fitted, "test_metrics": ck.get("metrics", {})}

    @torch.inference_mode()
    def _classifier_probs(self, imgs: List[Image.Image]) -> np.ndarray:
        x = torch.stack([self.tf(im) for im in imgs]).to(self.device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            lo = self.model(x).float()
            if self.cfg.tta_flip:
                lo = (lo + self.model(torch.flip(x, dims=[3])).float()) / 2
        return torch.softmax(lo / self.temperature, dim=-1).cpu().numpy()

    def components(self, imgs: List[Image.Image]) -> Dict[str, Optional[np.ndarray]]:
        """Raw per-model outputs for a batch of already-loaded images (used by predict and evaluate.py).
        mat_probs [B,C] / sup_bins [B,len(BINS)] from the classifier, prompt_probs [B,P] from CLIP."""
        out: Dict[str, Optional[np.ndarray]] = {"mat_probs": None, "sup_bins": None, "prompt_probs": None}
        if self.model is not None:
            out["mat_probs"] = self._classifier_probs(imgs)
            out["sup_bins"] = out["mat_probs"] @ self.class_to_bin
        if self.clip is not None:
            out["prompt_probs"] = self.clip.prompt_probs(imgs)
        return out

    def predict(self, image, session_id: Optional[str] = None) -> dict:
        t0 = time.perf_counter()
        img = load_image(image)
        c = self.components([img])

        p_sup_bins, materials = None, []
        if c["mat_probs"] is not None:
            probs = c["mat_probs"][0]
            p_sup_bins = c["sup_bins"][0]
            materials = [{"label": self.classes[i], "prob": round(float(probs[i]), 4)} for i in np.argsort(-probs)[:3]]

        p_clip_bins, no_item, items = None, 0.0, []
        if c["prompt_probs"] is not None:
            from .clip_zeroshot import clip_bins

            pp = c["prompt_probs"][0]
            b, ni = clip_bins(pp, self.clip.prompt_matrix, self.cfg.clip_agg)
            p_clip_bins, no_item = b[0], float(ni[0])
            items = [{"label": self.clip.descs[i], "prob": round(float(pp[i]), 4)} for i in np.argsort(-pp)[:3]]

        fused = fuse_bin_probs(p_sup_bins, p_clip_bins, self.cfg.w_supervised)
        if session_id:
            vec = self.smoother.update(session_id, np.concatenate([fused, [no_item]]))
            fused, no_item = vec[:-1], float(vec[-1])

        status, bin_name, conf = decide(fused, no_item, self.cfg.conf_threshold, self.cfg.no_item_threshold)
        info = BIN_INFO.get(bin_name, {}) if bin_name else {}
        return {
            "status": status,                       # ok | uncertain | no_item
            "bin": bin_name,                        # recyclable | organic | non_recyclable | hazardous | None
            "bin_display": info.get("display"),
            "tip": info.get("tip"),
            "confidence": round(conf, 4),
            "material": materials[0]["label"] if materials else None,
            "material_confidence": materials[0]["prob"] if materials else None,
            "materials": materials,
            "items": items,                         # CLIP's top item guesses
            "bin_probs": {b: round(float(p), 4) for b, p in zip(BINS, fused)},
            "no_item_prob": round(no_item, 4),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
