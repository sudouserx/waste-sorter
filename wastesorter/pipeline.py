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
    w_supervised: float = 0.6            # weight of the fine-tuned classifier in the fusion
    conf_threshold: float = 0.5          # below this -> status "uncertain"
    no_item_threshold: float = 0.6       # CLIP "empty scene" probability above this -> "no_item"
    tta_flip: bool = True
    clip_agg: str = "sum"               # "sum" | "max" (see clip_zeroshot.aggregate_prompts)
    min_temperature: float = 1.0         # never sharpen probabilities beyond the raw model (OOD safety)
    smooth_alpha: float = 0.6            # weight of newest frame in live-feed smoothing
    smooth_ttl: float = 8.0              # seconds before a session's history expires


def load_image(src) -> Image.Image:
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")  # phones store rotation in EXIF
    img.thumbnail((1600, 1600))
    return img


def fuse_bin_probs(p_sup: Optional[np.ndarray], p_clip: Optional[np.ndarray], w: float) -> np.ndarray:
    """Fuse bin distributions. The supervised model has no 'hazardous' class, so it defers to CLIP's
    hazardous probability h and spreads the remaining (1-h) over its own bins."""
    if p_sup is None and p_clip is None:
        raise ValueError("no predictions to fuse")
    if p_sup is None:
        return p_clip / p_clip.sum()
    if p_clip is None:
        return p_sup / p_sup.sum()
    h = float(p_clip[HAZ])
    sup = p_sup * (1.0 - h)
    sup[HAZ] = h
    fused = w * sup + (1.0 - w) * p_clip
    return fused / fused.sum()


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
        if prev is not None:
            vec = self.alpha * vec + (1.0 - self.alpha) * prev[1]
        self._d[sid] = (now, vec)
        return vec


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

    def _load_classifier(self, path: str):
        import timm
        import torchvision.transforms as T

        ck = torch.load(path, map_location="cpu")
        self.classes = list(ck["classes"])
        m = timm.create_model(ck["model_name"], pretrained=False, num_classes=len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.eval().to(self.device).to(memory_format=torch.channels_last)
        self.temperature = max(self.cfg.min_temperature, float(ck.get("temperature", 1.0)))
        self.class_to_bin = class_to_bin_matrix(self.classes)
        s = int(ck["img_size"])
        self.tf = T.Compose([
            T.Resize((s, s), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(ck["mean"], ck["std"]),
        ])
        self.model_info = {"backbone": ck["model_name"], "img_size": s, "temperature": self.temperature,
                           "test_metrics": ck.get("metrics", {})}

    @torch.inference_mode()
    def _classifier_probs(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img).unsqueeze(0).to(self.device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            lo = self.model(x).float()
            if self.cfg.tta_flip:
                lo = (lo + self.model(torch.flip(x, dims=[3])).float()) / 2
        return torch.softmax(lo / self.temperature, dim=-1)[0].cpu().numpy()

    def predict(self, image, session_id: Optional[str] = None) -> dict:
        t0 = time.perf_counter()
        img = load_image(image)

        p_sup_bins, materials = None, []
        if self.model is not None:
            probs = self._classifier_probs(img)
            p_sup_bins = probs @ self.class_to_bin
            top = np.argsort(-probs)[:3]
            materials = [{"label": self.classes[i], "prob": round(float(probs[i]), 4)} for i in top]

        p_clip_bins, no_item, items = None, 0.0, []
        if self.clip is not None:
            item_probs, out_probs = self.clip.predict([img], agg=self.cfg.clip_agg)
            item_probs, out = item_probs[0], out_probs[0]
            no_item = float(out[-1])
            b = out[:-1]
            p_clip_bins = b / max(float(b.sum()), 1e-8)
            for i in np.argsort(-item_probs)[:3]:
                items.append({"label": self.clip.descs[i], "prob": round(float(item_probs[i]), 4)})

        fused = fuse_bin_probs(p_sup_bins, p_clip_bins, self.cfg.w_supervised)
        vec = np.concatenate([fused, [no_item]])
        if session_id:
            vec = self.smoother.update(session_id, vec)
        fused, no_item = vec[:-1], float(vec[-1])

        top_bin = int(np.argmax(fused))
        conf = float(fused[top_bin])
        if no_item >= self.cfg.no_item_threshold:
            status, bin_name = "no_item", None
        elif conf < self.cfg.conf_threshold:
            status, bin_name = "uncertain", BINS[top_bin]
        else:
            status, bin_name = "ok", BINS[top_bin]

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