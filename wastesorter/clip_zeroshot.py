"""Zero-shot item recognition with CLIP (open_clip). Gives item-level hints, a 'hazardous' bin,
and a 'no item in view' gate without any training data."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .taxonomy import ITEM_PROMPTS, NO_ITEM, prompt_matrix

# Waste-item prompts. The "hand holding" template matches how people point a phone at an item.
ITEM_TEMPLATES = [
    "a photo of {}.",
    "a top-down photo of {} on a plain surface.",
    "a close-up photo of {}, a piece of rubbish.",
    "a photo of a hand holding {}.",
]
# "No item" prompts describe scenes; "an empty table, a piece of rubbish" would contradict itself.
SCENE_TEMPLATES = [
    "a photo of {}.",
    "a phone camera photo of {}.",
    "a blurry photo of {}.",
]


def aggregate_prompts(probs: np.ndarray, pm: np.ndarray, mode: str = "sum") -> np.ndarray:
    """[B,P] prompt probs -> [B,K] per-output scores (rows sum to 1).
    "sum" is the marginal probability but is biased toward outputs that own more prompts;
    "max" takes each output's best prompt, which has no prompt-count bias."""
    if mode == "sum":
        out = probs @ pm
    elif mode == "max":
        out = np.max(probs[:, :, None] * pm[None, :, :], axis=1)
    else:
        raise ValueError(f"unknown aggregation: {mode}")
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-8)


def clip_bins(prompt_probs: np.ndarray, pm: np.ndarray, mode: str = "sum") -> Tuple[np.ndarray, np.ndarray]:
    """[B,P] prompt probs -> (bin probs [B, len(BINS)] renormalised without no_item, no_item prob [B])."""
    out = aggregate_prompts(np.atleast_2d(prompt_probs), pm, mode)
    no_item = out[:, -1]
    b = out[:, :-1]
    return b / np.maximum(b.sum(axis=1, keepdims=True), 1e-8), no_item


class ClipZeroShot:
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", device: Optional[str] = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained or None)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.descs, self.prompt_matrix = prompt_matrix()

        def encode(texts):
            with torch.inference_mode(), self._autocast():
                t = self.model.encode_text(tokenizer(texts).to(self.device))
            return F.normalize(t.float(), dim=-1)

        # Prompt ensembling: average normalised text embeddings over each prompt's own template set.
        groups = [
            ([i for i, (_, t) in enumerate(ITEM_PROMPTS) if t != NO_ITEM], ITEM_TEMPLATES),
            ([i for i, (_, t) in enumerate(ITEM_PROMPTS) if t == NO_ITEM], SCENE_TEMPLATES),
        ]
        emb = None
        for idxs, templates in groups:
            if not idxs:
                continue
            acc = sum(encode([tpl.format(self.descs[i]) for i in idxs]) for tpl in templates)
            if emb is None:
                emb = torch.zeros(len(self.descs), acc.shape[1], device=self.device)
            emb[idxs] = F.normalize(acc, dim=-1)
        self.text_emb = emb
        self.logit_scale = 100.0

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda")

    @torch.inference_mode()
    def prompt_probs(self, images: List[Image.Image]) -> np.ndarray:
        """[B,P] softmax over all item/scene prompts."""
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._autocast():
            f = self.model.encode_image(x)
        f = F.normalize(f.float(), dim=-1)
        return (self.logit_scale * f @ self.text_emb.T).softmax(dim=-1).cpu().numpy()

    def predict(self, images: List[Image.Image], agg: str = "sum") -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prompt_probs [B,P], output_probs [B, len(CLIP_OUTPUTS)])."""
        probs = self.prompt_probs(images)
        return probs, aggregate_prompts(probs, self.prompt_matrix, agg)
