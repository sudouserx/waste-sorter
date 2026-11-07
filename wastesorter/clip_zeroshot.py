"""Zero-shot item recognition with CLIP (open_clip). Gives item-level hints, a 'hazardous' bin,
and a 'no item in view' gate without any training data."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .taxonomy import prompt_matrix

TEMPLATES = [
    "a photo of {}.",
    "a top-down photo of {} on a plain surface.",
    "a close-up photo of {}, a piece of rubbish.",
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


class ClipZeroShot:
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", device: Optional[str] = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.descs, self.prompt_matrix = prompt_matrix()
        embs = []
        for tpl in TEMPLATES:  # prompt ensembling: average normalized text embeddings over templates
            tokens = tokenizer([tpl.format(d) for d in self.descs]).to(self.device)
            with torch.inference_mode(), self._autocast():
                t = self.model.encode_text(tokens)
            embs.append(F.normalize(t.float(), dim=-1))
        self.text_emb = F.normalize(torch.stack(embs).mean(0), dim=-1)
        self.logit_scale = 100.0

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda")

    @torch.inference_mode()
    def predict(self, images: List[Image.Image], agg: str = "sum") -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prompt_probs [B,P], output_probs [B, len(CLIP_OUTPUTS)])."""
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._autocast():
            f = self.model.encode_image(x)
        f = F.normalize(f.float(), dim=-1)
        probs = (self.logit_scale * f @ self.text_emb.T).softmax(dim=-1)
        probs = probs.cpu().numpy()
        return probs, aggregate_prompts(probs, self.prompt_matrix, agg)