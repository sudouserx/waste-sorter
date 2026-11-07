"""Zero-shot item recognition with CLIP (open_clip). Gives item-level hints, a 'hazardous' bin,
and a 'no item in view' gate without any training data."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .taxonomy import prompt_matrix

TEMPLATE = "a photo of {}."


class ClipZeroShot:
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", device: Optional[str] = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.descs, self.prompt_matrix = prompt_matrix()
        tokens = tokenizer([TEMPLATE.format(d) for d in self.descs]).to(self.device)
        with torch.inference_mode(), self._autocast():
            t = self.model.encode_text(tokens)
        self.text_emb = F.normalize(t.float(), dim=-1)
        self.logit_scale = 100.0
        self.pm = torch.from_numpy(self.prompt_matrix).to(self.device)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda")

    @torch.inference_mode()
    def predict(self, images: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prompt_probs [B,P], output_probs [B, len(CLIP_OUTPUTS)])."""
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._autocast():
            f = self.model.encode_image(x)
        f = F.normalize(f.float(), dim=-1)
        probs = (self.logit_scale * f @ self.text_emb.T).softmax(dim=-1)
        out = probs @ self.pm
        return probs.cpu().numpy(), out.cpu().numpy()
