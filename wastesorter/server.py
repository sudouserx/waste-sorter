"""FastAPI server for the mobile app.

POST /predict   multipart: file=<jpeg/png>, optional session_id=<str> (live-feed smoothing)
GET  /health    liveness + model info
GET  /bins      bin names / display strings

Env: WASTE_CKPT, WASTE_API_KEY, WASTE_USE_CLIP (1/0), WASTE_W_SUP, WASTE_CONF
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .pipeline import PipelineConfig, WastePredictor, load_image
from .taxonomy import BIN_INFO, BINS

MAX_BYTES = 10 * 1024 * 1024
state = {}
_lock = threading.Lock()  # one GPU, serialize inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = PipelineConfig(
        checkpoint=os.environ.get("WASTE_CKPT"),
        use_clip=os.environ.get("WASTE_USE_CLIP", "1") == "1",
        w_supervised=float(os.environ.get("WASTE_W_SUP", "0.6")),
        conf_threshold=float(os.environ.get("WASTE_CONF", "0.5")),
    )
    state["predictor"] = WastePredictor(cfg)
    yield
    state.clear()


app = FastAPI(title="Waste Sorter API", version="0.1.0", lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("WASTE_API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health():
    p = state.get("predictor")
    return {"ok": p is not None, "classifier": bool(p and p.model is not None),
            "clip": bool(p and p.clip is not None), "info": p.model_info if p else {}}


@app.get("/bins", dependencies=[Depends(check_key)])
def bins():
    return {b: BIN_INFO[b] for b in BINS}


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(file: UploadFile = File(...), session_id: Optional[str] = Form(default=None)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image larger than 10 MB")
    try:
        img = load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="could not decode image")
    with _lock:
        return state["predictor"].predict(img, session_id=session_id)
