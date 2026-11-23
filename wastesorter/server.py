"""FastAPI server for the mobile app.

GET  /          web app: upload / take a photo, see the bin (open it as <url>/#key=<API key>)
POST /predict   multipart: file=<jpeg/png>, optional session_id=<str> (live-feed smoothing)
GET  /health    liveness + model info (no API key needed)
GET  /bins      bin names / display strings

Env: WASTE_CKPT, WASTE_API_KEY, WASTE_USE_CLIP (1/0), WASTE_W_SUP, WASTE_CONF, WASTE_NO_ITEM, WASTE_CLIP_AGG,
     WASTE_CORS_ORIGINS (comma list, only needed for browser clients such as Expo web; native apps ignore CORS)
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .pipeline import PipelineConfig, WastePredictor, load_image
from .taxonomy import BIN_INFO, BINS

MAX_BYTES = 10 * 1024 * 1024
UI_FILE = Path(__file__).resolve().parent / "static" / "index.html"
MAX_SESSION_ID = 128
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs early (a 10 MB file can claim huge dimensions)
state = {}
_lock = threading.Lock()  # one GPU, serialize inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = PipelineConfig(
        checkpoint=os.environ.get("WASTE_CKPT"),
        use_clip=os.environ.get("WASTE_USE_CLIP", "1") == "1",
        w_supervised=float(os.environ.get("WASTE_W_SUP", "0.6")),
        conf_threshold=float(os.environ.get("WASTE_CONF", "0.5")),
        no_item_threshold=float(os.environ.get("WASTE_NO_ITEM", "0.6")),
        clip_agg=os.environ.get("WASTE_CLIP_AGG", "sum"),
    )
    state["predictor"] = WastePredictor(cfg)
    yield
    state.clear()


app = FastAPI(title="Waste Sorter API", version="0.2.0", lifespan=lifespan)

_origins = [o.strip() for o in os.environ.get("WASTE_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                       allow_headers=["X-API-Key", "Content-Type"])


def check_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("WASTE_API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/", include_in_schema=False)
def ui():
    """The web app. It reads the API key from the link's #key=... fragment (never sent to the server in the URL)."""
    if not UI_FILE.is_file():
        raise HTTPException(status_code=404, detail="web UI not bundled")
    return FileResponse(UI_FILE, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    p = state.get("predictor")
    return {"ok": p is not None, "auth": bool(os.environ.get("WASTE_API_KEY")), "classifier": bool(p and p.model is not None),
            "clip": bool(p and p.clip is not None), "info": p.model_info if p else {}}


@app.get("/bins", dependencies=[Depends(check_key)])
def bins():
    return {b: BIN_INFO[b] for b in BINS}


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(file: UploadFile = File(...), session_id: Optional[str] = Form(default=None)):
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if session_id is not None and len(session_id) > MAX_SESSION_ID:
        raise HTTPException(status_code=422, detail=f"session_id longer than {MAX_SESSION_ID} characters")
    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image larger than 10 MB")
    try:
        img = load_image(data)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="could not decode image")
    with _lock:
        return predictor.predict(img, session_id=session_id or None)
