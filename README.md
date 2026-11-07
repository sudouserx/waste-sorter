# Waste Sorter MVP

Photo / camera frame -> which bin: **recyclable**, **organic (compost)**, **non_recyclable**, or **hazardous (special drop-off)**.
Runs on a free Kaggle T4 (16 GB). One FastAPI endpoint for the mobile app.

## Method (hybrid)
1. **Supervised material classifier**: `convnext_tiny` (ImageNet-22k pretrained, via timm), fine-tuned at 288 px with fp16 AMP on
   TrashNet + RealWaste merged into 8 materials (cardboard, glass, metal, organic, paper, plastic, textile, trash).
   Material -> bin mapping lives in `wastesorter/taxonomy.py`.
2. **Zero-shot CLIP** (open_clip ViT-B/16): ~60 item prompts mapped to bins. Adds item hints ("a plastic bottle"),
   a **hazardous/e-waste** bin (no training data needed) and a **no item in view** gate for live camera mode.
3. **Fusion**: weighted average of bin probabilities (default 0.6 supervised / 0.4 CLIP); hazardous probability comes from CLIP.
   Low confidence -> `status: "uncertain"`. Optional `session_id` smooths a live feed over time.
Works CLIP-only before you train anything.

## Layout
```
wastesorter/  taxonomy.py data.py train.py clip_zeroshot.py pipeline.py server.py predict.py
notebooks/    kaggle_run.ipynb (+ kaggle_run.py)
tests/        test_logic.py
```

## Kaggle quickstart
Push this repo to GitHub, open a new Kaggle notebook (GPU T4, Internet On), upload `notebooks/kaggle_run.ipynb`, set `REPO_URL`, Run All.

## API
```
curl -X POST $URL/predict -H "X-API-Key: $KEY" -F file=@photo.jpg -F session_id=abc
```
Response: `status` (ok | uncertain | no_item), `bin`, `confidence`, `material`, `items`, `bin_probs`, `tip`, `latency_ms`.
Live camera: send a frame every ~300-500 ms with a stable `session_id`; show the result when `status == "ok"`.

## Caveats
- **RealWaste is CC BY-NC-SA 4.0 (non-commercial)**; TrashNet is MIT. Check licensing before commercial release.
- TrashNet photos are clean, on plain backgrounds, with several shots per object: random splits can overstate accuracy. Compare the `per_source` metrics.
- One item per image. Multi-object scenes need a detector (TACO-style), which is the next step.
- Bin rules differ by city/country; edit `MATERIAL_TO_BIN` and `ITEM_PROMPTS`.
- Kaggle tunnels/sessions are temporary: fine for development, not production hosting.
