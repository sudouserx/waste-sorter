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
   The weight and CLIP aggregation are **not tuned yet**: use `wastesorter.evaluate` (below) to pick them.
Works CLIP-only before you train anything.

## Results so far (first Kaggle run, chunked split, 14 epochs, 12 min on a T4)
| | material acc | macro-F1 | bin acc |
|---|---|---|---|
| test (851 imgs) | 0.953 | 0.952 | 0.971 |
| RealWaste part (550) | 0.956 | | 0.969 |
| TrashNet part (301) | 0.947 | | 0.973 |

Weakest class: `trash` (recall 0.78, mostly confused with paper/plastic). These are **in-distribution** numbers on two
curated datasets: expect noticeably lower accuracy on real phone photos. Run the notebook diagnostics
(`RUN_DIAGNOSTICS = True`) for the random-split and cross-domain numbers, and evaluate on your own photos.

## Layout
```
wastesorter/  taxonomy.py data.py train.py evaluate.py clip_zeroshot.py pipeline.py server.py predict.py
notebooks/    kaggle_run.py (source) -> kaggle_run.ipynb (python notebooks/build_notebook.py)
tests/        test_logic.py
```

## Kaggle quickstart
Push this repo to GitHub, open a new Kaggle notebook (GPU T4, Internet On), upload `notebooks/kaggle_run.ipynb`,
check `REPO_URL`, Run All. Recommended flow:
1. **Save Version -> Save & Run All** once: trains + evaluates (serving is skipped in commit runs); the checkpoint becomes the notebook output.
2. Interactive session with that output attached (*Add Input -> Your Work*): the checkpoint is reused and the API is up in minutes.

## Evaluate / tune the full pipeline
```
python -m wastesorter.evaluate --ckpt ckpt/best.pt --splits ckpt/splits.json --leakage       # held-out split
python -m wastesorter.evaluate --ckpt ckpt/best.pt --folder my_photos/                       # your own photos
```
`my_photos/` has one sub-folder per bin (`recyclable/ organic/ non_recyclable/ hazardous/ no_item/`) or per material
(`plastic/ glass/ ...`). Reports bin accuracy for classifier-only, CLIP-only (sum / max) and fused at w = 0.0 ... 1.0,
plus the share of "ok" answers, their accuracy, and false "no item" rate. 100-200 phone photos taken where the app
will be used are worth more than any number on the public datasets.

## API
```
curl -X POST $URL/predict -H "X-API-Key: $KEY" -F file=@photo.jpg -F session_id=abc
```
Response: `status` (ok | uncertain | no_item), `bin`, `confidence`, `material`, `items`, `bin_probs`, `tip`, `latency_ms`.
Live camera: send a frame every ~300-500 ms with a stable `session_id`; show the result when `status == "ok"`.
Server env: `WASTE_CKPT`, `WASTE_API_KEY`, `WASTE_USE_CLIP`, `WASTE_W_SUP`, `WASTE_CONF`, `WASTE_NO_ITEM`, `WASTE_CLIP_AGG`,
`WASTE_CORS_ORIGINS` (only for browser clients).

## Caveats
- **RealWaste is CC BY-NC-SA 4.0 (non-commercial)**; TrashNet is MIT. Check licensing before commercial release.
- TrashNet photos are clean, on plain backgrounds, with several shots per object: random splits can overstate accuracy. Compare the `per_source` metrics and the `--leakage` check.
- One item per image. Multi-object scenes need a detector (TACO-style), which is the next step.
- Bin rules differ by city/country; edit `MATERIAL_TO_BIN` and `ITEM_PROMPTS`.
- Kaggle tunnels/sessions are temporary: fine for development, not production hosting. The quick-tunnel hostname can
  take a short while to resolve; the notebook waits and retries.
