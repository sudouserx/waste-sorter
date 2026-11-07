# Kaggle notebook script (jupytext-style cells). Lines starting with '!' are notebook shell magics.
# %% [markdown]
# # Waste Sorter MVP: train + serve on Kaggle T4
# **Before running:** Settings -> Accelerator = *GPU T4* (x1 or x2, only one is used), Internet = *On*.
# Edit `REPO_URL` in the first code cell. Optional Kaggle Secrets: `WASTE_API_KEY`.

# %%
REPO_URL = "https://github.com/YOUR_USER/waste-sorter.git"   # <-- edit
BRANCH = "main"

import os, subprocess, sys
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/waste-sorter"
CKPT_DIR = f"{WORK}/ckpt"
DATA_DIR = f"{WORK}/data"

if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(os.listdir(REPO_DIR))

# %%
!pip install -q timm open_clip_torch fastapi uvicorn python-multipart
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())

# %%
!python tests/test_logic.py

# %% [markdown]
# ## Train (about 25-40 min on a T4 with the defaults; downloads TrashNet from HF and RealWaste from GitHub)

# %%
!python -m wastesorter.train --out {CKPT_DIR} --data-dir {DATA_DIR}

# %% [markdown]
# Quick debug run (a few minutes, no pretrained weights, tiny subset): `!python -m wastesorter.train --out /kaggle/working/dbg --max-per-group 20 --epochs 2 --head-epochs 1 --no-pretrained --model resnet18`

# %%
import json
from IPython.display import Image as IPImage, display
m = json.load(open(f"{CKPT_DIR}/metrics.json"))
print({k: m[k] for k in ["n", "accuracy", "macro_f1", "bin_accuracy", "temperature"]})
print(json.dumps(m["per_source"], indent=2))
display(IPImage(f"{CKPT_DIR}/confusion_matrix.png"))

# %% [markdown]
# ## Demo: full pipeline (classifier + CLIP) on random test images

# %%
import json, random
from PIL import Image
import matplotlib.pyplot as plt
from wastesorter.pipeline import WastePredictor, PipelineConfig
from wastesorter.taxonomy import MATERIALS

predictor = WastePredictor(PipelineConfig(checkpoint=f"{CKPT_DIR}/best.pt"))
test = json.load(open(f"{CKPT_DIR}/splits.json"))["test"]
pick = random.Random(0).sample(test, 12)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, t in zip(axes.ravel(), pick):
    r = predictor.predict(Image.open(t["path"]))
    ax.imshow(Image.open(t["path"])); ax.axis("off")
    ax.set_title(f"{r['status']}: {r['bin']} ({r['confidence']:.2f})\n"
                 f"model: {r['material']} | truth: {MATERIALS[t['label']]} [{t['source']}]\n"
                 f"CLIP: {r['items'][0]['label'] if r['items'] else '-'}", fontsize=9)
plt.tight_layout(); plt.show()
print("latency ms:", r["latency_ms"])

# %% [markdown]
# ## Serve the API + public tunnel (for the mobile app)
# Starts uvicorn on :8000 and a Cloudflare quick tunnel (no account needed). The URL changes every run, and the Kaggle session is temporary, so this is for development and demos only.

# %%
import secrets, time, re, requests, subprocess, sys, os
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("WASTE_API_KEY")
except Exception:
    API_KEY = secrets.token_urlsafe(16)
print("X-API-Key:", API_KEY)

env = {**os.environ, "WASTE_CKPT": f"{CKPT_DIR}/best.pt", "WASTE_API_KEY": API_KEY}
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "wastesorter.server:app", "--host", "0.0.0.0", "--port", "8000"],
                          cwd=REPO_DIR, env=env, stdout=open(f"{WORK}/server.log", "w"), stderr=subprocess.STDOUT)
for _ in range(120):
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=2).ok: break
    except Exception: time.sleep(2)
print(requests.get("http://127.0.0.1:8000/health").json())

# %%
!wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared
tunnel_log = open(f"{WORK}/tunnel.log", "w")
tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                          stdout=tunnel_log, stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(60):
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open(f"{WORK}/tunnel.log").read())
    if m: PUBLIC_URL = m.group(0); break
    time.sleep(1)
print("PUBLIC_URL:", PUBLIC_URL)

# %%
# Test through the public URL exactly like the mobile app would
test_img = test[0]["path"]
with open(test_img, "rb") as f:
    r = requests.post(f"{PUBLIC_URL}/predict", headers={"X-API-Key": API_KEY},
                      files={"file": ("photo.jpg", f, "image/jpeg")}, data={"session_id": "demo"}, timeout=60)
print(r.status_code, json.dumps(r.json(), indent=2))

# %%
# Keep the session alive while you test from the app (interrupt the kernel to stop)
try:
    while True: time.sleep(60)
except KeyboardInterrupt:
    server.terminate(); tunnel.terminate()

