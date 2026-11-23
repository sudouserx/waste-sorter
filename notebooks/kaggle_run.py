# Kaggle notebook script (jupytext-style cells). Lines starting with '!' are notebook shell magics.
# kaggle_run.ipynb is generated from this file: python notebooks/build_notebook.py
# %% [markdown]
# # Waste Sorter MVP: train + serve on Kaggle T4
# **Before running:** Settings -> Accelerator = *GPU T4* (x1 or x2, only one is used), Internet = *On*.
# Optional Kaggle Secrets: `WASTE_API_KEY` (fixed key for the app), `HF_TOKEN` (silences the HF rate-limit warning).
#
# **Two ways to run**
# * **Interactive session -> Run All**: trains (or reuses an attached checkpoint), evaluates, then serves the API
#   through a Cloudflare quick tunnel for `KEEPALIVE_MIN` minutes. Keep the browser tab open while testing.
# * **Save Version -> Save & Run All (commit)**: trains and evaluates only; serving is skipped automatically.
#   `/kaggle/working/ckpt` becomes the notebook output. In the next interactive session, *Add Input -> Your Work*
#   -> this notebook, and the checkpoint is reused, so the API is up in a few minutes instead of ~15.

# %%
REPO_URL = "https://github.com/sudouserx/waste-sorter.git"
BRANCH = "main"
FORCE_TRAIN = False        # True: retrain even if a finished checkpoint exists / is attached as input
RUN_DIAGNOSTICS = False    # True: two extra ~10 min runs (random split, cross-domain) to see how optimistic the test score is
FIELD_DIR = None           # optional: folder of your own photos with sub-folders recyclable/ organic/ non_recyclable/ hazardous/ no_item/
KEEPALIVE_MIN = 90         # minutes the API stays up in the last cell

import glob, os, shutil, subprocess, sys
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/waste-sorter"
CKPT_DIR = f"{WORK}/ckpt"
DATA_DIR = f"{WORK}/data"
CKPT = f"{CKPT_DIR}/best.pt"
IS_BATCH = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive") == "Batch"   # "Batch" = Save Version run
print("run type:", "commit (no serving)" if IS_BATCH else "interactive")


def run(cmd, cwd=None):
    """Run a command and stream its output into the notebook (plain subprocess output would not show up)."""
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed (exit {p.returncode}): {' '.join(map(str, cmd))}")


if os.path.exists(REPO_DIR):
    run(["git", "-C", REPO_DIR, "pull", "--ff-only"])
else:
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(os.listdir(REPO_DIR))

try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN"))
except Exception:
    pass  # no HF_TOKEN secret: downloads still work, just with a warning

# %%
!pip install -q timm open_clip_torch fastapi uvicorn python-multipart
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())

# %%
!python tests/test_logic.py

# %% [markdown]
# ## Train, or reuse a finished checkpoint
# About 12-15 min on a T4 with the defaults (the first run took 12.1 min for 14 epochs). Downloads TrashNet from
# the HF Hub and RealWaste from GitHub. A checkpoint counts as finished when `metrics.json` sits next to `best.pt`.

# %%
def finished(d):
    return os.path.exists(f"{d}/best.pt") and os.path.exists(f"{d}/metrics.json")

if not FORCE_TRAIN and not finished(CKPT_DIR):
    for p in sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)):
        if finished(os.path.dirname(p)):
            shutil.copytree(os.path.dirname(p), CKPT_DIR, dirs_exist_ok=True)
            print("reusing checkpoint from", os.path.dirname(p))
            break

if FORCE_TRAIN or not finished(CKPT_DIR):
    run([sys.executable, "-m", "wastesorter.train", "--out", CKPT_DIR, "--data-dir", DATA_DIR], cwd=REPO_DIR)
else:
    print("finished checkpoint present, skipping training:", CKPT)

# The images are needed for evaluation and the demo (a reused checkpoint arrives without them). No-op if present.
run([sys.executable, "-c", f"from wastesorter.data import get_samples; print(len(get_samples({DATA_DIR!r})), 'images ready')"],
    cwd=REPO_DIR)

# %% [markdown]
# Quick debug run (a few minutes, no pretrained weights, tiny subset): `!python -m wastesorter.train --out /kaggle/working/dbg --max-per-group 20 --epochs 2 --head-epochs 1 --no-pretrained --model resnet18`

# %%
import json
from IPython.display import Image as IPImage, display
m = json.load(open(f"{CKPT_DIR}/metrics.json"))
print({k: m.get(k) for k in ["n", "accuracy", "macro_f1", "bin_accuracy", "temperature", "best_epoch"]})
print(json.dumps(m["per_source"], indent=2))
if os.path.exists(f"{CKPT_DIR}/confusion_matrix.png"):
    display(IPImage(f"{CKPT_DIR}/confusion_matrix.png"))

# %% [markdown]
# ## Optional diagnostics: how optimistic is the test score?
# * **random split** vs the default **chunked** split: a big gap means near-duplicate photos leak across splits.
# * **cross-domain**: train on RealWaste only, test on TrashNet. The closest proxy here for photos from a new
#   setting (like phone photos), and usually much lower than the in-distribution number.

# %%
if RUN_DIAGNOSTICS:
    diag = {}
    for name, extra in [("random_split", ["--split-mode", "random"]),
                        ("cross_rw_to_tn", ["--train-sources", "realwaste", "--cross-eval-source", "trashnet"])]:
        out = f"{WORK}/diag_{name}"
        run([sys.executable, "-m", "wastesorter.train", "--out", out, "--data-dir", DATA_DIR, "--epochs", "10", *extra],
            cwd=REPO_DIR)
        diag[name] = json.load(open(f"{out}/metrics.json"))
    print(f"{'run':<16}{'test acc':>10}{'bin acc':>10}{'cross acc':>11}{'cross bin':>11}")
    for name, mm in [("chunked (main)", m), *diag.items()]:
        cd = mm.get("cross_domain", {})
        print(f"{name:<16}{mm['accuracy']:>10.3f}{mm['bin_accuracy']:>10.3f}"
              f"{cd.get('accuracy', float('nan')):>11.3f}{cd.get('bin_accuracy', float('nan')):>11.3f}")
else:
    print("RUN_DIAGNOSTICS is False: skipped")

# %% [markdown]
# ## Evaluate the full pipeline (classifier + CLIP) at bin level
# train.py only scores the classifier. This sweeps the fusion weight and CLIP aggregation, reports how often the app
# would answer "ok" and how accurate those answers are, and checks test images for near-duplicates in train.
# Set `WASTE_W_SUP` / `WASTE_CLIP_AGG` for the server from what you see here, ideally from your own photos (`FIELD_DIR`).

# %%
cmd = [sys.executable, "-m", "wastesorter.evaluate", "--ckpt", CKPT, "--splits", f"{CKPT_DIR}/splits.json",
       "--leakage", "--out", f"{CKPT_DIR}/eval.json"]
if FIELD_DIR:
    cmd += ["--folder", FIELD_DIR]
run(cmd, cwd=REPO_DIR)

# %% [markdown]
# ## Demo: full pipeline on random test images

# %%
import gc, random
import matplotlib.pyplot as plt
from wastesorter.pipeline import WastePredictor, PipelineConfig, load_image
from wastesorter.taxonomy import MATERIALS

predictor = WastePredictor(PipelineConfig(checkpoint=CKPT))
test = json.load(open(f"{CKPT_DIR}/splits.json"))["test"]
pick = random.Random(0).sample(test, 12)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
lat = []
for ax, t in zip(axes.ravel(), pick):
    img = load_image(t["path"])
    r = predictor.predict(img)
    lat.append(r["latency_ms"])
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{r['status']}: {r['bin']} ({r['confidence']:.2f})\n"
                 f"model: {r['material']} | truth: {MATERIALS[t['label']]} [{t['source']}]\n"
                 f"CLIP: {r['items'][0]['label'] if r['items'] else '-'}", fontsize=9)
plt.tight_layout(); plt.show()
print("latency ms (median of 12, first call includes warm-up):", sorted(lat)[len(lat) // 2])

# The server loads its own copy of the models; free this one.
del predictor; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Serve the web app + API through a public link (interactive sessions only)
# Starts uvicorn on :8000, checks it locally, then opens a Cloudflare quick tunnel (no account needed).
# The public link opens a small web app (upload or take a photo -> which bin). The same server still exposes
# `POST /predict` for the mobile app. The printed link carries the API key after `#`, which browsers never send
# to the server; the page stores it and removes it from the address bar.
# The URL changes on every run and the Kaggle session is temporary: development and demos only.
#
# Why the first run failed: the old cell read the URL from cloudflared's log as soon as it was printed, which is
# *before* the tunnel registers with Cloudflare, and then requested it immediately. Kaggle's DNS resolver could not
# resolve the brand-new hostname yet (`NameResolutionError`). The server itself was healthy. The cells below wait for
# `Registered tunnel connection`, retry, and fall back to DNS-over-HTTPS + `curl --resolve` if Kaggle's resolver
# is still behind. That fallback only affects this notebook check; phones use their own DNS.

# %%
import json, re, secrets, socket, time
import requests

if IS_BATCH:
    print("commit run: skipping the API server")
else:
    try:
        from kaggle_secrets import UserSecretsClient
        API_KEY = UserSecretsClient().get_secret("WASTE_API_KEY")
    except Exception:
        API_KEY = secrets.token_urlsafe(16)
    print("X-API-Key:", API_KEY, "(don't publish this notebook's output with a fixed key in it)")

    env = {**os.environ, "WASTE_CKPT": CKPT, "WASTE_API_KEY": API_KEY}
    SERVER_LOG = f"{WORK}/server.log"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "wastesorter.server:app", "--host", "0.0.0.0", "--port", "8000"],
                              cwd=REPO_DIR, env=env, stdout=open(SERVER_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(150):
        if server.poll() is not None:
            raise RuntimeError("server exited during startup:\n" + open(SERVER_LOG).read()[-3000:])
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("server not healthy after 5 min:\n" + open(SERVER_LOG).read()[-3000:])
    print(json.dumps(requests.get("http://127.0.0.1:8000/health").json(), indent=2))

    # End-to-end request against the local server: proves the API works independently of the tunnel.
    TEST_IMG = json.load(open(f"{CKPT_DIR}/splits.json"))["test"][0]["path"]
    with open(TEST_IMG, "rb") as f:
        r = requests.post("http://127.0.0.1:8000/predict", headers={"X-API-Key": API_KEY},
                          files={"file": ("photo.jpg", f, "image/jpeg")}, data={"session_id": "local"}, timeout=60)
    print("local /predict:", r.status_code, json.dumps(r.json(), indent=2))
    ui = requests.get("http://127.0.0.1:8000/", timeout=10)
    print("local web app:", ui.status_code, ui.headers.get("content-type"))

# %%
if not IS_BATCH:
    if not os.path.exists("/tmp/cloudflared"):
        run(["wget", "-q", "-O", "/tmp/cloudflared",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
        os.chmod("/tmp/cloudflared", 0o755)
    TUNNEL_LOG = f"{WORK}/tunnel.log"
    # 127.0.0.1, not "localhost": localhost may resolve to IPv6 ::1, where uvicorn (bound to 0.0.0.0) isn't listening.
    tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
                              stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)
    PUBLIC_URL, registered = None, False
    for _ in range(90):
        log = open(TUNNEL_LOG).read()
        m_url = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", log)
        PUBLIC_URL = m_url.group(0) if m_url else None
        registered = "Registered tunnel connection" in log
        if PUBLIC_URL and registered:
            break
        if tunnel.poll() is not None:
            raise RuntimeError("cloudflared exited:\n" + log[-3000:])
        time.sleep(1)
    print("PUBLIC_URL:", PUBLIC_URL, "| tunnel registered:", registered)
    if not registered:
        print(open(TUNNEL_LOG).read()[-2000:])

# %%
# Test through the public URL like the mobile app would, with retries while DNS propagates.
from urllib.parse import urlparse

def doh_resolve(host):
    """Resolve via Cloudflare DNS-over-HTTPS, bypassing Kaggle's resolver (and its negative cache)."""
    try:
        r = requests.get("https://cloudflare-dns.com/dns-query", params={"name": host, "type": "A"},
                         headers={"accept": "application/dns-json"}, timeout=10)
        return [a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1]
    except Exception:
        return []

def post_public(host, ip=None):
    """POST the test image to the public URL; with ip, pin the hostname to that IP via curl --resolve."""
    if ip is None:
        try:
            with open(TEST_IMG, "rb") as f:
                r = requests.post(f"{PUBLIC_URL}/predict", headers={"X-API-Key": API_KEY},
                                  files={"file": ("photo.jpg", f, "image/jpeg")}, data={"session_id": "demo"}, timeout=60)
            return r.status_code, r.text
        except requests.RequestException as e:
            return 0, str(e)
    out = subprocess.run(["curl", "-sS", "--max-time", "60", "--resolve", f"{host}:443:{ip}",
                          "-H", f"X-API-Key: {API_KEY}", "-F", f"file=@{TEST_IMG}", "-F", "session_id=demo",
                          "-w", "\n%{http_code}", f"{PUBLIC_URL}/predict"], capture_output=True, text=True)
    body, _, code = out.stdout.rpartition("\n")
    return (int(code) if code.isdigit() else 0), (body or out.stderr)

if not IS_BATCH and PUBLIC_URL:
    host = urlparse(PUBLIC_URL).hostname
    via, code, body = "-", 0, ""
    deadline = time.time() + 150
    while time.time() < deadline:
        try:
            socket.getaddrinfo(host, 443)
            via, (code, body) = "system DNS", post_public(host)
        except socket.gaierror:
            ips = doh_resolve(host)
            if ips:
                via, (code, body) = f"DNS-over-HTTPS ({ips[0]}); Kaggle's resolver is still behind", post_public(host, ips[0])
            else:
                via, code, body = "not resolvable yet", 0, ""
        if code == 200:
            break
        time.sleep(5)
    print(f"public /predict via {via}: HTTP {code}")
    try:
        print(json.dumps(json.loads(body), indent=2))
    except ValueError:
        print(body[:1000])
    if code != 200:
        print("Public check failed. The local check above passed, so try the URL from your phone anyway; "
              "if that fails too, re-run the tunnel cell to get a new URL. Tail of tunnel.log:")
        print(open(TUNNEL_LOG).read()[-1500:])
    APP_LINK = f"{PUBLIC_URL}/#key={API_KEY}"
    from IPython.display import HTML, display as _display
    _display(HTML(f'<p style="font-size:1.2em">Web app: <a href="{APP_LINK}" target="_blank">{APP_LINK}</a></p>'))
    print("Web app (open on your phone or laptop):", APP_LINK)
    print(f"Mobile app / API -> base URL: {PUBLIC_URL}   header X-API-Key: {API_KEY}")

# %%
# Keep the API up for KEEPALIVE_MIN minutes (interrupt the kernel to stop earlier).
if not IS_BATCH:
    print(f"serving {APP_LINK if 'APP_LINK' in globals() else PUBLIC_URL} for {KEEPALIVE_MIN} min")
    try:
        end = time.time() + KEEPALIVE_MIN * 60
        while time.time() < end:
            time.sleep(30)
            if server.poll() is not None:
                print("server stopped; tail of server.log:\n", open(SERVER_LOG).read()[-2000:]); break
            if "tunnel" in globals() and tunnel.poll() is not None:
                print("tunnel stopped; tail of tunnel.log:\n", open(TUNNEL_LOG).read()[-2000:]); break
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (globals().get("server"), globals().get("tunnel")):
            if proc is not None:
                proc.terminate()
        print("stopped server and tunnel")
else:
    shutil.rmtree(DATA_DIR, ignore_errors=True)  # keep the saved output small: only ckpt/ (and the repo) remain
    print("commit run done; output:", os.listdir(WORK))
