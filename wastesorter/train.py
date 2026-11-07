"""Fine-tune a timm backbone on TrashNet + RealWaste (unified 8 material classes).

python -m wastesorter.train --out /kaggle/working/ckpt --data-dir /kaggle/working/data
Outputs: best.pt (weights + metadata + temperature), metrics.json, confusion_matrix.png, splits.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .data import WasteDataset, get_samples, split_samples, subsample
from .taxonomy import BINS, MATERIAL_TO_BIN, MATERIALS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--trashnet-dir", default=None, help="skip HF download, use this folder")
    p.add_argument("--realwaste-dir", default=None, help="skip git clone, use this folder")
    p.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    p.add_argument("--img-size", type=int, default=288)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--head-epochs", type=int, default=2, help="initial epochs with frozen backbone")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-per-group", type=int, default=0, help="debug: cap images per (source,class)")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_transforms(size, mean, std):
    bicubic = T.InterpolationMode.BICUBIC
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.85, 1.18), interpolation=bicubic),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.15),
    ])
    evalt = T.Compose([T.Resize((size, size), interpolation=bicubic), T.ToTensor(), T.Normalize(mean, std)])
    return train, evalt


@torch.no_grad()
def collect_logits(model, loader, device, amp):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lo = model(x)
        outs.append(lo.float().cpu()); ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def compute_metrics(logits, labels, sources):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    pred, y = logits.argmax(1).numpy(), labels.numpy()
    to_bin = np.array([BINS.index(MATERIAL_TO_BIN[m]) for m in MATERIALS])
    allc = list(range(len(MATERIALS)))
    res = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=allc, zero_division=0)),
        "bin_accuracy": float((to_bin[pred] == to_bin[y]).mean()),
        "per_source": {},
    }
    sources = np.array(sources)
    for src in sorted(set(sources.tolist())):
        m = sources == src
        res["per_source"][src] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                                  "bin_accuracy": float((to_bin[pred[m]] == to_bin[y[m]]).mean())}
    res["report"] = classification_report(y, pred, labels=allc, target_names=MATERIALS, output_dict=True, zero_division=0)
    res["confusion_matrix"] = confusion_matrix(y, pred, labels=allc).tolist()
    return res


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.2, 5.0))


def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(MATERIALS))); ax.set_xticklabels(MATERIALS, rotation=45, ha="right")
    ax.set_yticks(range(len(MATERIALS))); ax.set_yticklabels(MATERIALS)
    for i in range(len(MATERIALS)):
        for j in range(len(MATERIALS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    a = parse_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} amp={amp}")

    samples = get_samples(a.data_dir, a.trashnet_dir, a.realwaste_dir)
    if a.max_per_group:
        samples = subsample(samples, a.max_per_group, a.seed)
    tr_s, va_s, te_s = split_samples(samples, seed=a.seed)
    print(f"images: train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
    counts = np.bincount([s.label for s in tr_s], minlength=len(MATERIALS))
    print("train class counts:", dict(zip(MATERIALS, counts.tolist())))
    (out / "splits.json").write_text(json.dumps({
        k: [{"path": s.path, "label": s.label, "source": s.source} for s in v]
        for k, v in {"train": tr_s, "val": va_s, "test": te_s}.items()}))

    model = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=len(MATERIALS), drop_path_rate=0.1)
    model = model.to(device).to(memory_format=torch.channels_last)
    dc = timm.data.resolve_data_config({}, model=model)
    mean, std = list(dc["mean"]), list(dc["std"])
    t_train, t_eval = make_transforms(a.img_size, mean, std)

    mk = lambda ss, tf, shuf: DataLoader(WasteDataset(ss, tf), batch_size=a.batch_size, shuffle=shuf,
                                         num_workers=a.workers, pin_memory=amp, drop_last=shuf,
                                         persistent_workers=a.workers > 0)
    train_loader, val_loader, test_loader = mk(tr_s, t_train, True), mk(va_s, t_eval, False), mk(te_s, t_eval, False)

    w = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)

    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr}, {"params": head, "lr": a.lr * a.head_lr_mult}],
                            weight_decay=a.weight_decay)
    steps_per_epoch = len(train_loader)
    total, warm = a.epochs * steps_per_epoch, max(1, steps_per_epoch // 2)
    lam = lambda s: (s + 1) / warm if s < warm else max(0.01, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def save(path, state, temperature, metrics=None):
        torch.save({"model_name": a.model, "classes": MATERIALS, "img_size": a.img_size, "mean": mean, "std": std,
                    "temperature": temperature, "state_dict": state, "metrics": metrics or {}}, path)

    best_f1, best_state, t0 = -1.0, None, time.time()
    for epoch in range(a.epochs):
        head_only = epoch < a.head_epochs
        for p in backbone:
            p.requires_grad = not head_only
        model.train()
        run = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if i % 50 == 0:
                print(f"  ep{epoch+1} step {i}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)
        vl, vy = collect_logits(model, val_loader, device, amp)
        vm = compute_metrics(vl, vy, [s.source for s in va_s])
        print(f"epoch {epoch+1}/{a.epochs} {'(head only)' if head_only else ''} train_loss={run/steps_per_epoch:.3f} "
              f"val_acc={vm['accuracy']:.4f} val_macroF1={vm['macro_f1']:.4f} val_bin_acc={vm['bin_accuracy']:.4f} "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            save(out / "best.pt", best_state, 1.0)

    model.load_state_dict(best_state)
    vl, vy = collect_logits(model, val_loader, device, amp)
    temp = fit_temperature(vl, vy)
    tl, ty = collect_logits(model, test_loader, device, amp)
    tm = compute_metrics(tl, ty, [s.source for s in te_s])
    tm["temperature"] = temp
    (out / "metrics.json").write_text(json.dumps(tm, indent=2))
    plot_confusion(tm["confusion_matrix"], out / "confusion_matrix.png")
    summary = {"accuracy": tm["accuracy"], "macro_f1": tm["macro_f1"], "bin_accuracy": tm["bin_accuracy"],
               "per_source": tm["per_source"]}
    save(out / "best.pt", best_state, temp, summary)
    print("TEST:", json.dumps(summary, indent=2))
    print(f"saved to {out/'best.pt'} (T={temp:.3f})")


if __name__ == "__main__":
    main()
