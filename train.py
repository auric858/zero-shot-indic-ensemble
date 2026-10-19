"""Fine-tuning one model on english only, then evaluate zero-shot on Santali and Bengali.
Saves softmax probabilities to outputs/predictions/{model}_seed{seed}.npz"""
import argparse
import gc
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import MODELS, DEFAULTS, PRED_DIR, MODEL_DIR, SEEDS, TEST_LANGS
from data_utils import load_english, load_test
from models import build, get_device
from evaluate import predict_probs, metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(name, seed, epochs=None, tag="", save_model=False):
    out_path = PRED_DIR / f"{tag}{name}_seed{seed}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    cfg = {**DEFAULTS, **MODELS[name].get("overrides", {})}
    if epochs:
        cfg["epochs"] = epochs
    set_seed(seed)
    device = get_device()
    t0 = time.time()

    # data: English only for training and model selection
    (train_x, train_y, _), (val_x, val_y, _) = load_english()
    tests = {lang: load_test(lang) for lang in TEST_LANGS}

    # building the model
    tok, model = build(name)
    if cfg.get("freeze_embeddings"):
        model.base_model.embeddings.word_embeddings.requires_grad_(False)
    model.to(device)

    # optimizer: AdamW, no weight decay on biases/LayerNorm
    no_decay = ("bias", "LayerNorm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [
        {"params": [p for n, p in trainable if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in trainable if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg["lr"])

    loader = DataLoader(list(range(len(train_x))), batch_size=cfg["batch_size"], shuffle=True)
    steps_per_epoch = math.ceil(len(loader) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = device.type == "cuda"

    # training loop
    best_f1, best_state = -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        bar = tqdm(loader, desc=f"{name} seed{seed} epoch {epoch}/{cfg['epochs']}", leave=False)
        for k, batch in enumerate(bar):
            ids = batch.tolist()
            enc = tok([train_x[i] for i in ids], padding=True, truncation=True,
                      max_length=cfg["max_len"], return_tensors="pt").to(device)
            labels = torch.tensor([train_y[i] for i in ids], device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**enc, labels=labels).loss / cfg["grad_accum"]
            loss.backward()

            if (k + 1) % cfg["grad_accum"] == 0 or (k + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            bar.set_postfix(loss=f"{loss.item() * cfg['grad_accum']:.3f}")

        # model selection uses english validation only
        val_probs = predict_probs(model, tok, val_x, device, cfg["max_len"])
        val_f1 = metrics(val_y, val_probs.argmax(1))["macro_f1"]
        print(f"  epoch {epoch}: English val macro-F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    #restores the best English checkpoint,then runs zero-shot tests once
    model.load_state_dict(best_state)
    val_probs = predict_probs(model, tok, val_x, device, cfg["max_len"])
    saved = {"val_probs": val_probs, "val_labels": np.array(val_y), "val_f1": best_f1}

    summary = f"[{name} seed {seed}] English val F1 = {best_f1:.4f}"
    for lang, (x, y, _) in tests.items():
        probs = predict_probs(model, tok, x, device, cfg["max_len"])
        m = metrics(y, probs.argmax(1))
        saved[f"{lang}_probs"] = probs
        saved[f"{lang}_labels"] = np.array(y)
        summary += f" | {lang}: acc {m['accuracy']:.3f}, F1 {m['macro_f1']:.3f}"
    print(summary + f" | {time.time() - t0:.0f}s")

    np.savez(out_path, **saved)
    if save_model and not tag:
        model.save_pretrained(MODEL_DIR / name)
        tok.save_pretrained(MODEL_DIR / name)

    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, saved under a 'smoke_' prefix (does not block real runs)")
    ap.add_argument("--save-model", nargs="*", default=["xlmr_base"],
                    help="models whose first-seed checkpoint is kept (used by Mundari step)")
    args = ap.parse_args()

    for m in args.models:
        for i, s in enumerate(args.seeds):
            train_one(m, s,
                      epochs=1 if args.smoke else args.epochs,
                      tag="smoke_" if args.smoke else "",
                      save_model=(m in args.save_model and i == 0))