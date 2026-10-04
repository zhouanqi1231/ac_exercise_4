"""
Train the audio-visual emotion-recognition baseline on mteb/RAVDESS_AV.

Frozen SigLIP2 (video) + frozen Wav2Vec2 (audio) + trainable fusion MLP.
Only the MLP is optimised, so training is fast and the checkpoint is tiny.

Example
-------
    python train.py --epochs 20 --batch-size 8 --num-frames 8 \
                    --output-dir ./ckpt --subject-independent
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from av_emotion import (
    AVCollator,
    AVEmotionModel,
    RavdessAVDataset,
    DEFAULT_SIGLIP,
    DEFAULT_WAV2VEC2,
    label_names,
    load_processors,
    load_ravdess,
    make_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AV emotion recognition baseline (RAVDESS_AV)")
    p.add_argument("--siglip-id", default=DEFAULT_SIGLIP)
    p.add_argument("--wav2vec2-id", default=DEFAULT_WAV2VEC2)
    p.add_argument("--num-frames", type=int, default=8, help="frames sampled per clip")
    p.add_argument("--max-audio-seconds", type=float, default=6.0)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--subject-independent", action="store_true",
                   help="actor-disjoint train/val split (standard RAVDESS protocol)")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--limit", type=int, default=0, help="debug: cap #samples (0=all)")
    p.add_argument("--cache-dir", default='./hf_cache')
    p.add_argument("--output-dir", default="./ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch["labels"].to(device)
        logits = model(
            pixel_values=batch["pixel_values"].to(device),
            n_frames=batch["n_frames"],
            input_values=batch["input_values"].to(device),
            input_lengths=batch["input_lengths"].to(device),
        )
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_labels).numpy()
    return loss_sum / max(total, 1), correct / max(total, 1), preds, targets


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- data ----
    print("Loading dataset mteb/RAVDESS_AV ...")
    print(args.cache_dir)
    ds = load_ravdess(split="test", cache_dir=args.cache_dir)
    classes = label_names(ds)
    print(f"{len(ds)} clips, {len(classes)} classes: {classes}")

    train_idx, val_idx, tst_idx, subj = make_split(
        ds, val_ratio=args.val_ratio, seed=args.seed,
        subject_independent=args.subject_independent,
    )
    if args.limit:
        train_idx = train_idx[: args.limit]
        val_idx = val_idx[: max(1, args.limit // 4)]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} "
          f"({'subject-independent' if subj else 'stratified'})")

    train_ds = RavdessAVDataset(ds.select(train_idx), args.num_frames, args.max_audio_seconds)
    val_ds = RavdessAVDataset(ds.select(val_idx), args.num_frames, args.max_audio_seconds)
    tst_ds = RavdessAVDataset(ds.select(tst_idx), args.num_frames, args.max_audio_seconds)

    image_processor, feature_extractor = load_processors(args.siglip_id, args.wav2vec2_id)
    collate = AVCollator(image_processor, feature_extractor, args.num_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(tst_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    # ---- model ----
    model = AVEmotionModel(
        num_classes=len(classes),
        siglip_id=args.siglip_id,
        wav2vec2_id=args.wav2vec2_id,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze=True,
    ).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable params (MLP only): {n_train:,}")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for step, batch in enumerate(train_loader, 1):
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(
                    pixel_values=batch["pixel_values"].to(device),
                    n_frames=batch["n_frames"],
                    input_values=batch["input_values"].to(device),
                    input_lengths=batch["input_lengths"].to(device),
                )
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            seen += labels.size(0)
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running/seen:.4f} acc={correct/seen:.4f}")

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device, criterion)
        print(f"[epoch {epoch}] train_loss={running/max(seen,1):.4f} "
              f"train_acc={correct/max(seen,1):.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "classifier_state": model.classifier.state_dict(),
                "label_names": classes,
                "config": {
                    "siglip_id": args.siglip_id,
                    "wav2vec2_id": args.wav2vec2_id,
                    "num_frames": args.num_frames,
                    "max_audio_seconds": args.max_audio_seconds,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "video_dim": model.video_dim,
                    "audio_dim": model.audio_dim,
                },
            }, best_path)
            print(f"  -> saved best checkpoint (val_acc={best_acc:.4f}) to {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.classifier.load_state_dict(ckpt["classifier_state"])
    _, val_acc, preds, targets = evaluate(model, val_loader, device, criterion)
    print(f"\nBest val accuracy: {val_acc:.4f}")
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nClassification report:")
        print(classification_report(targets, preds, target_names=classes, digits=4, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(targets, preds))
    except Exception:
        pass

    with open(os.path.join(args.output_dir, "label_names.json"), "w") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
