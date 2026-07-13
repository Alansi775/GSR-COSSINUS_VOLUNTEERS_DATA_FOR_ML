"""
Trains the hybrid CNN+LSTM+MLP stress classifier and saves:
  - outputs/best_model_stress.pt
  - outputs/global_mean.npy, outputs/global_std.npy          (sequence channels)
  - outputs/global_feat_mean.npy, outputs/global_feat_std.npy (scalar features)
  - outputs/training_curves.png  (loss + accuracy, train & val, per epoch)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from data_loader import load_all_volunteers
from feature_extraction import build_dataset
from dataset import WindowDataset, subject_wise_split, compute_global_stats, apply_global_stats
from model import CNNLSTMHybrid


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_weighted_loss(y_train, device):
    classes, counts = np.unique(y_train, return_counts=True)
    counts = counts.astype(np.float32)
    weights = counts.sum() / (len(classes) * counts)
    weight_tensor = torch.zeros(len(classes), dtype=torch.float32)
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = float(w)
    print(f"Class counts: {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"Class weights: {weight_tensor.tolist()}")
    return nn.CrossEntropyLoss(weight=weight_tensor.to(device))


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb_seq, xb_feat, yb in loader:
            xb_seq, xb_feat, yb = xb_seq.to(device), xb_feat.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb_seq, xb_feat)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb_seq.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb_seq.size(0)

    return total_loss / total, correct / total


def plot_training_curves(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    val_acc = history["val_acc"]
    best_idx = int(np.argmax(val_acc))
    best_epoch = epochs[best_idx]
    best_acc = val_acc[best_idx]

    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, val_acc, label="Validation accuracy")
    axes[1].scatter([best_epoch], [best_acc], color="red", zorder=5, s=60)
    axes[1].annotate(
        f"Best: {best_acc*100:.2f}%\n(epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch, best_acc - 0.12 if best_acc > 0.5 else best_acc + 0.12),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9),
        zorder=6,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy over training  (best val acc = {best_acc*100:.2f}%)")
    axes[1].legend(loc="upper left", framealpha=0.9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")
    print(f"Best validation accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("\n=== Loading data ===")
    volunteer_data, skipped = load_all_volunteers()

    print("\n=== Building windows ===")
    X_seq, X_feat, y, groups = build_dataset(volunteer_data)
    print(f"Total windows: seq {X_seq.shape}, feat {X_feat.shape}, labels {np.bincount(y)}")
    print(f"Unique volunteers with usable windows: {len(set(groups))}")

    print("\n=== Subject-wise train/val/test split ===")
    splits = subject_wise_split(X_seq, X_feat, y, groups)
    Xs_train, Xf_train, y_train, g_train = splits["train"]
    Xs_val, Xf_val, y_val, g_val = splits["val"]
    Xs_test, Xf_test, y_test, g_test = splits["test"]
    print(f"Train: {Xs_train.shape[0]} windows / {len(set(g_train))} volunteers")
    print(f"Val:   {Xs_val.shape[0]} windows / {len(set(g_val))} volunteers")
    print(f"Test:  {Xs_test.shape[0]} windows / {len(set(g_test))} volunteers")

    print("\n=== Global normalization (fit on train only) ===")
    seq_mean, seq_std = compute_global_stats(Xs_train)
    feat_mean, feat_std = compute_global_stats(Xf_train)
    np.save(cfg.GLOBAL_MEAN_PATH, seq_mean)
    np.save(cfg.GLOBAL_STD_PATH, seq_std)
    np.save(cfg.GLOBAL_FEAT_MEAN_PATH, feat_mean)
    np.save(cfg.GLOBAL_FEAT_STD_PATH, feat_std)
    print(f"Saved sequence + scalar-feature normalization stats to {cfg.OUTPUT_DIR}/")

    Xs_train = apply_global_stats(Xs_train, seq_mean, seq_std)
    Xs_val = apply_global_stats(Xs_val, seq_mean, seq_std)
    Xs_test = apply_global_stats(Xs_test, seq_mean, seq_std)
    Xf_train = apply_global_stats(Xf_train, feat_mean, feat_std)
    Xf_val = apply_global_stats(Xf_val, feat_mean, feat_std)
    Xf_test = apply_global_stats(Xf_test, feat_mean, feat_std)

    train_loader = DataLoader(WindowDataset(Xs_train, Xf_train, y_train),
                               batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xs_val, Xf_val, y_val),
                             batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(WindowDataset(Xs_test, Xf_test, y_test),
                              batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = CNNLSTMHybrid(n_channels=cfg.N_CHANNELS,
                           n_scalar_features=cfg.N_SCALAR_FEATURES).to(device)
    criterion = make_weighted_loss(y_train, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n=== Training ===")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  -> new best model saved to {cfg.BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg.EARLY_STOP_PATIENCE} epochs)")
                break

    plot_training_curves(history, cfg.TRAINING_CURVES_PATH)

    print("\n=== Final test evaluation (best checkpoint) ===")
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    return model, (Xs_test, Xf_test, y_test, g_test), device


if __name__ == "__main__":
    main()