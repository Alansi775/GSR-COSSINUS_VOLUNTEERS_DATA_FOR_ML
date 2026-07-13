"""
Subject-wise k-fold cross-validation for the stress classifier.

Why this exists: a single train/val/test split only tells you how the
model did on one particular 21-volunteer test set. Different splits can
swing the reported accuracy by several points just from which volunteers
happened to land in test. This script trains N_FOLDS independent models,
each with a different held-out set of volunteers, and reports
mean ± std accuracy — the number that's actually defensible in a paper.

Usage:
    python kfold_eval.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from data_loader import load_all_volunteers
from feature_extraction import build_dataset
from dataset import WindowDataset, compute_global_stats, apply_global_stats
from model import CNNLSTMHybrid
from train import make_weighted_loss, run_epoch, get_device

N_FOLDS = 5
VAL_FRACTION_WITHIN_FOLD = 0.2
MAX_EPOCHS = 40
PATIENCE = 8


def carve_out_validation(train_idx, groups, val_fraction, seed):
    """Splits off a validation subset (by volunteer) from a fold's training indices."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    sub_train_rel, sub_val_rel = next(gss.split(train_idx, groups=groups[train_idx]))
    return train_idx[sub_train_rel], train_idx[sub_val_rel]


def train_one_fold(Xs_tr, Xf_tr, y_tr, Xs_val, Xf_val, y_val, device, seed):
    torch.manual_seed(seed)

    seq_mean, seq_std = compute_global_stats(Xs_tr)
    feat_mean, feat_std = compute_global_stats(Xf_tr)

    Xs_tr_n = apply_global_stats(Xs_tr, seq_mean, seq_std)
    Xs_val_n = apply_global_stats(Xs_val, seq_mean, seq_std)
    Xf_tr_n = apply_global_stats(Xf_tr, feat_mean, feat_std)
    Xf_val_n = apply_global_stats(Xf_val, feat_mean, feat_std)

    train_loader = DataLoader(WindowDataset(Xs_tr_n, Xf_tr_n, y_tr),
                               batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xs_val_n, Xf_val_n, y_val),
                             batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = CNNLSTMHybrid(n_channels=cfg.N_CHANNELS,
                           n_scalar_features=cfg.N_SCALAR_FEATURES).to(device)
    criterion = make_weighted_loss(y_tr, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, seq_mean, seq_std, feat_mean, feat_std


@torch.no_grad()
def evaluate_fold(model, Xs_test_n, Xf_test_n, device):
    model.eval()
    seq_t = torch.from_numpy(Xs_test_n).float().to(device)
    feat_t = torch.from_numpy(Xf_test_n).float().to(device)
    logits = model(seq_t, feat_t)
    return logits.argmax(dim=1).cpu().numpy()


def main():
    import os
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("\n=== Loading data ===")
    volunteer_data, skipped = load_all_volunteers()

    print("\n=== Building windows ===")
    X_seq, X_feat, y, groups = build_dataset(volunteer_data)
    print(f"Total windows: {X_seq.shape[0]}, volunteers: {len(set(groups))}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_accuracies = []
    all_preds, all_true = [], []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
            gkf.split(X_seq, y, groups), start=1):

        seed = cfg.RANDOM_SEED + fold_idx
        train_idx, val_idx = carve_out_validation(
            trainval_idx, groups, VAL_FRACTION_WITHIN_FOLD, seed)

        # sanity: no volunteer leakage across train/val/test in this fold
        assert not (set(groups[train_idx]) & set(groups[test_idx])), "leakage: train/test"
        assert not (set(groups[val_idx]) & set(groups[test_idx])), "leakage: val/test"

        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
        print(f"Train: {len(train_idx)} windows / {len(set(groups[train_idx]))} volunteers")
        print(f"Val:   {len(val_idx)} windows / {len(set(groups[val_idx]))} volunteers")
        print(f"Test:  {len(test_idx)} windows / {len(set(groups[test_idx]))} volunteers")

        model, seq_mean, seq_std, feat_mean, feat_std = train_one_fold(
            X_seq[train_idx], X_feat[train_idx], y[train_idx],
            X_seq[val_idx], X_feat[val_idx], y[val_idx],
            device, seed)

        Xs_test_n = apply_global_stats(X_seq[test_idx], seq_mean, seq_std)
        Xf_test_n = apply_global_stats(X_feat[test_idx], feat_mean, feat_std)
        preds = evaluate_fold(model, Xs_test_n, Xf_test_n, device)

        acc = float((preds == y[test_idx]).mean())
        fold_accuracies.append(acc)
        all_preds.extend(preds.tolist())
        all_true.extend(y[test_idx].tolist())
        print(f"Fold {fold_idx} test accuracy: {acc*100:.2f}%")

    fold_accuracies = np.array(fold_accuracies)
    mean_acc = fold_accuracies.mean()
    std_acc = fold_accuracies.std()

    print("\n=== K-Fold Summary ===")
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i}: {acc*100:.2f}%")
    print(f"\nMean accuracy: {mean_acc*100:.2f}%  +/-  {std_acc*100:.2f}%")

    cm = confusion_matrix(all_true, all_preds)
    print("\nPooled confusion matrix (all folds combined):")
    print(cm)

    report = classification_report(all_true, all_preds,
                                    target_names=["Relaxed", "Stressed"], digits=4)
    print("\nPooled classification report (all folds combined):")
    print(report)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fold_labels = [f"Fold {i}" for i in range(1, N_FOLDS + 1)]
    bars = ax.bar(fold_labels, fold_accuracies * 100, color="#4C72B0")
    ax.axhline(mean_acc * 100, color="red", linestyle="--",
               label=f"Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    for bar, acc in zip(bars, fold_accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc * 100 + 1,
                 f"{acc*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{N_FOLDS}-Fold Subject-Wise Cross-Validation")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = f"{cfg.OUTPUT_DIR}/kfold_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPer-fold accuracy chart saved to {out_path}")

    with open(f"{cfg.OUTPUT_DIR}/kfold_summary.txt", "w") as f:
        f.write(f"{N_FOLDS}-Fold Subject-Wise Cross-Validation\n")
        f.write("=" * 45 + "\n")
        for i, acc in enumerate(fold_accuracies, start=1):
            f.write(f"Fold {i}: {acc*100:.2f}%\n")
        f.write(f"\nMean accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n")
        f.write("\nPooled confusion matrix (all folds combined):\n")
        f.write(str(cm) + "\n")
        f.write("\nPooled classification report:\n")
        f.write(report)
    print(f"Summary written to {cfg.OUTPUT_DIR}/kfold_summary.txt")


if __name__ == "__main__":
    main()