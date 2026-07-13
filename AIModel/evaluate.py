"""
Loads the best saved model and produces a classification report +
confusion matrix on the subject-wise held-out test set.
Run this after train.py, or import evaluate_model() directly.
"""
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import CNNLSTMHybrid


@torch.no_grad()
def get_predictions(model, X_seq, X_feat, y_true, device, batch_size=cfg.BATCH_SIZE):
    model.eval()
    seq_tensor = torch.from_numpy(X_seq).float()
    feat_tensor = torch.from_numpy(X_feat).float()
    preds = []
    for i in range(0, len(seq_tensor), batch_size):
        seq_batch = seq_tensor[i:i + batch_size].to(device)
        feat_batch = feat_tensor[i:i + batch_size].to(device)
        logits = model(seq_batch, feat_batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), y_true


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stress Detection - Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(X_seq, X_feat, y_test, model_path=cfg.BEST_MODEL_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTMHybrid(n_channels=cfg.N_CHANNELS,
                           n_scalar_features=cfg.N_SCALAR_FEATURES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_pred, y_true = get_predictions(model, X_seq, X_feat, y_test, device)

    class_names = ["Relaxed", "Stressed"]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(cm)

    out_path = f"{cfg.OUTPUT_DIR}/confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, out_path)
    print(f"Confusion matrix plot saved to {out_path}")

    with open(f"{cfg.OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    return report, cm