from huggingface_hub import upload_file
from datetime import datetime

HF_USERNAME = "malansi"
REPO_NAME   = "gsr-hrv-stress-detection-cnn-lstm"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

files = [
    ("outputs/README.md",                "README.md"),
    ("outputs/best_model_stress.pt",     "best_model_stress.pt"),
    ("outputs/global_mean.npy",          "global_mean.npy"),
    ("outputs/global_std.npy",           "global_std.npy"),
    ("outputs/global_feat_mean.npy",     "global_feat_mean.npy"),
    ("outputs/global_feat_std.npy",      "global_feat_std.npy"),
    ("outputs/confusion_matrix.png",     "confusion_matrix.png"),
    ("outputs/training_curves.png",      "training_curves.png"),
    ("outputs/kfold_accuracy.png",       "kfold_accuracy.png"),
    ("outputs/classification_report.txt","classification_report.txt"),
    ("outputs/kfold_summary.txt",        "kfold_summary.txt"),
]

print(f"Uploading to {REPO_ID}...\n")
for local, remote in files:
    try:
        upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Update {local} — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"  ✅  {local}")
    except Exception as e:
        print(f"  ❌  {local}: {e}")

print(f"\n✅ Done → https://huggingface.co/{REPO_ID}")
