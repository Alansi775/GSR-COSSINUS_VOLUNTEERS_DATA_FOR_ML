from huggingface_hub import HfApi

REPO_ID = "malansi/gsr-hrv-stress-detection-dataset"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)

print(f"Uploading cleaned dataset to {REPO_ID} (public)...\n")
api.upload_folder(
    folder_path="../cleaned_data",
    repo_id=REPO_ID,
    repo_type="dataset",
    ignore_patterns=["**/.DS_Store"],
    commit_message="Upload cleaned, anonymized volunteer GSR+HRV dataset",
)

print(f"\n✅ Done → https://huggingface.co/datasets/{REPO_ID}")
