from huggingface_hub import HfApi

REPO_ID = "malansi/gsr-cossinus-volunteers-raw-data"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

print(f"Uploading raw volunteer data to {REPO_ID} (private)...\n")
api.upload_folder(
    folder_path="..",
    repo_id=REPO_ID,
    repo_type="dataset",
    ignore_patterns=[
        "**/.DS_Store",
        ".git/**",
        "AIModel/**",
        "cleaned_data/**",
    ],
    commit_message="Upload raw volunteer sensor data (backup)",
)

print(f"\n✅ Done → https://huggingface.co/datasets/{REPO_ID}")
