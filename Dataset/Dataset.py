import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

def download_and_save(repo_id, local_path):
    """Downloads a dataset and saves it to disk if it doesn't exist."""
    print(f"Loading {repo_id}...")
    ds = load_dataset(repo_id)
    ds.save_to_disk(local_path)
    print(f"Successfully saved to {local_path}")

def main():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    datasets_to_fetch = {
        "Piro17/affectnethq": "./datasets/AffectnetHQ/",
        "deanngkl/raf-db-7emotions": "./datasets/RAF-DB-7emotions/"
    }

    for repo, path in datasets_to_fetch.items():
        if not os.path.exists(path):
            download_and_save(repo, path)
        else:
            print(f"Dataset already exists at {path}, skipping.")

if __name__ == "__main__":
    main()