from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./dataset",
    repo_id="l0ulan/chinese_modern_poems",
    repo_type="dataset",
)
