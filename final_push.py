import os
from huggingface_hub import HfApi, login
from openenv.core.registration import push_to_hub

# 1. Login
HF_TOKEN = "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" # Paste your token
login(token=HF_TOKEN)

# 2. Push the environment
# This will look for your openenv.yaml and warehouse_env folder
try:
    print("Starting push to Hugging Face...")
    push_to_hub(
        repo_id="Tanmay7506/warehouse-bot-env", # Change to your HF username
        folder_path=".",
        commit_message="Initial submission for Meta Scaler Hackathon"
    )
    print("Push Successful! Submit this URL to the Scaler Dashboard.")
except Exception as e:
    print(f"Error during push: {e}")
