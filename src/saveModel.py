from safetensors.torch import save_file, load_file
import json
from huggingface_hub import Repository, HfApi, HfFolder
import os

"""
This is the code for saving and loading the model with safetensors format,
dividing the model weights into N amount files and generating an index file.
"""

def save_model_weights(model, base_path, num_files=1):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    chunk_size = len(keys) // num_files

    index = {}

    for i in range(num_files):
        chunk_keys = keys[i * chunk_size:(i + 1) * chunk_size]
        chunk_state_dict = {key: state_dict[key] for key in chunk_keys}
        save_file(chunk_state_dict, f"{base_path}_part_{i}.safetensors")
        index[f"../output/{base_path}_part_{i}.safetensors"] = list(chunk_state_dict.keys())

    with open(f"../output/{base_path}.index.json", 'w') as f:
        json.dump(index, f, indent=4)

def load_model_weights(model, base_path, num_files=1):
    state_dict = {}

    with open(f"../output/{base_path}.index.json", 'r') as f:
        index = json.load(f)

    for i in range(num_files):
        file_path = f"../output/{base_path}_part_{i}.safetensors"
        if file_path in index:
            chunk_state_dict = load_file(file_path)
            state_dict.update(chunk_state_dict)

    model.load_state_dict(state_dict)

def push_to_hf(model_dir, model_name, user):
    """
    Push the trained model to HuggingFace Hub.
    
    Args:
        model_dir: Directory where the model is saved.
        model_name: Desired name of the model on HuggingFace Hub.
        user: HuggingFace username.
        token: HuggingFace access token.
    """

    token = os.getenv('HF_TOKEN')  # Ensure your HF token is set as an environment variable

    repo_id = f"{user}/{model_name}"
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repository creation failed: {e}")

    repo = Repository(local_dir=model_dir, clone_from=repo_id, use_auth_token=token)
    repo.git_add()
    repo.git_commit("Upload fine-tuned custom Transformer model")
    repo.git_push()
