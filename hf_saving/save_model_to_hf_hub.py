import torch
torch.cuda.is_available = lambda: False
import torch.nn as nn
from SAE import SAE

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
device = torch.device('cpu')
api = HfApi(token=os.environ['HF_TOKEN'])
print(api)

model = SAE(1280, 128)
state_dict_path = '../st_dicts/sae_for_kandinsky.pth'
repo_id = "rexp10r/sae-for-kandinsky"
model_path = "SAE.py"

state_dict = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state_dict)
print("Model loaded")

api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=os.environ['HF_TOKEN']
)
print("Repo created")

api.upload_file(
	path_or_fileobj=model_path,
	path_in_repo='model.py',
	repo_id=repo_id,
	repo_type="model"
)
print("Model code uploaded")

api.upload_file(
	path_or_fileobj=state_dict_path,
	path_in_repo='st_dict.pth',
	repo_id=repo_id,
	repo_type="model"
)
print("Model state dict uploaded")
