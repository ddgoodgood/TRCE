import random
import os
import csv
from pathlib import Path
import torch
import json
import numpy as np
from PIL import Image

 
def seed_everything(seed, workers: bool = False) -> int:

    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
  
def get_image(img):
  img = (img.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
  return Image.fromarray(img)

def read_csv_as_dict(fp):
  dict_list = []
  with open(fp, mode='r', encoding='utf-8') as file:
      csv_reader = csv.DictReader(file)
      for row in csv_reader:
          dict_list.append(row)
  return dict_list
          
def read_negative_concepts(fp):
  with  open(fp, 'r') as file:
    data = json.load(file)
    return data
  
