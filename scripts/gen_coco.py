import os
import random
import numpy as np
import torch
import csv
import sys
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

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

def main(unet_path,save_dir):
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    device = torch.device("cuda")
    
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,torch_dtype= torch.float16).to(device)
    pipe.unet = UNet2DConditionModel.from_pretrained(unet_path,torch_dtype= torch.float16).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)

    csv_file_path = 'data/prompts_csv/coco_30k_val.csv'
    dict_list = []

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            dict_list.append(row)

    coco_save_dir = os.path.join(save_dir, "coco-val")
    print(coco_save_dir)
    if not os.path.exists(coco_save_dir):
        os.makedirs(coco_save_dir)

    for item in tqdm(dict_list):
        prompt = item['prompt']
        seed = int(item['evaluation_seed'])
        case_number = item['case_number']
        coco_id = item['coco_id']
        guidance_scale = 7.5

        seed_everything(seed)

        num_inference_steps = 30
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                      guidance_scale=guidance_scale, 
                      num_images_per_prompt=1).images
        image = images[0]
        output_dir = f"{coco_save_dir}/{case_number}_{coco_id}.jpg"
        image.save(output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_coco.py <unet_path> <save_dir>")
        sys.exit(1)

    unet_path = sys.argv[1]
    save_dir = sys.argv[2]
    main(unet_path,save_dir)

