import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tools import *
from tqdm import tqdm
import sys



def main(unet_path,save_dir):
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda")
    
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)
    pipe.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)
    num_inference_steps = 30
    
    # MMA-diffusion
    mma_file = "data/adv_prompts_csv/nudity-mma-diffusion.csv"
    mma_dataset = [{"prompt":item['adv_prompt'],'seed':42,"guidance":7.5} for item in read_csv_as_dict(mma_file)]
    # Ring-A-Bell
    ring_file = "data/adv_prompts_csv/nudity-ring-a-bell.csv"
    ring_dataset = [{"prompt":item['sensitive prompt'],'seed':42,"guidance":7.5} for item in read_csv_as_dict(ring_file)]
    # UnlearnDiff
    unlearn_file = "data/adv_prompts_csv/nudity-unlearndiff.csv"
    unlearn_dataset = [{"prompt":item['prompt'],'seed':item['evaluation_seed'],'guidance':item['evaluation_guidance']} for item in read_csv_as_dict(unlearn_file)]
    # P4D
    p4d_file = "data/adv_prompts_csv/nudity-p4d.csv"
    p4d_dataset = [{"prompt":item['prompt'],'seed':item['evaluation_seed'],'guidance':item['evaluation_guidance']} for item in read_csv_as_dict(p4d_file)]
    # I2P
    i2p_file = "data/adv_prompts_csv/i2p.csv"
    i2p_dict = read_csv_as_dict(i2p_file)
    sexual_list = []
    for item in i2p_dict:
        if 'sexual' in [x.strip() for x in item['categories'].split(',')]:
            sexual_list.append(item)
    i2p_dataset = [{"prompt":item['prompt'],'seed':item['evaluation_seed'],'guidance':item['evaluation_guidance']} for item in sexual_list]

    i2p_save_dir = os.path.join(save_dir,"i2p")
    print(i2p_save_dir)
    if not os.path.exists(i2p_save_dir):
        os.makedirs(i2p_save_dir)
    for i,item in tqdm(enumerate(i2p_dataset)):
        prompt = item['prompt']
        seed = int(item['seed'])
        guidance_scale = float(item['guidance'])
        num_inference_steps = 30
        seed_everything(seed)
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    num_images_per_prompt=1,
                    height = 512,
                    width = 512).images
        image = images[0]
        output_dir = f"{i2p_save_dir}/{i}.jpg"
        image.save(output_dir)
    print("I2P done")
    
    mma_save_dir = os.path.join(save_dir,"mma")
    print(mma_save_dir)
    if not os.path.exists(mma_save_dir):
        os.makedirs(mma_save_dir)
    for i,item in tqdm(enumerate(mma_dataset)):
        prompt = item['prompt']
        seed = int(item['seed'])
        seed_everything(seed)
        guidance_scale = float(item['guidance'])
        num_inference_steps = 30
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    num_images_per_prompt=1,
                    height = 512,
                    width = 512).images
        image = images[0]
        output_dir = f"{mma_save_dir}/{i}.jpg"
        image.save(output_dir)
        # display(image)
    print("MMA-diffusion done")
    
    p4d_save_dir = os.path.join(save_dir,"p4d")
    print(save_dir)
    if not os.path.exists(p4d_save_dir):
        os.makedirs(p4d_save_dir)
    for i,item in tqdm(enumerate(p4d_dataset)):
        prompt = item['prompt']
        seed = int(item['seed'])
        guidance_scale = float(item['guidance'])
        num_inference_steps = 30
        seed_everything(seed)
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    num_images_per_prompt=1,
                    height = 512,
                    width = 512).images
        image = images[0]
        output_dir = f"{p4d_save_dir}/{i}.jpg"
        image.save(output_dir)
    print("P4D done")
    
    
    unlearn_save_dir = os.path.join(save_dir,"unlearn")
    print(unlearn_save_dir)
    if not os.path.exists(unlearn_save_dir):
        os.makedirs(unlearn_save_dir)
    for i,item in tqdm(enumerate(unlearn_dataset)):
        prompt = item['prompt']
        seed = int(item['seed'])
        guidance_scale = float(item['guidance'])
        num_inference_steps = 30
        seed_everything(seed)
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    num_images_per_prompt=1,
                    height = 512,
                    width = 512).images
        image = images[0]
        output_dir = f"{unlearn_save_dir}/{i}.jpg"
        image.save(output_dir)
    print("UnlearnDiff done")
    
    ring_save_dir = os.path.join(save_dir,"ring")
    print(ring_save_dir)
    if not os.path.exists(ring_save_dir):
        os.makedirs(ring_save_dir)
    for i,item in tqdm(enumerate(ring_dataset)):
        prompt = item['prompt']
        seed = int(item['seed'])
        guidance_scale = float(item['guidance'])
        num_inference_steps = 30
        images = pipe(prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    num_images_per_prompt=1,
                    height = 512,
                    width = 512).images
        image = images[0]
        output_dir = f"{ring_save_dir}/{i}.jpg"
        image.save(output_dir)
    print("Ring-A-Bell done")
    
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_nudity.py <unet_path> <save_dir>")
        sys.exit(1)

    unet_path = sys.argv[1]
    save_dir = sys.argv[2]
    main(unet_path,save_dir)
