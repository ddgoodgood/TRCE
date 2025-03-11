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

    # I2P
    i2p_file = "data/adv_prompts_csv/nudity-mma-diffusion.csv"
    i2p_dict = read_csv_as_dict(i2p_file)
    i2p_dataset = [{"prompt":item['prompt'],'seed':item['evaluation_seed'],'guidance':item['evaluation_guidance']} for item in i2p_dict]

    
    save_dir = os.path.join(save_dir,"unsafe")
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
        output_dir = f"{save_dir}/{i}.jpg"
        image.save(output_dir)
    print("I2P done")
    
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_unsafe.py <unet_path> <save_dir>")
        sys.exit(1)

    unet_path = sys.argv[1]
    save_dir = sys.argv[2]
    main(unet_path,save_dir)
