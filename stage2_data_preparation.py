import random
import os
import torch
import torch.utils.checkpoint
from diffusers import DDIMScheduler

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import pickle

from src.cfr_utils import *
from src.gen_utils import *
from src.tools import *

def get_embeddings(prompts):
    batch_size = len(prompts)
    
    text_input = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    if True:
      max_length = text_input.input_ids.shape[-1]
      uc_text = ""
      unconditional_input = tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
      unconditional_embeddings = text_encoder(unconditional_input.input_ids.to(device))[0]
      # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
    return text_embeddings, unconditional_embeddings

def gen_image(latents,prompts,guidance_scale = 3,num_inference_steps = 50):
    batch_size = len(prompts)
    
    text_input = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    if guidance_scale > 1.:
      max_length = text_input.input_ids.shape[-1]
      uc_text = ""
      unconditional_input = tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
      unconditional_embeddings = text_encoder(unconditional_input.input_ids.to(device))[0]
      # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
    
    scheduler.set_timesteps(num_inference_steps)
    
    results = []
    for i,t in enumerate(scheduler.timesteps):
      noise_pred_con = unet(latents, t, encoder_hidden_states=text_embeddings).sample
      if guidance_scale > 1.:
        with torch.no_grad():
          noise_pred_uncon = unet(latents, t, encoder_hidden_states=unconditional_embeddings).sample
        noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
      else:
        noise_pred = noise_pred_con
      latents,pred_x0 = step(scheduler, noise_pred, t, latents)
      results.append([latents.detach().cpu(),noise_pred.detach().cpu(), t])
      # break
    return latents,results

if __name__ == "__main__":
    dtype= torch.float32
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    # model initialization
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    text_encoder.to(device,dtype=dtype)
    vae.to(device,dtype=dtype)
    unet.to(device,dtype=dtype)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    ### generate trajectory for "sexual"
    print("Generating trajectory for sexual")
    sexual_concepts_num = 20
    concepts = read_negative_concepts("data/concepts/sexual.json")[:sexual_concepts_num]
    prompt_template = "a photo of {}"
    images_per_prompt = 5
    
    output_dir = "./tmp-data/sexual"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pbar = tqdm(total=sexual_concepts_num * images_per_prompt)
    for i,concept in enumerate(concepts):
        neg_concept = concept[0]
        safe_concept = concept[1]
        neg_prompt = prompt_template.format(neg_concept)
        safe_prompt = prompt_template.format(safe_concept)
        for j in range(images_per_prompt):
            seed = random.randint(0,10000000)
            seed_everything(seed)
            text_embeddings, unconditional_embeddings = get_embeddings([neg_prompt])
            input_latents = torch.randn([1, 4, 64, 64], device=device,dtype=dtype)
            pred_final,pred_list = gen_image(input_latents,[neg_prompt],guidance_scale=3,num_inference_steps=50,)
            data = {
                "input":input_latents.cpu().numpy(),
                "pred_list":pred_list,
                "neg_prompt":neg_prompt,
                "safe_prompt":safe_prompt,
            }
            filename = f'{output_dir}/{i}_{j}_{seed}.p'
            pickle.dump(data, open(filename, 'wb'))
            pbar.update(1)
    pbar.close()
    
    # generate trajectory for multiple malicious concepts
    print("Generating trajectory for multiple malicious concepts")
    multi_concepts_num = 60
    concepts = read_negative_concepts("data/concepts/unsafe.json")[:multi_concepts_num]
    prompt_template = "an image depicting {}"
    images_per_prompt = 5
    
    output_dir = "./tmp-data/unsafe"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pbar = tqdm(total=multi_concepts_num * images_per_prompt)
    for i, concept in enumerate(concepts):
        neg_concept = concept[0]
        safe_concept = concept[1]
        neg_prompt = prompt_template.format(neg_concept)
        safe_prompt = prompt_template.format(safe_concept)
        for j in range(images_per_prompt):
            seed = random.randint(0, 10000000)
            seed_everything(seed)
            text_embeddings, unconditional_embeddings = get_embeddings([neg_prompt])
            input_latents = torch.randn([1, 4, 64, 64], device=device, dtype=dtype)
            pred_final, pred_list = gen_image(input_latents, [neg_prompt], guidance_scale=3, num_inference_steps=50)
            data = {
                "input": input_latents.cpu().numpy(),
                "pred_list": pred_list,
                "neg_prompt": neg_prompt,
                "safe_prompt": safe_prompt,
            }
            filename = f'{output_dir}/{i}_{j}_{seed}.p'
            pickle.dump(data, open(filename, 'wb'))
            pbar.update(1)
    pbar.close()

    # prepare unconditional sampling for regularization term
    print("Generating unconditional samples")
    import pickle
    output_dir = "./tmp-data/regularization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    uncond_num = 2000
    for i in tqdm(range(uncond_num)):
        seed = random.randint(0,10000000)
        seed_everything(seed)
        input_latents = torch.randn([1, 4, 64, 64], device=device,dtype=dtype)
        pred_final,pred_list = gen_image(input_latents,[""],guidance_scale=7.5,num_inference_steps=50)
        data = {
            "input":input_latents.cpu().numpy(),
            "pred_list":pred_list,
        }
        filename = f'{output_dir}/{i}_{seed}.p'
        pickle.dump(data, open(filename, 'wb')) 
    
    print("Done")

