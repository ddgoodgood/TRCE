import torch.nn.functional as F
import torch.utils.checkpoint

import sys
import random
import numpy as np
import pickle
from omegaconf import OmegaConf


from src.tools import *
from src.cfr_utils import *
from src.gen_utils import *


if __name__ == "__main__":
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    
    # stage-1 close_form_edit
    dtype= torch.float32
    pretrained_model_name_or_path = conf.pretrained_model_name_or_path
    pretrained_model_name_or_path = "/data1/cdd/checkpoints/stable-diffusion-v1-4"
    stage1_save_path = conf.stage1_save_path
    
    # stage-2 fine-tuning
    unet_ft = UNet2DConditionModel.from_pretrained(stage1_save_path)
    # continue fine-tuning stage1 model
    # using original u-net as reference model
    unet_ref = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    # reloading stage1 model for regularization
    unet_reg = UNet2DConditionModel.from_pretrained(stage1_save_path)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    text_encoder.to(device)
    unet_ft.to(device)
    unet_ref.to(device)
    unet_reg.to(device)
    text_encoder.requires_grad_(False)
    unet_ref.requires_grad_(False)
    unet_reg.requires_grad_(False)
    
    # select trainable params
    unet_trainable_params = []
    unet_non_trainable_params = []
    for name, param in unet_ft.named_parameters():
        if ('attn1' in name) or (('attn2' in name) and ('to_q' in name)):
            unet_trainable_params.append(param)
        else:
            unet_non_trainable_params.append(param)
            param.requires_grad_(False)
    
    lr = conf.learning_rate
    
    optimizer = torch.optim.AdamW(
        [{'params': unet_trainable_params}],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # set training data
    neg_input_path = conf.neg_input_path
    safe_input_path = conf.safe_input_path

    file_list_neg = os.listdir(neg_input_path)
    file_list_neg = sorted(file_list_neg, key=lambda x: int(x.split("_")[0]))

    file_list_safe = os.listdir(safe_input_path)
    file_list_safe = sorted(file_list_safe, key=lambda x: int(x.split("_")[0]))

    epochs = conf.epochs
    guidance = conf.guidance
    preserve_scale = conf.preserve_scale
    task_name = conf.task_name
    loss_curve = []
    loss_target_total = 0
    loss_preserve_total = 0
    beta = conf.beta
    guidance = conf.guidance
    margin = conf.margin
    max_early_timestep = conf.max_early_timestep
    for epoch in tqdm(range(epochs)):
        print(f"Traning Epoch:{epoch+1}")
        pbar = tqdm(enumerate(file_list_neg))
        for i, (neg_file) in pbar:
            # calculate target loss
            neg_file = os.path.join(neg_input_path, neg_file)
            item = pickle.load(open(neg_file, "rb"))
            pred_list = item["pred_list"]
            
            neg_prompt = item["neg_prompt"]
            safe_prompt = item["safe_prompt"]
            
            with torch.no_grad():
                uncond_embedding = get_text_embedding(tokenizer, text_encoder, "", device)
                neg_embedding = get_text_embedding(tokenizer, text_encoder, neg_prompt, device)
                safe_embedding = get_text_embedding(tokenizer, text_encoder, safe_prompt, device)
            
            
            sample_t = random.randint(0, max_early_timestep)
            t = torch.randint((49-sample_t)*20, (50-sample_t)*20, (1,), device=device,dtype=torch.int64)
            t = t.to(device)
            try:
                latent, noise_pred, t = pred_list[sample_t]
            except:
                continue
            latent = latent.to(device,dtype=dtype)
            
            
            # compute reference score
            with torch.no_grad():
                neg_ref = unet_ref(latent, t, encoder_hidden_states = neg_embedding).sample
                neg_ref.requires_grad_(False)
                
                safe_ref = unet_ref(latent, t, encoder_hidden_states = safe_embedding).sample
                safe_ref.requires_grad_(False)
                uncond_ref = unet_ref(latent, t, encoder_hidden_states = uncond_embedding).sample
                uncond_ref.requires_grad_(False)

                prefer_ref = safe_ref + guidance * (safe_ref - uncond_ref)
                reject_ref = neg_ref + guidance * (neg_ref - uncond_ref)

            # compute loss for conditional prediction
            neg_ft = unet_ft(latent, t, encoder_hidden_states = neg_embedding).sample
            prefered_score = F.mse_loss(neg_ft, prefer_ref, reduction = 'none').mean(dim=(1, 2, 3))
            reject_score = F.mse_loss(neg_ft, reject_ref, reduction = 'none').mean(dim=(1, 2, 3))
            logits = beta * ( prefered_score - reject_score + margin)
            logits = torch.clamp(logits, min=0)
            
            loss_target_neg = logits
            
            # compute loss for unconditional prediction
            uncond_ft = unet_ft(latent, t, encoder_hidden_states = uncond_embedding).sample
            preferend_score_uncond = F.mse_loss(uncond_ft, prefer_ref, reduction = 'none').mean(dim=(1, 2, 3))
            reject_score_uncond = F.mse_loss(uncond_ft, reject_ref, reduction = 'none').mean(dim=(1, 2, 3))
            logits_uncond = beta * (preferend_score_uncond - reject_score_uncond + margin)
            logits_uncond = torch.clamp(logits_uncond, min=0)
            
            loss_target_uncond = logits_uncond
            
            loss_target = (loss_target_neg + loss_target_uncond)/2

            # regularization term
            reg_batch_size = conf.regularization_batch_size
            preserve_loss_sum = 0.01

            safe_indices = [(i*reg_batch_size + j) for j in range(reg_batch_size)]
            batch_safe_files = [file_list_safe[idx % len(file_list_safe)] for idx in safe_indices]

            for safe_file in batch_safe_files:
                safe_file_path = os.path.join(safe_input_path, safe_file)
                safe_item = pickle.load(open(safe_file_path, "rb"))
                safe_pred_list = safe_item["pred_list"]
                
                sample_t = random.randint(0, 50)
                t = torch.randint((49-sample_t)*20, (50-sample_t)*20, (1,), device=device,dtype=torch.int64)
                t = t.to(device)
                
                try:
                    safe_latent, safe_noise_pred, safe_t = safe_pred_list[sample_t]
                except:
                    continue
                    
                safe_latent = safe_latent.to(device, dtype=dtype)
                
                with torch.no_grad():
                    safe_prev_ref = unet_reg(safe_latent, t, encoder_hidden_states=uncond_embedding).sample
                    safe_prev_ref.requires_grad_(False)
                    
                safe_prev_ft = unet_ft(safe_latent, t, encoder_hidden_states=uncond_embedding).sample
                batch_preserve_loss = F.mse_loss(safe_prev_ft, safe_prev_ref)
                preserve_loss_sum += batch_preserve_loss

            loss_preserve = preserve_loss_sum / reg_batch_size
            
            loss = loss_target + preserve_scale * loss_preserve
            
            loss.backward()
            optimizer.step()
            
            loss_target_total += loss_target.item()
            loss_preserve_total += loss_preserve.item()
            loss_curve.append([loss_target_total, loss_preserve_total])
            pbar.set_postfix_str(f"Iteration:{i}, Loss_target:{loss_target_total/len(loss_curve)}, Loss_preserve:{loss_preserve_total/len(loss_curve)}")
            if (i+1) % 50 == 0 or i ==0:
                print(f"Iteration:{i}, Loss_target:{loss_target_total/len(loss_curve)}, Loss_preserve:{loss_preserve_total/len(loss_curve)}")
        
        
    ckpt_path = f"saved_ckpt/stage2/{task_name}/{epoch}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    unet_ft.save_pretrained(ckpt_path)
    OmegaConf.save(conf, os.path.join(ckpt_path, "config.yaml"))
    print(f"Model saved at {ckpt_path}")
        
    
    