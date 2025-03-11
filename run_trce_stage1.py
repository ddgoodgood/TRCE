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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer,text_encoder,unet = import_models_for_cfr(pretrained_model_name_or_path,device=device,dtype=torch.float32)
    augment = conf.augment.use_augment
    concept_num = conf.augment.concept_num
    augment_num = conf.augment.augment_num
    concepts = read_negative_concepts(conf.concept_file_path)[:concept_num]
    negative_concepts = [x[0]for x in concepts]
    target_concepts = [x[1]for x in concepts]

    negative_concepts = [prompt_augmentation(x,augment=augment,augment_num=augment_num) for x in negative_concepts]
    target_concepts = [prompt_augmentation(x,augment=augment,augment_num=augment_num) for x in target_concepts]
    dict_for_close_form = []
    for negative,target in zip(negative_concepts,target_concepts):
        for i in range(len(negative)):
            dict_for_close_form.append({"old": negative[i], "new": target[i]})
    len(dict_for_close_form)
    
    # prepare k,v for close_form_edit
    print("Preparing k,v for close_form_edit")
    projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
    contexts, valuess, weight_base = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                            dict_for_close_form, tokenizer,device=device)
    
    # conduct close_form_edit
    train_preserve_scale = conf.preserve_scale 
    prior_preservation_cache_path = conf.prior_cache_path
    prior_preservation_cache_dict = torch.load(prior_preservation_cache_path, map_location=projection_matrices[0].weight.device)

    CFR_dict = {}
    for layer_num in tqdm(range(len(projection_matrices))):
        CFR_dict[f'{layer_num}_for_mat1'] = .0
        CFR_dict[f'{layer_num}_for_mat2'] = .0 

    cache_dict = {}
    for key in CFR_dict:
        cache_dict[key] = train_preserve_scale * (prior_preservation_cache_dict[key]) + CFR_dict[key]
    
    task_name = conf.task_name
    # the weight for optimizing [SoT], first [EOT], last [EoT], [Key]
    weight_scale = conf.token_weight_scale
    projection_matrices, _, _ = get_ca_layers(unet, with_to_k=True)
    closed_form_refinement(projection_matrices, contexts, valuess, lamb=0.0, 
                                preserve_scale=train_preserve_scale, cache_dict=cache_dict, all_weight_base=weight_base, weight_scale=weight_scale)
    stage1_save_path = f"saved_ckpt/stage1/{task_name}"
    unet.save_pretrained(stage1_save_path)
    OmegaConf.save(conf, f"saved_ckpt/stage1/{task_name}/config.yaml")
    
    print("Stage1 done, saved at",stage1_save_path)
    
        
    
    