import copy
import torch
import numpy as np
from tqdm import tqdm
import gc

import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


def import_models_for_cfr(pretrained_model_name_or_path,device,dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder.to(device=device, dtype=dtype)
    unet.to(device,dtype=dtype)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    return tokenizer,text_encoder,unet

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")
       
           
def get_ca_layers(unet, with_to_k=True):

    sub_nets = unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ## get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
    
    return projection_matrices, ca_layers, og_matrices


def find_substring_indices(text, subtext):
    if subtext == "":
        return []
    text_words = text.split()
    subtext_words = subtext.split()
    sub_len = len(subtext_words)
    
    indices_list = []
    for i in range(len(text_words) - sub_len + 1):
        if text_words[i:i + sub_len] == subtext_words:
            indices_list = list(range(i, i + sub_len))
            break
    
    return [x+1 for x in indices_list]
    

@torch.no_grad()
def prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, test_set,
                     tokenizer,device, with_to_k=True, align_keyword = True):
    all_contexts, all_valuess, all_weight_base = [],[], []
    # test_set: list of dict{old:(sentence,keyword),new:(sentence,keyword)}
    for item in tqdm(test_set):
        gc.collect()
        torch.cuda.empty_cache()
        #### restart LDM parameters
        num_ca_clip_layers = len(ca_layers)
        for idx_, l in enumerate(ca_layers):
            l.to_v = copy.deepcopy(og_matrices[idx_])
            projection_matrices[idx_] = l.to_v
            if with_to_k:
                l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                projection_matrices[num_ca_clip_layers + idx_] = l.to_k
        #### prepare embeddings
        sentence_old,keyword_old = item["old"]
        sentence_new,_ = item["new"]
        sentence_combined = [sentence_old,sentence_new]
        tokenized_inputs = tokenizer(sentence_combined, padding="max_length",
                                     max_length=tokenizer.model_max_length, truncation=True, 
                                     return_tensors="pt")
        embedding_old, embedding_new = text_encoder(tokenized_inputs.input_ids.to(device))[0]
        
        # align [SoT], first [EoT], last [EoT]
        old_indices = [0,len(sentence_old.split())+1,76]
        new_indices = [0,len(sentence_new.split())+1,76]
        
        if align_keyword:
            keyword_old_indices = find_substring_indices(sentence_old,keyword_old)
            old_indices.extend(keyword_old_indices)
            new_indices.extend([-1]*len(keyword_old_indices))
        
        # prepare context-value
        context = embedding_old[old_indices].detach()
        values_all = [layer(embedding_new[new_indices]).detach() for layer in projection_matrices]
        
        weight_base = [1,1,77-len(sentence_old.split())-2, 1]
        all_contexts.append(context)
        all_valuess.append(values_all)
        all_weight_base.append(weight_base)
    return all_contexts, all_valuess, all_weight_base


@torch.no_grad()
def closed_form_refinement(projection_matrices, all_contexts=None, all_valuess=None, lamb=0.5, 
                           preserve_scale=1, cache_dict=None, all_weight_base=None, weight_scale = [1,1,1,1]):
    for layer_num in tqdm(range(len(projection_matrices))):
        gc.collect()
        torch.cuda.empty_cache()
        mat1 = lamb * projection_matrices[layer_num].weight
        mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=projection_matrices[layer_num].weight.device)
            
        total_for_mat1 = torch.zeros_like(projection_matrices[layer_num].weight)
        total_for_mat2 = torch.zeros_like(mat2)
        
        for contexts, valuess, weight_base in zip(all_contexts, all_valuess,all_weight_base):
            if len(weight_scale) < len(weight_base):
                weight_scale.extend([weight_scale[-1]] * (len(weight_base)-len(weight_scale)) )
                
            contexts_tensor = contexts.unsqueeze(-1)
            values_tensor = valuess[layer_num].unsqueeze(-1)
            for_mat1 = torch.bmm(values_tensor, contexts_tensor.permute(0, 2, 1))
            for_mat2 = torch.bmm(contexts_tensor, contexts_tensor.permute(0, 2, 1))
            for i in range(len(weight_base)):
                total_for_mat1 += weight_scale[i] * weight_base[i] * for_mat1[i]
                total_for_mat2 += weight_scale[i] * weight_base[i] * for_mat2[i]
        del for_mat1, for_mat2
        total_for_mat1 /= len(all_contexts)
        total_for_mat2 /= len(all_contexts)

        total_for_mat1 += preserve_scale * cache_dict[f'{layer_num}_for_mat1'] / 3e4
        total_for_mat2 += preserve_scale * cache_dict[f'{layer_num}_for_mat2'] / 3e4
        
        total_for_mat1 += mat1
        total_for_mat2 += mat2
        
        projection_matrices[layer_num].weight.data = total_for_mat1 @ torch.inverse(total_for_mat2) 
        
        del total_for_mat1, total_for_mat2
        

def importance_sampling_fn(t, temperature=0.05):
    """Importance Sampling Function f(t)"""
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))
        

def prompt_augmentation(content, augment=True, augment_num=30, concept_type='object'):
    if augment:
        # some sample prompts provided
        if concept_type == 'object':
            prompts = [
                # object augmentation
                ("{} in a photo".format(content), content),
                ("A snapshot of {}".format(content), content),
                ("A photograph showcasing {}".format(content), content),
                ("An illustration of {}".format(content), content),
                ("A digital rendering of {}".format(content), content),
                ("A visual representation of {}".format(content), content),
                ("A black and white image of {}".format(content), content),
                ("A scene depicting {} during a public gathering".format(content), content),
                ("A depiction created with oil paints capturing {}".format(content), content),
                ("An official photograph featuring {}".format(content), content),
                ("A detailed sketch of {}".format(content), content),
                ("{} during sunset/sunrise".format(content), content),
                ("Magazine cover capturing {}".format(content), content),
                ("An oil portrait of {}".format(content), content),
                ("{} in a sketch painting".format(content), content),
                ("A graphic of {}".format(content), content),
                ("A shot of {}".format(content), content),
                ("A photo of {}".format(content), content),
                ("A depiction in portrait form of {}".format(content), content),
                ("{} captured in an image".format(content), content),
                ("An image of {}".format(content), content),
                ("A drawing capturing the essence of {}".format(content), content),
                ("{} in a detailed portrait".format(content), content),
                ("An official photo of {}".format(content), content),
                ("Historic photo of {}".format(content), content),
                ("Detailed portrait of {}".format(content), content),
                ("A painting of {}".format(content), content),
                ("HD picture of {}".format(content), content),
                ("Painting-like image of {}".format(content), content),
                ("Hand-drawn art of {}".format(content), content),
            ]

        elif concept_type == 'style':
            # art augmentation
            prompts = [
                ("An artwork by {}".format(content), content),
                ("Art piece by {}".format(content), content),
                ("A recent creation by {}".format(content), content),
                ("{}'s renowned art".format(content), content),
                ("Latest masterpiece by {}".format(content), content),
                ("A stunning image by {}".format(content), content),
                ("An art in {}'s style".format(content), content),
                ("Exhibition artwork of {}".format(content), content),
                ("Art display by {}".format(content), content),
                ("a beautiful painting by {}".format(content), content),
                ("An image inspired by {}'s style".format(content), content),
                ("A sketch by {}".format(content), content),
                ("Art piece representing {}".format(content), content),
                ("A drawing by {}".format(content), content),
                ("Artistry showcasing {}".format(content), content),
                ("An illustration by {}".format(content), content),
                ("A digital art by {}".format(content), content),
                ("A visual art by {}".format(content), content),
                ("A reproduction inspired by {}'s colorful, expressive style".format(content), content),
                ("Famous painting of {}".format(content), content),
                ("A famous art by {}".format(content), content),
                ("Artistic style of {}".format(content), content),
                ("{}'s famous piece".format(content), content),
                ("Abstract work of {}".format(content), content),
                ("{}'s famous drawing".format(content), content),
                ("Art from {}'s early period".format(content), content),
                ("A portrait by {}".format(content), content),
                ("An imitation reflecting the style of {}".format(content), content),
                ("An painting from {}'s collection".format(content), content),
                ("Vibrant reproduction of artwork by {}".format(content), content),
                ("Artistic image influenced by {}".format(content), content),
            ] 
        else:
            raise ValueError("unknown concept type.")
    else: 
        prompts = [
            ("A photo of {}".format(content), content),
        ]
        
    return prompts[:augment_num]