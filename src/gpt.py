from torch.utils.data import Dataset
from src.cfr_utils import *
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import os
from openai import OpenAI
import regex as re


BASE_URL = ''
API_KEY = ''


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(
        r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(
        r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', '') for x in class_prompt_collection]
    return class_prompt_collection


def text_augmentation(erased_concept, mapping_concept, concept_type, num_text_augmentations=100):
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    
    class_prompt_collection = []

    if concept_type == 'object':
        messages = [
            {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
            {"role": "user", "content": f"Generate {num_text_augmentations} captions for images containing {erased_concept}. The caption should also contain the word '{erased_concept}'. Please do not use any emojis in the captions."},
        ]
        
        while True:
            completion = client.chat.completions.create(
                model="gpt-4-o",
                messages=messages,
            )
            class_prompt_collection += [x for x in completion.choices[0].message.content.lower(
            ).split('\n') if erased_concept in x]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content})
            messages.append(
                {"role": "user", "content": f"Generate {num_text_augmentations-len(class_prompt_collection)} more captions"})
            if len(class_prompt_collection) >= num_text_augmentations:
                break
            
        class_prompt_collection = clean_prompt(class_prompt_collection)[:num_text_augmentations]
        class_prompt_formated = []
        mapping_prompt_formated = []
        
        for prompt in class_prompt_collection:
            class_prompt_formated.append((prompt, erased_concept))
            mapping_prompt_formated.append((prompt.replace(erased_concept, mapping_concept), mapping_concept))
    
        return class_prompt_formated, mapping_prompt_formated