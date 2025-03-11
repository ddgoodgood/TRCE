# TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models


<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2503.07389" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2503.07389-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/ddgoodgood/trce-erased-model' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/ddgoodgood/TRCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>

</div>



## Setup

The environment we conduct experiments are as follows:
+ python: 3.10
+ torch: 2.1.2
+ CUDA Version: 12.4

Please run `pip install -r requirement.txt` to install dependency packages.

The erased model can be found :hugs:[here](https://huggingface.co/ddgoodgood/trce-erased-model/tree/main). Currently, our implementation is based only on SD1.4. We will release the implementation of TRCE on newer model in the future.

## RUN

You can find the pre-cached COCO embeddings :hugs:[here](https://huggingface.co/ddgoodgood/trce-erased-model/tree/main). Please download the `cache` directory and place it in `data/cache`.

### Run stage-1 TRCE

In the first stage, TRCE starts with a closed-form edit for the cross-attention layers, simply run:

``` bash
# for erasing "sexual"
python run_trce_stage1.py config/stage1/stage1_sexual_default.yaml

# for erasing multiple malicious concepts
python run_trce_stage1.py config/stage1/stage1_unsafe_default.yaml
```

You can modify the base model path and the output directory for the first-stage fine-tuned model in the configuration files.

### Run stage-2 TRCE

Before the second stage, you need to prepare the denosing trajectory samples for the fine-tuning:

```bash
python stage2_data_preparation.py
```

This script generates samples for both "sexual" and "multi-concept" fine-tuning, as well as unconditional samples for the regularization loss.

Then, you can run the stage-2 using the following scripts:
``` bash
# for erasing "sexual"
python run_trce_stage2.py config/stage2/stage2_sexual_default.yaml

# for erasing multiple malicious concepts
python run_trce_stage2.py config/stage2/stage2_unsafe_default.yaml
```

## Evaluation
The evaluation relies on the following repositories: [NudeNet](https://github.com/notAI-tech/NudeNet), [Q16 Detector](https://github.com/ml-research/Q16), [Pytorch FID](https://github.com/mseitzer/pytorch-fid), and [CLIP Score](https://github.com/Taited/clip-score). Please install these repositories according to their instructions before proceeding with the evaluation.

### Generate image using erased model

Firstly, use the following scripts with the specified UNet path and output path to generate images for different evaluation tasks.
```
# for evaluate "sexual" erasure
python gen_sexual.py <erased-model-dir> <output_path>

# for evaluate "multi concepts" erasure
python gen_unsafe.py <erased-model-dir> <output_path>

# for evaluate knowledge preservation on coco
python gen_coco.py <erased-model-dir> <output_path>
```

Then, you can follow the instructions in `eval_nudenet_batch.ipynb`,  `eval_unsafe.ipynb` and `eval_coco_batch.ipynb` to evaluate and review the performance of the erasure.

If you encounter any issues while using this repository, please feel free to leave messages in issues or contact me at chenruidong@tju.edu.cn. I will respond as soon as possible.

## Citation
```
@article{chen2025reliable,
    title={TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models}, 
    author={Ruidong, Chen and Honglin, Guo and Lanjun, Wang and Chenyu, Zhang and Weizh, Nie and An-An, Liu},
    journal={arXiv preprint arXiv:2503.07389},
    year={2025}
}
```
## Acknowledgement
We built this repository based on the excellent work of previous projects: [RECE](https://github.com/CharlesGong12/RECE/tree/main), [MACE](https://github.com/Shilin-LU/MACE), and [Safree](https://github.com/jaehong31/SAFREE). Thank you to all who contributed.
