# FLIP: Flow-Centric Generative Planning for General-Purpose Manipulation Tasks

Official Code Repository for **FLIP: Flow-Centric Generative Planning for General-Purpose Manipulation Tasks**.

[Chongkai Gao](https://chongkaigao.com/)<sup>1</sup>, [Haozhuo Zhang](https://haozhuo-zhang.github.io/)<sup>2</sup>, [Zhixuan Xu](https://ariszxxu.github.io/)<sup>1</sup>, Zhehao Cai<sup>1</sup>, [Lin Shao](https://linsats.github.io/)<sup>1</sup>

<sup>1</sup>National University of Singapore, <sup>2</sup>Peking University

<p align="center">
    <a href='https://arxiv.org/abs/2412.08261'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://nus-lins-lab.github.io/flipweb/'>
      <img src='https://img.shields.io/badge/Project-Page-66C0FF?style=plastic&logo=Google%20chrome&logoColor=66C0FF' alt='Project Page'>
    </a>
</p>
<div align="center">
  <img src="imgs/teaser.png" alt="main" width="95%">
</div>

In this paper, we present FLIP, a model-based planning algorithm on visual space that features three key modules: 1. a multi-modal flow generation model as the general-purpose action proposal module; 2. a flow-conditioned video generation model as the dynamics module; and 3. a vision-language representation learning model as the value module. Given an initial image and language instruction as the goal, FLIP can progressively search for long-horizon flow and video plans that maximize the discounted return to accomplish the task. FLIP is able to synthesize long-horizon plans across objects, robots, and tasks with image flows as the general action representation, and the dense flow information also provides rich guidance for long-horizon video generation. In addition, the synthesized flow and video plans can guide the training of low-level control policies for robot execution.

## Installaltion

### 1. Create Python Environment

```
conda create -n flip python==3.8
conda activate flip
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Download CoTracker V2 Checkpoint
```
cd flip/co_tracker
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
```

### 4. Download Meta Llama 3.1 8B
1. Get the download access from https://huggingface.co/meta-llama/Llama-3.1-8B.
2. Put the downloaded folder at `./`. You should have a file structure like this:
```
...
- liv
- llama_models
- Meta-Llama-3.1-8B
  - consolidated.00.pth
  - params.json
  - tokenizer.model
- scripts
...
```

### 5. Download LIV Pretrained Models

1. Download the `model.pt` and `config.yaml` accroding to `https://github.com/penn-pal-lab/LIV/blob/main/liv/__init__.py#L33`.
2. `mkdir liv/resnet50`.
3. Put the `model.pt` and `config.yaml` under `liv/resnet50`. You should have a file structure like this:
```
...
- liv
  - cfgs
  - dataset
  - examples
  - models
  - resnet50
    - config.yaml
    - model.pt
  - utils
  __init__.py
  train_liv.py
  trainer.py
- llama_models
...
```

## Data Preparation

### 1. Download the LIBERO-LONG Dataset

1. `wget https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip`

2.  `mkdir data/libero_10`

3. unzip and put the 10 LIBERO-10 hdf5 files into `data/libero_10`

### 2. Replay

`python scripts/replay_libero_data_from_hdf5.py`

By default, the resolution is 128 $\times$ 128.

### 3. Flow Tracking

`python scripts/video_tracking.py`.

By default, we only track the agentview demos. You may change the `eye_in_hand` to `true` in the  `config/libero_10/tracking.yaml` to track the eye_in_hand demos.

### 4. Data Preprocessing

`python scripts/preprocess_data_to_hdf5.py`.

By default, we only preprocess the agentview demos. You may change the `eye_in_hand` to `true` in the  `config/libero_10/preprocess.yaml` to preprocess the eye_in_hand demos.


## Training

### 1. Train the Flow Generation Model (Action Module)

`torchrun --nnodes=1 --nproc_per_node=2 scripts/train_cvae.py`

You can change `config/libero_10/cvae.yaml` for custom training. Current config is for A100 40G GPUs.

### 2. Train the Video Generation Model (Dynamics Module)

`torchrun --nnodes=1 --nproc_per_node=2 scripts/train_dynamics.py`

You can change `config/libero_10/dynamics.yaml` for custom training. Current config is for A100 40G GPUs.

### 3. Finetune the LIV Model with Video Clips (Value Module)

`python scripts/finetune_liv.py`

This script will first make a liv dataset and then train on it.

You may change the configs in `config/libero_10/finetune_liv.yaml`, `liv/cfgs/dataset/libero_10.yaml`, and `liv/cfgs/training/finetune.yaml` according to your own tasks.

### 4. Finetune the Pretrained VAE Encoder (for the Dynamics Module)

`torchrun --nnodes=1 --nproc_per_node=8 scripts/finetune_vae.py`

You can change `config/libero_10/finetune_vae.yaml` for custom training.


## Testing

1. `makedir models/libero_10`

2. put all the trained models (agentview_dynamics.pt, cvae.pt, finetuned_vae.pt, reward.pt) under `models/libero_10`

3. `torchrun scripts/hill_climbing.py`


## Separate Testing of Action Module and Dynamics Module

1. Action Module: `python scripts/eval_cvae.py`.
2. Dynamics Module: `scripts/train_dynamics.py`.

## Citation

If you find our codes or models useful in your work, please cite [our paper](https://nus-lins-lab.github.io/flipweb/):

```
TODO
```


## Contact

If you have any questions, feel free to contact me through email ([gaochongkai@u.nus.edu](gaochongkai@u.nus.edu))!
