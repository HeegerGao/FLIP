import sys
sys.path.append('.')
sys.path.append('..')
import os
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from glob import glob
from time import time

from flip.dataloader.flip_dataloader import FLIPDataset
from flip.network.cvae import CVAE_models
from utils.model_utils import load_models, ConfigArgs
from utils.yaml_utils import load_config_as_namespace
from utils.vis_utils import vis_pred_flow, connect_and_save_videos
from utils.data_utils import CustomBatchSampler
from utils.module_utils import post_process_video
    
def main(cfg):
    assert torch.cuda.is_available(), "Evaluating requires at least one GPU."
    torch.set_grad_enabled(False)
    device = cfg.device

    seed = cfg.seed
    torch.manual_seed(seed)
    
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Create model:
    model = CVAE_models[cfg.model](
        flow_horizon=cfg.flow_horizon,
        obs_history=cfg.obs_history,
        img_size=cfg.img_size,
        img_patch_size=cfg.img_patch_size,
    ).to(device)
    ckpt_path = cfg.ckpt
    state_dict = load_models(ckpt_path)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_M = total_params / 1_000_000
    print(f"CVAE Model Parameters: {total_params_in_M}M")

    dataset = FLIPDataset(
        data_dir=cfg.data_dir,
        obs_history=cfg.obs_history,
        flow_horizon=cfg.flow_horizon,
        calculate_scale_and_direction=cfg.calculate_scale_and_direction,
        return_future_img=False,
        data_aug=cfg.data_aug,
        eye_in_hand=cfg.eye_in_hand,
    )
    
    indices = list(range(cfg.start_index, len(dataset)))
    sampler = CustomBatchSampler(indices, 1)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    data_iter = iter(loader)
    print(f"Dataset contains {len(dataset):,} obs-flow-language pairs ({cfg.data_dir})")

    # sample grid
    x = torch.linspace(0, cfg.img_size[1], int(cfg.grid_pt_num ** 0.5))
    y = torch.linspace(0, cfg.img_size[0], int(cfg.grid_pt_num ** 0.5))
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    query_points = torch.stack([xx.ravel(), yy.ravel()], dim=-1)
    query_points = query_points.unsqueeze(0).to(device)  # [1, N, 2]

    with torch.no_grad():
        video_frames = []
        flow_video_clip = []
        for i in range(cfg.test_clip_num):
            print(f"Processing video clip {i}")
            data_sample = next(data_iter)
            
            observation = data_sample['observation'].to(device)
            flow = data_sample['flow'].to(device)    # [B, N, T, 2]
            sentence_embedding = data_sample['sentence_embedding'].to(device)

            video_frames.append(observation[0][-1])

            # use ground truth query points if needed
            if cfg.use_gt_query_points:
                query_points = flow[:, :, 0]
                selected_index = torch.randperm(query_points.shape[1])[:cfg.grid_pt_num]
                query_points = query_points[:, selected_index]

            z_mu, z_logvar = torch.zeros([observation.shape[0], model.hidden_size]).to(device), torch.zeros([observation.shape[0], model.hidden_size]).to(device)
            z = torch.distributions.normal.Normal(z_mu, torch.exp(torch.exp(0.5 * z_logvar))).rsample().to(device)
            
            if cfg.use_z_mu:
                scale, direction = model.inference(query_points, observation, sentence_embedding, z_mu)
            else:
                scale, direction = model.inference(query_points, observation, sentence_embedding, z)
            predicted_flow = model.reconstruct_flow(query_points, scale, direction)

            # draw the predicted flow on video
            flow_video_clip.append(vis_pred_flow(flow=predicted_flow, init_img=post_process_video(observation[0][-1])))

            for _ in range(cfg.flow_horizon-1):
                data_sample = next(data_iter)
                video_frames.append(data_sample['observation'][0][-1].to(device))

        video_frames = torch.stack(video_frames)
        video_frames = post_process_video(video_frames, normalize=True)
        connect_and_save_videos(flow_video_clip, video_frames, os.path.join(cfg.results_dir, f"cvae_eval.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/cvae_eval.yaml", help="Path to the config file")
    cfg = parser.parse_args()

    cfg = load_config_as_namespace(cfg.config)
    main(cfg)