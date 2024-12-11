import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from flip.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from flip.network.dynamics import DynamicsModel_models
import argparse
from flip.dataloader.flip_dataloader import FLIPDataset
from torch.utils.data import DataLoader
from einops import rearrange
from utils.model_utils import load_models, ConfigArgs
from utils.yaml_utils import load_config_as_namespace
from utils.train_utils import save_video
from utils.data_utils import CustomBatchSampler

def main(cfg):
    assert torch.cuda.is_available(), "Evaluating requires at least one GPU."
    # Setup PyTorch:
    torch.manual_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device = "cuda:0"

    # Load model:
    model = DynamicsModel_models[cfg.model](
        flow_horizon=cfg.flow_horizon,
        obs_history=cfg.obs_history,
        input_size=cfg.img_size if not cfg.vae_encode else [cfg.img_size[0] // 8, cfg.img_size[1] // 8],
        in_channels=3 if not cfg.vae_encode else 4,
    ).to(device)
    state_dict = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_M = total_params / 1_000_000    
    print(f"Dynamics Model Parameters: {total_params_in_M}M")
    
    eval_diffusion = create_diffusion(timestep_respacing="", diffusion_steps=cfg.eval_diffusion_step, learn_sigma=cfg.learn_sigma)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    state_dict = torch.load(cfg.vae_ckpt_path, map_location=lambda storage, loc: storage)
    vae.load_state_dict(state_dict["model"])

    # Setup data:
    dataset = FLIPDataset(
        data_dir=cfg.data_dir,
        obs_history=cfg.obs_history,
        flow_horizon=cfg.flow_horizon,
        calculate_scale_and_direction=False,
        return_future_img=True,
        data_aug=None,
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

    
    with torch.no_grad():
        for i in range(cfg.test_clip_num):
            print(f"Processing clip {i}")
            batch_data = next(data_iter)
            
            observation = batch_data['observation'].to(device)
            sentence_embedding = batch_data['sentence_embedding'].to(device)

            # downsample flow points
            flow = batch_data['flow'].to(device)    # [B, N, T, 2]

            future_imgs = batch_data['future_img'].to(device)  # [B, T, C, H, W]
            video = torch.cat([observation, future_imgs], dim=1)  # [B, T, C, H, W]

            B, T, C, H, W = video.shape
            obs_condition = rearrange(observation, 'b t c h w -> (b t) c h w')
            obs_condition = vae.encode(obs_condition).latent_dist.sample().mul_(0.18215)
            obs_condition = rearrange(obs_condition, '(b t) c h w -> b t c h w', b=B)

            z = torch.randn(B, T, 4, H//8, W//8, device=device)
            z[:, :cfg.obs_history] = obs_condition

            model_kwargs = dict(
                flow=flow,
                sentence=sentence_embedding,
            )

            # Sample images:
            samples = eval_diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, flow_flag=True, history_length=cfg.obs_history
            )
            samples = rearrange(samples, 'b t c h w -> (b t) c h w')
            samples = vae.decode(samples / 0.18215).sample
            samples = rearrange(samples, '(b t) c h w -> b t c h w', b=B)
            
            # Save and display images:
            save_video(samples.clone(), cfg.results_dir, normalize=True, value_range=None, video_name=[f"clip_{i}"])
            
            for _ in range(cfg.flow_horizon-1):
                data_sample = next(data_iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/dynamics_eval.yaml", help="Path to the config file")
    cfg = parser.parse_args()

    cfg = load_config_as_namespace(cfg.config)
    main(cfg)