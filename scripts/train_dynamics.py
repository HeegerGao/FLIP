import sys
sys.path.append('.')
sys.path.append('..')
import argparse
from copy import deepcopy
from glob import glob
from time import time
import wandb
from einops import rearrange

from diffusers.models import AutoencoderKL

from flip.dataloader.flip_dataloader import FLIPDataset
from flip.network.dynamics import DynamicsModel_models
from flip.diffusion import create_diffusion

from utils.yaml_utils import load_config_as_namespace

def main(cfg):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert cfg.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = cfg.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        if cfg.wandb:
            wandb.init(
                project="Dynamics Model Training",
                config=vars(cfg),
            )
        # Setup an experiment folder:
        os.makedirs(cfg.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.results_dir}/*"))
        model_string_name = cfg.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if cfg.eval:
            eval_diffusion = create_diffusion(timestep_respacing="", diffusion_steps=cfg.eval_diffusion_step, learn_sigma=cfg.learn_sigma)
            eval_dir = f"{experiment_dir}/eval"  # Stores evaluation results
            os.makedirs(eval_dir, exist_ok=True)
    else:
        logger = create_logger(None)

    # Create model:
    model = DynamicsModel_models[cfg.model](
        flow_horizon=cfg.flow_horizon,
        obs_history=cfg.obs_history,
        input_size=cfg.img_size if not cfg.vae_encode else [cfg.img_size[0] // 8, cfg.img_size[1] // 8],
        in_channels=3 if not cfg.vae_encode else 4,
    )
    if cfg.load_ckpt:
        state_dict = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        model.load_state_dict(state_dict)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=cfg.learn_sigma)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    logger.info(f"Dynamics Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0)

    dataset = FLIPDataset(
        cfg.data_dir, 
        cfg.obs_history,
        cfg.flow_horizon,
        calculate_scale_and_direction=False,
        return_future_img=True,
        data_aug=None,
        flow_aug=cfg.flow_aug,
        eye_in_hand=cfg.eye_in_hand,
    )
        
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.global_batch_size // dist.get_world_size()),
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Dataset contains {len(dataset):,} obs-flow-language pairs ({cfg.data_dir})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {cfg.epochs} epochs...")
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_data in loader:
            observation = batch_data['observation'].to(device)
            sentence_embedding = batch_data['sentence_embedding'].to(device)

            # downsample flow points
            flow = batch_data['flow'].to(device)    # [B, N, T, 2]
            N_new = torch.randint(cfg.min_track_pt, cfg.max_track_pt + 1, (1,)).item()
            indices = torch.randperm(flow.size(1))[:N_new]
            flow = flow[:, indices, :, :]    # [B, N_new, T, 2]

            future_imgs = batch_data['future_img'].to(device)  # [B, T, C, H, W]
            video = torch.cat([observation, future_imgs], dim=1)  # [B, T, C, H, W]

            if cfg.vae_encode:
                with torch.no_grad():
                    x = rearrange(video, 'b t c h w -> (b t) c h w')
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x = rearrange(x, '(b t) c h w -> b t c h w', b=video.shape[0])
            else:
                x = video
            B, T, C, H, W = x.shape

            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            model_kwargs = dict(
                flow=flow,
                sentence=sentence_embedding
            )

            noise = torch.randn_like(x)
            noise[:, :cfg.obs_history] *= 0.

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs, noise=noise, flow_flag=True, history_len=cfg.obs_history)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if rank == 0:
                if cfg.wandb:
                    wandb.log(
                        {
                            "loss": loss.item()
                        }
                    )

            if train_steps % cfg.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % cfg.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": cfg
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if cfg.eval:
                        with torch.no_grad():
                            # Create sampling noise:
                            if cfg.vae_encode:
                                obs_condition = rearrange(observation, 'b t c h w -> (b t) c h w')
                                obs_condition = vae.encode(obs_condition).latent_dist.sample().mul_(0.18215)
                                obs_condition = rearrange(obs_condition, '(b t) c h w -> b t c h w', b=B)
                            else:
                                obs_condition = observation

                            z = torch.randn(B, T, C, H, W, device=device)
                            z[:, :cfg.obs_history] = obs_condition

                            model_kwargs = dict(
                                flow=flow,
                                sentence=sentence_embedding,
                            )
                            # Sample images:
                            samples = eval_diffusion.p_sample_loop(
                                ema.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, flow_flag=True, history_length=cfg.obs_history
                            )
                            latent_l2_loss = torch.nn.MSELoss()(x[:, cfg.obs_history:], samples[:, cfg.obs_history:]).item()
                            if cfg.vae_encode:
                                samples = rearrange(samples, 'b t c h w -> (b t) c h w')
                                samples = vae.decode(samples / 0.18215).sample
                                samples = rearrange(samples, '(b t) c h w -> b t c h w', b=B)
                            
                            # Save and display images:
                            save_video(samples.clone(), os.path.join(eval_dir, f"{train_steps}"), normalize=True, value_range=(-1, 1))

                            if cfg.wandb:
                                wandb.log(
                                    {
                                        "latent l2 loss": latent_l2_loss,
                                        "epoch": epoch,
                                    }
                                )
                dist.barrier()

    model.eval()

    logger.info("Done!")
    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc_per_node=6 scripts/train_dynamics.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/dynamics.yaml", help="Path to the config file")
    args = parser.parse_args()

    cfg = load_config_as_namespace(args.config)
    if hasattr(cfg, "lr"):
        cfg.lr = float(cfg.lr)
     
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from utils.train_utils import create_logger, requires_grad, update_ema, save_video

    main(cfg)
