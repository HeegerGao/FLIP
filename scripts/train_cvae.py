import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import numpy as np
from glob import glob
from time import time
import wandb

from flip.dataloader.flip_dataloader import FLIPDataset
from flip.network.cvae import CVAE_models
from utils.model_utils import load_models
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

    seed = cfg.global_seed
    torch.manual_seed(seed)

    # Setup an experiment folder:
    if rank == 0:
        if cfg.wandb:
            wandb.init(
                project="CVAE Flow Prediction",
                config=vars(cfg),
            )
        os.makedirs(cfg.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.results_dir}/*"))
        model_string_name = cfg.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = CVAE_models[cfg.model](
        flow_horizon=cfg.flow_horizon,
        obs_history=cfg.obs_history,
        img_size=cfg.img_size,
        img_patch_size=cfg.img_patch_size,
    )

    if cfg.load_ckpt:
        ckpt_path = cfg.ckpt
        state_dict = load_models(ckpt_path)
        model.load_state_dict(state_dict["model"])

    model = DDP(model.to(device), device_ids=[rank])
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_M = total_params / 1_000_000
    logger.info(f"CVAE Model Parameters: {total_params_in_M}M")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0)

    dataset = FLIPDataset(
        data_dir=cfg.data_dir,
        obs_history=cfg.obs_history,
        flow_horizon=cfg.flow_horizon,
        calculate_scale_and_direction=cfg.calculate_scale_and_direction,
        return_future_img=False,
        data_aug=cfg.data_aug,
        eye_in_hand=cfg.eye_in_hand,
    )

    if cfg.eval:
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        train_dataset = dataset

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.global_batch_size // dist.get_world_size()),
        sampler=train_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if cfg.eval:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.global_batch_size // dist.get_world_size()),
            sampler=test_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False
        )

    logger.info(f"Dataset contains {len(dataset):,} obs-flow-language pairs ({cfg.data_dir})")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {cfg.epochs} epochs...")
    skip_time = 0
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_data in train_loader:
            model.train()
            # select random points from [cfg.min_point_num, cfg.track_point_num]
            point_num = torch.randint(cfg.min_point_num, cfg.max_point_num, (1,)).item()
            indices = torch.randperm(cfg.track_point_num)[:point_num]

            observation = batch_data['observation'].to(device)  
            flow = batch_data['flow'][:, indices].to(device)    # [B, N, T, 2]
            full_scale = batch_data['scale'].to(device)
            full_direction = batch_data['direction'].to(device)
            sentence_embedding = batch_data['sentence_embedding'].to(device)

            scale, direction, z_mu, z_logvar, img_output, recon_flow = model(flow, observation, sentence_embedding)

            if cfg.weight_loss:
                direction_loss = model.module.direction_loss(direction, full_direction[:, indices], scale_weight=full_scale[:, indices], weight=cfg.weight)
                scale_loss = model.module.scale_loss(scale, full_scale[:, indices], scale_weight=full_scale[:, indices], weight=cfg.weight)
                _, ade_loss = model.module.average_distance_error(flow.clone(), scale, direction, scale_weight=full_scale[:, indices], weight=cfg.weight)
            else:
                direction_loss = model.module.direction_loss(direction, full_direction[:, indices])
                scale_loss = model.module.scale_loss(scale, full_scale[:, indices])
                _, ade_loss = model.module.average_distance_error(flow.clone(), scale, direction)
   
            kl_loss = model.module.kl_loss(z_mu, z_logvar)

            img_recon_loss = model.module.img_recon_loss(img_output, observation)
            flow_recon_loss = model.module.flow_recon_loss(recon_flow, flow.clone())
                
            if cfg.aux:
                aux_weight = 1e-5
            else:
                aux_weight = 0.
            
            loss = 10.0 * direction_loss + 10.0 * scale_loss + 1e-4 * kl_loss + ade_loss + aux_weight * img_recon_loss + aux_weight * flow_recon_loss

            opt.zero_grad()
            loss.backward()

            # calculate the grad norm and record it
            total_norm = monitor_gradient_norm(model)
            if np.isnan(total_norm) or np.isinf(total_norm):
                logger.info(f"Non-finite gradient norm found in batch. Skipping {skip_time}-th...")
                skip_time += 1
                continue

            # clip the grad norm
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)

            opt.step()

            torch.cuda.empty_cache()

            # Log loss values:
            running_loss += loss.item()

            if rank == 0:
                if cfg.wandb:
                    wandb.log(
                        {
                            "direction_loss": direction_loss.item(),
                            "scale_loss": scale_loss.item(),
                            "kl_loss": kl_loss.item(),
                            "traj_level_loss": ade_loss.item(),
                            "img_recon_loss": img_recon_loss.item() if cfg.aux else 0.,
                            "flow_recon_loss": flow_recon_loss.item() if cfg.aux else 0.,
                            "loss": loss.item(),
                            "z_mu": z_mu.mean().item(),
                            "z_logvar": z_logvar.mean().item(),
                            "grad_norm": total_norm,
                        }
                    )

            if train_steps % cfg.log_every == 0 and log_steps > 0:
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
            if train_steps % cfg.ckpt_every == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": cfg
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                if cfg.eval:
                    model.eval()
                    with torch.no_grad():
                        eval_loss = 0.
                        eval_ratio = 0.
                        eval_samples = 0
                        for batch_data in test_loader:
                            observation = batch_data['observation'].to(device)
                            flow = batch_data['flow'][:, indices].to(device)    # [B, N, T, 2]
                            sentence_embedding = batch_data['sentence_embedding'].to(device)
                            full_scale = batch_data['scale'].to(device)

                            # eval mode, sample z, then inference
                            query_points = flow[:, :, 0]
                            z_mu, z_logvar = torch.zeros([observation.shape[0], model.module.hidden_size]).to(observation.device), torch.zeros([observation.shape[0], model.module.hidden_size]).to(observation.device)
                            z = torch.distributions.normal.Normal(z_mu, torch.exp(torch.exp(0.5 * z_logvar))).rsample().to(observation.device)
                            scale, direction = model.module.inference(query_points, observation, sentence_embedding, z)
                            ade_keepdim, ade = model.module.average_distance_error(flow.clone(), scale, direction, scale_weight=full_scale[:, indices], eval_mode=True)
                            average_delta_ratio = model.module.less_than_delta(ade_keepdim)
                            eval_loss += ade.item()
                            eval_ratio += average_delta_ratio.item() * flow.shape[0] * flow.shape[1] * flow.shape[2]   # ration * num 
                            eval_samples += flow.shape[0] * flow.shape[1] * flow.shape[2]

                    eval_loss *= eval_samples
                    
                    total_mse_loss_tensor = torch.tensor(eval_loss, dtype=torch.float32, device=rank)
                    total_ratio_tensor = torch.tensor(eval_ratio, dtype=torch.float32, device=rank)
                    total_samples_tensor = torch.tensor(eval_samples, dtype=torch.float32, device=rank)
                    
                    dist.all_reduce(total_mse_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_ratio_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

                    global_mse_loss = total_mse_loss_tensor.item() / total_samples_tensor.item()
                    global_ratio = total_ratio_tensor.item() / total_samples_tensor.item()

                    if rank == 0:
                        logger.info(f"(step={train_steps:07d}) Eval ADE Loss: {global_mse_loss:.4f} Eval Ratio: {global_ratio:.4f}")
                        if cfg.wandb:
                            wandb.log(
                                {
                                    "eval_ade": global_mse_loss,
                                    "eval_ratio": global_ratio,
                                }
                            )

                dist.barrier()
            
            log_steps += 1
            train_steps += 1

            torch.cuda.empty_cache()

    model.eval()

    logger.info("Done!")
    dist.destroy_process_group()    # clean up

if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc_per_node=2 scripts/train_cvae.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/cvae.yaml", help="Path to the config file")
    args = parser.parse_args()

    cfg = load_config_as_namespace(args.config)
    if hasattr(cfg, "lr"):
        cfg.lr = float(cfg.lr)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    import torch
    # the first flag below was False when we tested this script but True makes A100 training a lot faster:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import random_split
    from utils.train_utils import create_logger, monitor_gradient_norm

    main(cfg)