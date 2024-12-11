import sys
sys.path.append("./")
sys.path.append("../")
from diffusers.models import AutoencoderKL
from flip.dataloader.flip_dataloader import FLIPDataset
import argparse
from einops import rearrange
from utils.yaml_utils import load_config_as_namespace

def main(cfg):
    # Initialize distributed process group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Load the VAE model, freeze the encoder, and setup the optimizer for decoder finetuning
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)

    for param in vae.encoder.parameters():
        param.requires_grad = False
    vae = DDP(vae, device_ids=[rank])
    optimizer = optim.Adam(vae.module.decoder.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    # Setup dataset and dataloader with distributed sampling
    dataset = FLIPDataset(
        data_dir=cfg.data_dir,
        obs_history=cfg.obs_history,
        flow_horizon=cfg.flow_horizon,
        calculate_scale_and_direction=False,
        return_future_img=True,
        data_aug=None,
    )
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset, batch_size=int(cfg.global_batch_size // dist.get_world_size()), sampler=train_sampler, num_workers=cfg.num_workers)

    # Start training
    num_epochs = cfg.epochs
    train_steps = 0
    for epoch in range(num_epochs):
        vae.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch_data in train_dataloader:
            observation = batch_data['observation'].to(device)
            future_imgs = batch_data['future_img'].to(device)  # [B, T, C, H, W]
            video = torch.cat([observation, future_imgs], dim=1)  # [B, T, C, H, W]

            x = rearrange(video, 'b t c h w -> (b t) c h w')
            # Map input images to latent space + normalize latents:
            x = vae.module.encode(x).latent_dist.sample().mul_(0.18215)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=video.shape[0])

            recon_images = rearrange(x, 'b t c h w -> (b t) c h w')
            recon_images = vae.module.decode(recon_images / 0.18215).sample
            recon_images = rearrange(recon_images, '(b t) c h w -> b t c h w', b=video.shape[0])
                            
            # Calculate reconstruction loss
            loss = criterion(recon_images, video)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if train_steps % cfg.eval_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{train_steps+1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

                if rank == 0:
                    checkpoint = {
                        "model": vae.module.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{cfg.save_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                                    
                    save_video(recon_images, os.path.join(cfg.save_dir, f"recon_epoch_{cfg.epochs}"), normalize=True)
                    save_video(video, os.path.join(cfg.save_dir, f"gt_epoch_{cfg.epochs}"), normalize=True)

            train_steps += 1

    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc_per_node=8 scripts/finetune_vae.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/finetune_vae.yaml", help="Path to the config file")
    args = parser.parse_args()

    cfg = load_config_as_namespace(args.config)
    if hasattr(cfg, "lr"):
        cfg.lr = float(cfg.lr)    
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    
    import torch
    import torch.optim as optim
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.train_utils import save_video    
    
    main(cfg)
