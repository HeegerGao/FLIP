import torch.distributed as dist
import logging
from collections import OrderedDict
import torch
import os
import imageio

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def create_logger_single_gpu(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def monitor_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad and name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
         
def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))

def post_process_video(video_tensor, normalize=True, value_range=None):
    # rescale the pixel values to 0-255
    if normalize:
        norm_range(video_tensor, value_range)

    video_tensor = video_tensor.mul(255).add_(0.5).clamp_(0, 255)

    return video_tensor

def save_video(samples, output_path, normalize=True, value_range=None, video_name=None):
    """
    Args:
        samples (Tensor): 4D Tensor of shape (B, T, C, H, W) containing the video samples.
        output_file (str): Path to save the video.
    """
    assert samples.dim() == 5, f"Expected 5D Tensor, got {samples.dim()}D."
    os.makedirs(output_path, exist_ok=True)
    B, T, C, H, W = samples.shape
    
    if video_name is not None:
        assert isinstance(video_name, list), "video name should be a list of names!"
        assert len(video_name) == B, "video name should have the same length as the batch size!"

    samples = post_process_video(samples, normalize, value_range)

    for b in range(B):
        video_tensor = samples[b]
        video_np = video_tensor.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        if video_name is not None:
            video_name_b = video_name[b]
        else:
            video_name_b = b
        with imageio.get_writer(os.path.join(output_path, f"{video_name_b}.mp4"), fps=20) as writer:
            for i in range(T):
                writer.append_data(video_np[i])

    print(f"Video saved at {output_path}")

from libero.envs import *
def build_libero_env(
    resolution=128,
    env_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
):
    problem_name = 'libero_kitchen_tabletop_manipulation'
    env_kwags = {
        'robots': ['Panda'], 
        'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}, 
        'bddl_file_name': f'libero/bddl_files/libero_10/{env_name}.bddl', 
        'has_renderer': False, 
        'has_offscreen_renderer': True, 
        'ignore_done': True, 
        'use_camera_obs': True, 
        'camera_depths': False, 
        'camera_names': ['robot0_eye_in_hand', 'agentview'], 
        'reward_shaping': True, 
        'control_freq': 20, 
        'camera_heights': resolution, 
        'camera_widths': resolution, 
        'camera_segmentations': None
    }
    env = TASK_MAPPING[problem_name](**env_kwags)
    env.reset()
    
    return env