import sys
sys.path.append(".")
sys.path.append("..")
import os
import argparse
from glob import glob
import torch
import imageio.v3 as iio
import numpy as np
from flip.co_tracker.cotracker.utils.visualizer import Visualizer
import torch.nn.functional as F
import shutil
from tqdm import tqdm
from utils.yaml_utils import load_config_as_namespace

def resize_video(video, target_height, target_width):
    T = video.shape[1]
    C = video.shape[2]

    video_reshaped = video.view(-1, C, video.shape[3], video.shape[4])
    resized_video_reshaped = F.interpolate(video_reshaped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    resized_video = resized_video_reshaped.view(1, T, C, target_height, target_width)
    
    return resized_video

def sample_double_grid(grid_size, device="cuda", x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
    x = torch.linspace(x_range[0], x_range[1], grid_size)
    y = torch.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([xx.ravel(), yy.ravel()], dim=-1)

    return grid_points.to(device)

def track_video_iterative(cotracker, video, point_num, clip_length=16, device="cuda"):
    B, T, C, H, W = video.shape
    init_grid_size = np.sqrt(point_num).astype(int)
    init_grid_points = sample_double_grid(init_grid_size, device=device).unsqueeze(0)
    init_grid_points[:, :, 0] *= W
    init_grid_points[:, :, 1] *= H
    init_grid_points = init_grid_points.int()
    query_frame = torch.zeros(1, init_grid_points.shape[1], 1).to(device)
    init_grid_points = torch.cat([query_frame, init_grid_points], dim=-1)   # [1, N, 3]
    
    # clip the video to 16-frame clips
    pred_tracks = torch.zeros(1, T-clip_length+1, clip_length, point_num, 2).to(device) 
    pred_visibility = torch.zeros(1, T-clip_length+1, clip_length, point_num).to(device)
    
    for start_frame_idx in tqdm(range(0, T - clip_length + 1), desc="Processing Clips"):
        video_clip = video[:, start_frame_idx:start_frame_idx+clip_length, :, :, :]
        clip_tracks, clip_vis = cotracker(video_clip, queries=init_grid_points, backward_tracking=True) # [1, T, N, 2]
        pred_tracks[0, start_frame_idx, :, :, :] = clip_tracks
        pred_visibility[0, start_frame_idx, :, :] = clip_vis
    
    return pred_tracks, pred_visibility
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/libero_10/tracking.yaml", help="Config File")
    cfg = parser.parse_args()
    
    cfg = load_config_as_namespace(cfg.config_file)

    # load the cotracker model
    cotracker = torch.hub.load(cfg.tracker_dir, "cotracker2", source="local").eval().to(cfg.device)

    # get all video files under the source directory
    video_files = glob(os.path.join(cfg.source_dir, "**/*.mp4"), recursive=True)
    video_files.sort()
    
    if cfg.save_video:
        os.makedirs(cfg.save_video_dir, exist_ok=True)
        
    os.makedirs(cfg.save_tensor_dir, exist_ok=True)

    for video in video_files:
        if cfg.eye_in_hand is not None and cfg.eye_in_hand:
            # for libero_10 only
            if "eye_in_hand" not in video:
                continue
        video_source_dir = os.path.dirname(video)[5:]   # remove data/
        if cfg.save_video:
            tmp_save_video_dir = os.path.join(cfg.save_video_dir, video_source_dir)
            os.makedirs(tmp_save_video_dir, exist_ok=True)
        tmp_save_tensor_dir = os.path.join(cfg.save_tensor_dir, video_source_dir)
        os.makedirs(tmp_save_tensor_dir, exist_ok=True)
        video_name = os.path.basename(video).split(".")[0]

        print("processing video: ", video)
        frames = iio.imread(video, plugin="FFMPEG")  # plugin="pyav"
        video_cuda = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(cfg.device)  # B T C H W
        if video_cuda.shape[3] != cfg.new_imge_size[0] or video_cuda.shape[4] != cfg.new_imge_size[1]:
            video_cuda = resize_video(video_cuda, 128, video_cuda.shape[4] * 128 // video_cuda.shape[3])
            # # get the central 128 crop
            # video_cuda = video_cuda[:, :, :, :, (video_cuda.shape[4] - 128) // 2:(video_cuda.shape[4] + 128) // 2]
            # assert video_cuda.shape[-2] == 128 and video_cuda.shape[-1] == 128

        try:
            with torch.no_grad():
                pred_tracks, pred_visibility = track_video_iterative(cotracker, video_cuda, point_num=cfg.point_num, clip_length=cfg.clip_length, device=cfg.device)

            if cfg.save_video:  # the saved video could be very large
                vis = Visualizer(save_dir=tmp_save_video_dir, pad_value=0, fps=20, linewidth=1)
                # iteratively vis the tracking results with clips
                padded_video = torch.zeros(1, pred_tracks.shape[1]*cfg.clip_length, 3, cfg.new_imge_size[0], cfg.new_imge_size[1]).to(cfg.device)
                for i in range(pred_tracks.shape[1]):
                    padded_video[:, i*cfg.clip_length:(i+1)*cfg.clip_length, :, :, :] = video_cuda[:, i:i+cfg.clip_length, :, :, :]
                vis.visualize(padded_video, pred_tracks.reshape(1, -1, cfg.point_num, 2), pred_visibility.reshape(1, -1, cfg.point_num), filename=video_name)

            torch.save(pred_tracks.detach().cpu(), os.path.join(tmp_save_tensor_dir, video_name + "_tracks.pt"))
            torch.save(pred_visibility.detach().cpu(), os.path.join(tmp_save_tensor_dir, video_name + "_visibility.pt"))
        except torch.cuda.OutOfMemoryError as e:
            print("Out of memory error: ", e)
            continue
        
        # copy the original video and txt file to the tmp_save_tensor_dir
        shutil.copy(video, os.path.join(tmp_save_tensor_dir, video_name + ".mp4"))
        video_dir = os.path.dirname(video)
        if os.path.exists(os.path.join(video_dir, video_name+".txt")):
            shutil.copy(os.path.join(video_dir, video_name+".txt"), os.path.join(tmp_save_tensor_dir, video_name + ".txt"))
        elif os.path.exists(os.path.join(video_dir, "language_instruction.txt")):   # for libero
            shutil.copy(os.path.join(video_dir, "language_instruction.txt"), os.path.join(tmp_save_tensor_dir, video_name + ".txt"))

if __name__ == "__main__":
    main()
