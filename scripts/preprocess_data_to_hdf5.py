import sys
sys.path.append(".")
sys.path.append("..")
import torch
import os
import pickle
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import cv2
import h5py
import numpy as np
import argparse
from utils.yaml_utils import load_config_as_namespace
from utils.data_utils import padding_flow

''' preprocess the data and save them to save the data loading time.
    No visual data augmentation! So if you need to augment the data, you can do it in the training loop.
'''

def preprocess_and_save_to_hdf5(data_dir, output_dir, obs_history=4, flow_horizon=16, img_size=[128, 128], eye_in_hand=False, process_sentence_embedding=True):
    """
    Preprocess the data and save it into a preprocessed dataset format.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size[0], img_size[1])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {
        "video": [],
        "points": [],
        "scale": [],
        "direction": [],
        "sentence_embedding_name": [],
    }

    idx_indicator = []  # 1 for valid and 0 for padded

    if process_sentence_embedding:
        sentence_embeddings = {}

        sentence_embedding_files = glob(os.path.join(data_dir, "**/*_sentence_embedding.pt"), recursive=True)
        sentence_embedding_files.sort()

        for i in tqdm(range(len(sentence_embedding_files)), desc="Loading Sentence Embeddings"):
            sentence_embedding_file = sentence_embedding_files[i]
            sentence_embedding = torch.load(sentence_embedding_file).cpu()
            file_prefix = sentence_embedding_file.split("/")[-1].split(".")[0] # KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_sentence_embedding
            sentence_embeddings[file_prefix] = sentence_embedding

    video_files = glob(os.path.join(data_dir, "**/*.mp4"), recursive=True)
    if eye_in_hand:
        video_files = [f for f in video_files if "eye_in_hand" in f]
    else:
        video_files = [f for f in video_files if "agentview" in f]
    video_files.sort()

    for video_file in tqdm(video_files, desc="Preprocessing Data"):
        video_frames = []
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        cap.release()
        
        video_tensor = torch.stack([transform(frame) for frame in video_frames])  # [T, C, H, W]
        video_len = video_tensor.shape[0]

        file_prefix = video_file.split(".")[0]
        flow_file = file_prefix + "_tracks.pt"
        flow_tensor = torch.load(flow_file, map_location="cpu")[0]  # [T-flow_horizon+1, flow_horizon, N, 2]

        sentence_embedding_name = video_file.split(".")[0].split("/")[-1] + "_sentence_embedding"

        # Padding video and flow
        video_tensor_padded = torch.cat([
            video_tensor[0:1].repeat(obs_history - 1, 1, 1, 1), 
            video_tensor
        ], dim=0)

        # for future_img
        video_tensor_padded = torch.cat([
            video_tensor_padded,
            video_tensor[-1:].repeat(flow_horizon - 1, 1, 1, 1)
        ], dim=0)

        # also pad for flow
        flow_tensor_padded = padding_flow(flow_tensor, obs_history, flow_horizon, pad_history=True, pad_future=True)  # [T+obs_history+flow_horizon-2-flow_horizon+1, flow_horizon, N, 2]
        flow_tensor_padded = padding_flow(flow_tensor_padded, obs_history, flow_horizon, pad_history=False, pad_future=True)  # [T+obs_history+flow_horizon-2, flow_horizon, N, 2]

        flow_tensor_padded = flow_tensor_padded.transpose(1, 2)  # [T+obs_history+flow_horizon-2, N, flow_horizon, 2]

        # scale and direction
        # in this formulation, the scale and direction of the true video first frame should be all zero, since it is the scale and direction to the previous padded frame
        diffs = flow_tensor_padded[:, :, 1:] - flow_tensor_padded[:, :, :-1]
        distances_per_step = torch.sqrt(torch.sum(diffs ** 2, dim=-1))
        scale = distances_per_step.unsqueeze(-1)   # T+obs_history+flow_horizon-2, N flow_horizon-1 1
        direction = diffs / (scale + 1e-8)  # T+obs_history+flow_horizon-2, N flow_horizon-1 2

        # pad the scale and direction at the very last frame. This is OK since this frame will not be selected
        scale = torch.cat([scale, torch.zeros(scale.shape[0], scale.shape[1], 1, 1).to(scale.device)], dim=2)   # T+obs_history+flow_horizon-2, N flow_horizon 1
        direction = torch.cat([direction, torch.zeros(direction.shape[0], direction.shape[1], 1, 2).to(direction.device)], dim=2)   # T+obs_history+flow_horizon-2, N flow_horizon 2

        idx_indicator += [0] * (obs_history - 1)    # first obs_history-1 frames are padded
        idx_indicator += [1] * (video_len)  # valid frames
        idx_indicator += [0] * (flow_horizon - 1)   # last flow_horizon-1 frames are padded

        results["video"].append(video_tensor_padded)
        results["points"].append(flow_tensor_padded)
        results["scale"].append(scale)
        results["direction"].append(direction)
        results["sentence_embedding_name"] += [sentence_embedding_name] * (video_len+obs_history+flow_horizon-2)

    results["video"] = torch.cat(results["video"], dim=0)  # [total_padded_frames, C, H, W]
    results["points"] = torch.cat(results["points"], dim=0)  # [total_padded_frames, N, flow_horizon, 2]
    results["scale"] = torch.cat(results["scale"], dim=0)  # [total_padded_frames, N, flow_horizon, 1]
    results["direction"] = torch.cat(results["direction"], dim=0)  # [total_padded_frames, N, flow_horizon, 2]

    assert results["video"].shape[0] == results["points"].shape[0] == results["scale"].shape[0] == results["direction"].shape[0] == len(results["sentence_embedding_name"]) == len(idx_indicator), print(results["video"].shape, results["points"].shape, results["scale"].shape, results["direction"].shape, len(results["sentence_embedding_name"]), len(idx_indicator))

    file_name = "preprocessed_data_eye_in_hand.hdf5" if eye_in_hand else "preprocessed_data.hdf5"
    with h5py.File(os.path.join(output_dir, file_name), "w") as f:
        f.create_dataset("video", data=results["video"])
        f.create_dataset("points", data=results["points"])
        f.create_dataset("scale", data=results["scale"])
        f.create_dataset("direction", data=results["direction"])
        f.create_dataset("sentence_embedding_name", data=[n.encode("utf-8") for n in results["sentence_embedding_name"]])
        f.create_dataset("idx_indicator", data=np.array(idx_indicator))

    if process_sentence_embedding:
        with open(os.path.join(output_dir, "sentence_embeddings.pkl"), "wb") as f:
            pickle.dump(sentence_embeddings, f)
        
if __name__ == "__main__":
    # this will preprocess ALL data under the given directory iteratively
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/libero_10/preprocess.yaml", help="Config File")
    args = parser.parse_args()
    
    cfg = load_config_as_namespace(args.config_file)

    preprocess_and_save_to_hdf5(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir, 
        obs_history=cfg.obs_history, 
        flow_horizon=cfg.flow_horizon, 
        img_size=cfg.img_size,
        eye_in_hand=cfg.eye_in_hand,
        process_sentence_embedding=cfg.process_sentence_embedding
    )
