import sys
sys.path.append(".")
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.module_utils import load_llama, load_flow_model, load_dynamics_model_and_vae, load_reward_model, predict_flow, predict_video, predict_reward, get_sentence_embedding, draw_reward
from flip.diffusion import create_diffusion
import cv2
from torchvision import transforms
from utils.vis_utils import vis_pred_flow, connect_and_save_videos
from utils.module_utils import generate_query_points
from utils.model_utils import load_models, ConfigArgs
from colorama import Fore, Back, Style, init
from utils.train_utils import post_process_video
from glob import glob
  
def hill_climbing(
    flow_model, 
    dynamics_model, 
    reward_model, 
    image, 
    language, 
    sentence_embedding,
    diffusion,
    vae,
    args,
):
    img_history = image.unsqueeze(0).repeat(args.obs_history, 1, 1, 1)

    # since the video is the "future" video, so the shape is flow_horizon-1
    video_plans = torch.zeros([args.planning_horizon, args.planning_beam, args.flow_horizon-1, 3, args.img_size[0], args.img_size[1]]).to(args.device)
    flow_plans = torch.zeros([args.planning_horizon, args.planning_beam, args.grid_pt_num, args.flow_horizon-1, 2]).to(args.device)
    reward_plans = torch.zeros([args.planning_horizon, args.planning_beam, 1]).to(args.device)

    query_points = generate_query_points(args.img_size, args.device, grid_pt_num=args.grid_pt_num)
    with torch.no_grad():
        for h in range(args.planning_horizon):
            print(Fore.GREEN + f"Planning step {h} ...")
            if h > 0:
                img_history = combined_imgs[:, -args.obs_history:].clone()    # [beam, obs_history, 3, img_size, img_size]
            elif h == 0:
                img_history = img_history.unsqueeze(0).repeat(args.planning_beam, 1, 1, 1, 1)  # [beam, obs_history, 3, img_size, img_size]

            # start parallel beam seach. for every beam, we have <candidate> num of actions
            # expand to the action candidate num and reduce the dimensions of beam and candidate
            replicated_img_history = img_history.unsqueeze(1).repeat(1, args.action_candidates, 1, 1, 1, 1).reshape(-1, args.obs_history, 3, args.img_size[0], args.img_size[1]) # [beam*action_candidates, obs_history, 3, img_size, img_size]
            flow = predict_flow(flow_model, replicated_img_history.clone(), sentence_embedding.repeat(replicated_img_history.shape[0], 1, 1), args, given_query_points=query_points.repeat(replicated_img_history.shape[0], 1, 1))
            future_imgs = predict_video(dynamics_model, replicated_img_history.clone()[:, -args.dynamics_obs_history:], flow.clone(), sentence_embedding.repeat(replicated_img_history.shape[0], 1, 1), args, diffusion, vae)   # [beam*action_candidates, args.flow_horizon-1, 3, img_size, img_size]
            rewards = predict_reward(reward_model, future_imgs[:, -1, :, :, :].clone(), language)

            flow = flow.reshape(args.planning_beam, args.action_candidates, args.grid_pt_num, args.flow_horizon, 2)
            future_imgs = future_imgs.reshape(args.planning_beam, args.action_candidates, args.flow_horizon-1, 3, args.img_size[0], args.img_size[1])
            rewards = rewards.reshape(args.planning_beam, args.action_candidates, 1)

            # select the best action_candidates actions
            top_index = torch.argsort(rewards, descending=True, dim=1)[:, :1].to(args.device)
            flow = torch.gather(flow, 1, top_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, args.grid_pt_num, args.flow_horizon, 2)).squeeze(1)
            # future_imgs = torch.gather(future_imgs, 1, top_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, args.img_size, args.img_size)).squeeze(1)
            future_imgs = torch.gather(future_imgs, 1, top_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, args.flow_horizon-1, 3, args.img_size[0], args.img_size[1])).squeeze(1)
            rewards = torch.gather(rewards, 1, top_index).squeeze(1)

            flow_plans[h] = flow[:, :, 1:, :]   # remove the first flow frame since it is the initial flow and is replicated with the last frame of the previous one
            video_plans[h] = future_imgs
            reward_plans[h] = rewards
            
            # concate the future_imgs with the img_history for the img_history of next planning step
            combined_imgs = torch.cat([img_history, future_imgs], dim=1)

            # periodically replace the lowest reward plan with the best plan
            if (h+1) % args.replace_time == 0 and h > 0:
                total_rewards = torch.sum(reward_plans[:h+1], 0)    # [beam, 1], i.e., sum along the planning horizon
                worst_index = torch.argsort(total_rewards, descending=False, dim=0)[0]
                best_index = torch.argsort(total_rewards, descending=True, dim=0)[0]
                video_plans[:, worst_index] = video_plans[:, best_index]
                flow_plans[:, worst_index] = flow_plans[:, best_index]
                reward_plans[:, worst_index] = reward_plans[:, best_index]

    # select the best plan according to the whole trajectory reward
    total_rewards = torch.sum(reward_plans, 0)    # [beam, 1], i.e., sum along the planning horizon
    best_index = torch.argsort(total_rewards, descending=True, dim=0)[0]

    return video_plans, flow_plans, reward_plans, best_index

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_path", type=str, default="test_video")

    parser.add_argument("--flow_model_path", type=str, default="models/libero_10/cvae.pt")
    parser.add_argument("--flow_model_type", type=str, default="CVAE-B")
    parser.add_argument("--flow-horizon", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=[128, 128])

    parser.add_argument("--dynamics_model_path", type=str, default="models/libero_10/agentview_dynamics.pt")
    parser.add_argument("--dynamics_model_type", type=str, default="DynamicsModel-B")
    parser.add_argument("--dynamics-obs-history", type=int, default=4)
    parser.add_argument("--vae_path", type=str, default="models/libero_10/finetuned_vae.pt")
    parser.add_argument("--vae_encode", type=bool, default=True)
    parser.add_argument("--learn_sigma", type=bool, default=True)

    parser.add_argument("--reward_model_folder_path", type=str, default="models/libero_10/")
    parser.add_argument("--reward_model_path", type=str, default="models/libero_10/reward.pt")
    
    parser.add_argument("--planning_beam", type=int, default=2)
    parser.add_argument("--planning_horizon", type=int, default=30)
    parser.add_argument("--action_candidates", type=int, default=2)
    parser.add_argument("--replace_time", type=int, default=4)
    parser.add_argument("--grid_pt_num", type=int, default=529) # to be a square number

    parser.add_argument("--device", type=str, default="cuda:0") # should always from 0, because we have set the CUDA_VISIBLE_DEVICES
    parser.add_argument("--output_dir", type=str, default="output") # should always from 0, because we have set the CUDA_VISIBLE_DEVICES
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-sampling-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    init(autoreset=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # load the models
    llama_model = load_llama()

    flow_model = load_flow_model(args.flow_model_path, args, args.device).eval()
    dynamics_model, vae = load_dynamics_model_and_vae(args.dynamics_model_path, args.vae_path, args, args.device)
    reward_model = load_reward_model(args.reward_model_folder_path, args.reward_model_path, args.device).eval()

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.num_sampling_steps, learn_sigma=args.learn_sigma)

    images = []
    languages = []
    sentence_embeddings = []
    for test_idx in range(10):
        video_file = f"test_video/libero_10/agentview_demo_{test_idx}.mp4"
        with open(f"test_video/libero_10/agentview_demo_{test_idx}.txt", "r") as f:
            sentence = f.read()
        # args.language = sentence
        languages.append(sentence)
        video_frames = []
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        cap.release()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        video = torch.stack([transform(frame) for frame in video_frames]).to(args.device)

        image = video[0]
        
        # # apple
        # from PIL import Image
        # overlay_img = Image.open("test_video/libero_10/apple.jpg").convert("RGB").resize((32, 32))
        # overlay_img = transform(overlay_img).to(args.device)
        # image[:, 68:100, 8:40] = overlay_img
        
        images.append(image)

        with torch.no_grad():
            sentence_embedding = get_sentence_embedding(llama_model, sentence)
            sentence_embeddings.append(sentence_embedding)
            
    del llama_model # to save memory

    for idx in range(len(images)):
        image = images[idx]
        sentence_embedding = sentence_embeddings[idx]
        args.language = languages[idx]
        
        # run the hill climbing
        video_plan, flow_plan, reward_plan, best_index = hill_climbing(
            flow_model, 
            dynamics_model, 
            reward_model, 
            image, 
            args.language,
            sentence_embedding,
            diffusion, 
            vae, 
            args,
        )

        # scale video to [0, 255]
        video_plan = post_process_video(video_plan)

        # complete videos
        video_res = video_plan.permute(1, 0, 2, 3, 4, 5).reshape(args.planning_beam, args.planning_horizon*(args.flow_horizon-1), 3, args.img_size[0], args.img_size[1])   # [beam, plan_horizon*(flow_horizon-1), 3, img_size, img_size]

        # draw
        for i in range(video_plan.shape[1]):    # beam
            drew_flow = []
            for h in range(video_plan.shape[0]):    # planning horizon
                drew_flow.append(vis_pred_flow(flow=flow_plan[h, i].unsqueeze(0), init_img=video_plan[h][i][0]))
            connect_and_save_videos(drew_flow, video_plan[:, i].reshape(-1, 3, args.img_size[0], args.img_size[1]), f"{args.output_dir}/{idx}_plan_res_{i}.mp4", drop_last=False)

            # draw per step reward
            with torch.no_grad():
                i_th_video = video_res[i].reshape(-1, 3, args.img_size[0], args.img_size[1])
                i_th_reward = predict_reward(reward_model, i_th_video, args.language, need_post_process=False)
                
                draw_reward(i_th_reward.detach().cpu().numpy().reshape(-1), f"{args.output_dir}/{idx}_reward_res_{i}.gif", args.flow_horizon-1, no_stop=True)
        
        
    # torchrun --master_port=16666 scripts/hill_climbing.py