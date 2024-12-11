import sys
sys.path.append(".")
sys.path.append("..")
import torch
import torch.nn as nn
from utils.model_utils import load_models
from flip.network.cvae import CVAE_models
from diffusers.models import AutoencoderKL
from flip.network.dynamics import DynamicsModel_models
from liv import load_liv
from liv.models.model_liv import LIV
import clip
from einops import rearrange
import omegaconf
import os
import copy
from llama_models.models.llama3.reference_impl.generation import Llama
from utils.train_utils import post_process_video
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_flow_model(ckpt_path, args, device):
    model = CVAE_models[args.flow_model_type](
        flow_horizon=args.flow_horizon,
        obs_history=args.obs_history,
        img_size=args.img_size,
    ).to(device)
    state_dict = load_models(ckpt_path)
    model.load_state_dict(state_dict["model"])
    model.eval()
    print("Loaded flow model from", ckpt_path)

    return model

def load_dynamics_model_and_vae(dit_ckpt_path, vae_ckpt_path, args, device):
    model = DynamicsModel_models[args.dynamics_model_type](
        flow_horizon=args.flow_horizon,
        obs_history=args.dynamics_obs_history,
        input_size=args.img_size if not args.vae_encode else [args.img_size[0] // 8, args.img_size[1] // 8],
        in_channels=3 if not args.vae_encode else 4,
    ).to(device)
    state_dict = torch.load(dit_ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded dynamics model from", dit_ckpt_path)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    state_dict = torch.load(vae_ckpt_path, map_location=lambda storage, loc: storage)
    vae.load_state_dict(state_dict["model"])
    vae.eval()
    print("Loaded VAE from", vae_ckpt_path)

    return model, vae

def load_reward_model(ckpt_folder_path, model_path, device, load_from_official=False):
    if load_from_official:
        model = load_liv(home=ckpt_folder_path, specified_device=device, single_gpu=True)
        model.eval()
        print("Loaded reward model from", ckpt_folder_path)

        return model
    
    else:   # load from finetuned model
        configpath = os.path.join(ckpt_folder_path, ".hydra", "config.yaml")
        modelcfg = omegaconf.OmegaConf.load(configpath)
        config = copy.deepcopy(modelcfg)
        config["device"] = device
        # model = hydra.utils.instantiate(config).to(device)
        model = LIV().to(device)

        payload = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(payload['liv'])
        clip.model.convert_weights(model)
        model.to(device)
        model.device = device

        print("Loaded reward model from", ckpt_folder_path)

        return model

def load_llama(ckpt_dir = "Meta-Llama-3.1-8B", tokenizer_path = "Meta-Llama-3.1-8B/tokenizer.model", max_seq_len = 512, max_batch_size = 1, model_parallel_size = None):
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    llama.model = llama.model.float()
    llama.model = llama.model.to("cuda:0")

    return llama

def get_sentence_embedding(llama, sentence):
    def hook_fn(module, input, output):
        global h_embedding
        h_embedding = output

    handle = llama.model.layers[-1].register_forward_hook(hook_fn)
    encoded_input = llama.tokenizer.encode(sentence, bos=True, eos=False)

    with torch.inference_mode():
        tokens = torch.tensor([encoded_input], dtype=torch.long, device="cuda:0")
        logits = llama.model.forward(tokens, 0)

    sentence_embedding = h_embedding.clone()    # use the last hidden layer's output as the sentence embedding

    return sentence_embedding

def predict_flow(flow_model, obs_history, sentence_embedding, args, grid_pt_num=400, given_query_points=None, stationary_point_filter=False):
    assert grid_pt_num > 0 or given_query_points is not None, "At least one of grid_pt_num and given_query_points should be provided."
    if given_query_points is not None:
        query_points = given_query_points
    else:
        x = torch.linspace(0, args.img_size[0], int(grid_pt_num ** 0.5))
        y = torch.linspace(0, args.img_size[1], int(grid_pt_num ** 0.5))    # TODO: this is OK because we will use color seg for langauge-table
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        query_points = torch.stack([xx.ravel(), yy.ravel()], dim=-1)
        query_points = query_points.unsqueeze(0).to(obs_history.device)  # [1, N, 2]        

    z_mu, z_logvar = torch.zeros([obs_history.shape[0], flow_model.hidden_size]).to(obs_history.device), torch.zeros([obs_history.shape[0], flow_model.hidden_size]).to(obs_history.device)
    z = torch.distributions.normal.Normal(z_mu, torch.exp(torch.exp(0.5 * z_logvar))).rsample().to(obs_history.device)
    scale, direction = flow_model.inference(query_points, obs_history, sentence_embedding, z)

    predicted_flow = flow_model.reconstruct_flow(query_points, scale, direction)

    return predicted_flow

def predict_next_image(dynamics_model, llama, observation, flow, language, args, diffusion, vae=None):
    if args.vae_encode:
        assert vae is not None, "vae should be provided when args.vae_encode is True."
        with torch.no_grad():
            obs = vae.encode(observation).latent_dist.sample().mul_(0.18215)
    else:
        obs = observation

    sentence_embedding = get_sentence_embedding(llama, language)

    model_kwargs = dict(
        flow=flow,
        sentence=sentence_embedding,
        obs=obs,
    )

    if args.vae_encode:
        z = torch.randn(obs.shape[0], 4, args.img_size//8, args.img_size//8, device=args.device)
    else:
        z = torch.randn(obs.shape[0], 3, args.img_size, args.img_size, device=args.device)

    samples = diffusion.p_sample_loop(
        dynamics_model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device
    )

    if args.vae_encode:
        samples = vae.decode(samples / 0.18215).sample

    return samples

def predict_video(dynamics_model, observation, flow, sentence_embedding, args, diffusion, vae=None):
    if args.vae_encode:
        assert vae is not None, "vae should be provided when args.vae_encode is True."
        with torch.no_grad():
            B = observation.shape[0]
            obs_condition = rearrange(observation, 'b t c h w -> (b t) c h w')
            obs_condition = vae.encode(obs_condition).latent_dist.sample().mul_(0.18215)
            obs_condition = rearrange(obs_condition, '(b t) c h w -> b t c h w', b=B)
    else:
        obs_condition = observation

    model_kwargs = dict(
        flow=flow,
        sentence=sentence_embedding,
    )

    B, _, C, H, W = obs_condition.shape

    z = torch.randn(B, args.dynamics_obs_history+args.flow_horizon-1, C, H, W, device=args.device)
    z[:, :args.dynamics_obs_history] = obs_condition

    # Sample images:
    samples = diffusion.p_sample_loop(
        dynamics_model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device, flow_flag=True, history_length=args.dynamics_obs_history
    )

    if args.vae_encode:
        samples = rearrange(samples, 'b t c h w -> (b t) c h w')
        samples = vae.decode(samples / 0.18215).sample
        samples = rearrange(samples, '(b t) c h w -> b t c h w', b=B)

    samples = samples[:, args.dynamics_obs_history:]

    return samples

def predict_reward(reward_model, img, language, need_post_process=True):
    ''' reward is the similarity between the current embedding and the goal embedding
    '''
    if need_post_process:
        img = post_process_video(img, normalize=True)
    embeddings = reward_model(input=img, modality="vision")
    cur_embedding = embeddings
    token = clip.tokenize([language])
    goal_embedding_text = reward_model(input=token.to(reward_model.device), modality="text")
    goal_embedding_text = goal_embedding_text[0]

    cur_reward_text = reward_model.sim(goal_embedding_text.unsqueeze(0).repeat(cur_embedding.shape[0], 1), cur_embedding)

    return cur_reward_text

def draw_reward(rewards, output_file, flow_horizon=16, no_stop=False):
    N = flow_horizon    # stop frames
    M = flow_horizon    # play frames
    if no_stop:
        N = 0
    
    rewards = np.array(rewards)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,2))

    ax.plot(np.arange(len(rewards)), rewards, color="tab:blue", linewidth=3)
    ax.set_xlabel("Frame", fontsize=15)
    ax.set_ylabel("Embedding Vale", fontsize=15)
    ax.set_title("Reward", fontsize=15)

    plt.close()
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    
    def animate(i):
        ax.clear()
        ranges = np.arange(len(rewards))
        

        if i < N:
            end_frame = 0
        else:
            adjusted_frame = i - N
            current_frame = adjusted_frame // (M + N)
            
            if adjusted_frame % (M + N) < M:
                end_frame = current_frame * M + adjusted_frame % (M + N)
                if end_frame > len(rewards):
                    end_frame = len(rewards)
            else:
                end_frame = (current_frame + 1) * M
                if end_frame > len(rewards):
                    end_frame = len(rewards)

        line = ax.plot(ranges[0:end_frame], rewards[0:end_frame], color="tab:blue", linewidth=3)
        ax.set_xlabel("Frame", fontsize=15)
        ax.set_ylabel("Embedding Vale", fontsize=15)
        ax.set_title("Reward", fontsize=15)
        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)

        return line

    total_frames = N + (len(rewards) // M) * (M + N) + (len(rewards) % M)
    
    ani = FuncAnimation(fig, animate, interval=20, repeat=False, frames=total_frames+5)
    ani.save(output_file, dpi=100, writer=PillowWriter(fps=15))

def policy_inference(model, naction, noise_scheduler, agentview_embedding, eye_in_hand_embedding, sentence_embedding, state_history, agentview_flow_embedding=None, eye_in_hand_flow_embedding=None):
    for k in noise_scheduler.timesteps:
        # predict noise
        if agentview_flow_embedding is not None and eye_in_hand_flow_embedding is not None:
            noise_pred = model(naction, k, agentview_embedding, eye_in_hand_embedding, sentence_embedding, state_history, agentview_flow_embedding, eye_in_hand_flow_embedding)
        else:
            noise_pred = model(naction, k, agentview_embedding, eye_in_hand_embedding, sentence_embedding, state_history)

        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample
    
    return naction

def generate_query_points(image_size, device, grid_pt_num=196):
    x = torch.linspace(0, image_size[0], int(grid_pt_num ** 0.5))
    y = torch.linspace(0, image_size[1], int(grid_pt_num ** 0.5))
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    query_points = torch.stack([xx.ravel(), yy.ravel()], dim=-1)
    query_points = query_points.unsqueeze(0).to(device)  # [1, N, 2]

    return query_points
