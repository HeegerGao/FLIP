import sys
sys.path.append('.')
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from PIL import Image  
import clip 
from glob import glob
import hydra
import numpy as np
import pandas as pd 
import torch
import torchvision.transforms as T
import time

from liv import load_liv
from liv.utils import utils
from liv.utils.data_loaders import LIVBuffer, LIVLongHorizonBuffer
from liv.utils.logger import Logger
from liv.utils.plotter import plot_reward_curves

import cv2
import yaml
import wandb
from utils.yaml_utils import load_config_as_namespace
import argparse

mean_video_len_global = None

def make_network(cfg):
    model =  hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    if cfg.device == "cpu":
        model = model.module.to(cfg.device)
    return model

def make_dataset_yaml(yaml_name, dataset_path, yaml_path="liv/cfgs/dataset"):
    data = {
        'dataset': f'{yaml_name}',
        'datapath_train': f'{os.path.abspath(dataset_path)}',
        'wandbproject': f'liv_finetune_{yaml_name}',
        'hydra': {
            'job': {
                'name': f'train_liv_{yaml_name}',
            }
        }
    }

    with open(os.path.join(yaml_path, f"{yaml_name}.yaml"), 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
    print("YAML file created")

def make_dataset_from_video_dir(video_data_dir, output_path):
    if isinstance(video_data_dir, str):
        video_files = glob(os.path.join(video_data_dir, "**/*.mp4"), recursive=True)
    elif isinstance(video_data_dir, list):
        video_files = []
        for video_dir in video_data_dir:
            video_files += glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)

    manifest = {"index": [], "directory": [], "num_frames": [], "text": []}
    mean_video_len = 0
    for i in range(len(video_files)):
        video_file = video_files[i]
        output_dir = os.path.join(output_path, f"video{i}")
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        frame_count2 = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"{frame_count2}.png"
            frame_path = os.path.join(output_dir, frame_filename)

            # if frame_count >= 180 and frame_count2 <= 53:
            if True:
            # if frame_count <= cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2:
                cv2.imwrite(frame_path, frame)
                frame_count2 += 1
            frame_count += 1

        cap.release()
        print(f"Processed video: {video_file}, saved {frame_count2} frames to {output_dir}")
        mean_video_len += frame_count2

        manifest["index"].append(i)
        manifest["directory"].append(os.path.abspath(output_dir))
        manifest["num_frames"].append(frame_count2)

        txt_file = video_file.replace(".mp4", ".txt")
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            language = [line.strip() for line in lines]
        manifest["text"].append(language)
    
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(f"{output_path}/manifest.csv")

    mean_video_len /= len(video_files)
    print(f"Mean video length: {mean_video_len}")

    return mean_video_len

class Workspace:
    def __init__(self, cfg, mean_video_len=0):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logging = self.cfg.logging
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if self.logging:
            self.setup()

        if not cfg.eval:
            print("Creating Dataloader")
            train_iterable = LIVLongHorizonBuffer(datasource=self.cfg.dataset.dataset, datapath=self.cfg.dataset.datapath_train, num_workers=self.cfg.num_workers, num_demos=self.cfg.num_demos, doaug=self.cfg.doaug, alpha=self.cfg.alpha)
            self.train_dataset = train_iterable
            self.train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                            batch_size=self.cfg.batch_size,
                                            num_workers=self.cfg.num_workers,
                                            pin_memory=True))

        ## Init Model
        print("Initializing Model")
        self.model = make_network(cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0

        ## If reloading existing model
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.train_steps, 1)
        eval_freq = self.cfg.eval_freq
        eval_every_step = utils.Every(eval_freq, 1)

        # trainer = Trainer()
        trainer = hydra.utils.instantiate(self.cfg.trainer)

        ## Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            if eval_every_step(self.global_step):
                self.generate_reward_curves()
                self.save_snapshot()
            
            ## Sample Batch
            t0 = time.time()
            batch = next(self.train_loader)
            t1 = time.time()
            metrics, st = trainer.update(self.model, batch, self.global_step)

            # check nan
            for key in metrics:
                if np.isnan(metrics[key]):
                    print(f"Warning: NaN in {key}!")
                    sys.exit(1)

            wandb.log(metrics)            
            t2 = time.time()
            if self.logging:
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f'Sample time {t1-t0}, Update time {t2-t1}')
                
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["liv"] = self.model.module.state_dict()
        sdict["optimizer"] = self.model.module.encoder_opt.state_dict()
        sdict["global_step"] = self._global_step
        torch.save(sdict, snapshot)
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        if snapshot_path != 'liv':
            payload = torch.load(snapshot_path)
            self.model.module.load_state_dict(payload['liv'])
        else:
            self.model = load_liv()
        clip.model.convert_weights(self.model)
        try:
            self._global_step = payload['global_step']
        except:
            print("Warning: No global step found")

    def generate_reward_curves(self):
        self.model.eval()
        os.makedirs(f"{self.work_dir}/reward_curves", exist_ok=True)
        transform = T.Compose([T.ToTensor()])

        if self.cfg.dataset.dataset not in ["epickitchen"]:
            manifest = pd.read_csv(os.path.join(self.cfg.dataset.datapath_train, "manifest.csv"))
            tasks = manifest["text"].unique()
        else:
            manifest = pd.read_csv(os.path.join(self.cfg.dataset.datapath_train, "EPIC_100_validation.csv"))
            tasks = ["open microwave", "open cabinet", "open door"]

        fig_filename = f"{self.work_dir}/reward_curves/{self._global_step}_{self.cfg.dataset.dataset}"
        if self.cfg.dataset.dataset in ["epickitchen"]:
            def load_video(m):
                imgs_tensor = []
                start_frame = m["start_frame"]
                end_frame = m["stop_frame"]
                vid = f"/data2/jasonyma/EPIC-KITCHENS/frames/{m['participant_id']}/rgb_frames/{m['video_id']}"
                for index in range(start_frame, end_frame):
                    img = Image.open(f"{vid}/frame_0000{index+1:06}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor

        else:
            def load_video(m):
                imgs_tensor = []
                vid = m["directory"]
                for index in range(m["num_frames"]):
                    try:
                        img = Image.open(f"{vid}/{index}.png")
                    except:
                        img = Image.open(f"{vid}/{index}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor

        plot_reward_curves(
            manifest,
            tasks,
            load_video,
            self.model,
            fig_filename,
            animated=self.cfg.animate,
        )
        self.model.train()


@hydra.main(config_path='../liv/cfgs', config_name='config_liv')
def main(cfg):
    global mean_video_len_global
    from omegaconf import OmegaConf
    
    current_file_path = os.path.abspath(__file__)
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    cfg.dataset.datapath_train = os.path.join(repo_dir, cfg.dataset.datapath_train)
    print(f"datapath_train: {cfg.dataset.datapath_train}")
    
    wandb.init(
        project="Finetune LIV",
        config=OmegaConf.to_container(cfg, resolve=True),
    )    
    root_dir = Path.cwd()
    workspace = Workspace(cfg, mean_video_len_global)

    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)

    print(cfg)
    if not cfg.eval:
        workspace.train()
    else:
        workspace.generate_reward_curves()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_10/finetune_liv.yaml", help="Path to the config file")
    args = parser.parse_args()

    cfg = load_config_as_namespace(args.config)    
    # make new dataset with splited frames
    mean_video_len_global = make_dataset_from_video_dir(cfg.video_path, cfg.output_path)

    # make dataset yaml
    make_dataset_yaml(cfg.yaml_name, cfg.output_path)

    main()  # you also need to change the config files in liv/cfgs/config_liv.yaml, liv/cfgs/dataset/{dataset_name}.yaml, and liv/cfgs/training/finetune.yaml