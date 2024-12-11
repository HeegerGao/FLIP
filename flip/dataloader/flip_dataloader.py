import torch
from torch.utils.data import Dataset
import os
import pickle
from torchvision import transforms
import pickle
import h5py
import numpy as np

class FLIPDataset(Dataset):
    '''
        Load a preprocessd hdf5 file and return a dictionary
    '''
    def __init__(
        self, 
        data_dir, 
        obs_history,
        flow_horizon,
        calculate_scale_and_direction=False, 
        return_future_img=False,
        data_aug=None,
        flow_aug=False,
        eye_in_hand=False,
    ):
        self.data_dir = data_dir    # h5 and pkl files
        self.calculate_scale_and_direction = calculate_scale_and_direction
        self.return_future_img = return_future_img
        self.obs_history = obs_history
        self.flow_horizon = flow_horizon
        self.flow_aug = flow_aug

        self.data_aug = None
        if data_aug:
            self.data_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1)),
                transforms.ToTensor(),
        ])
        
        print("Loading preprocessed data...")
        if eye_in_hand:
            self.dataset = h5py.File(os.path.join(self.data_dir, "preprocessed_data_eye_in_hand.hdf5"), 'r')
        else:
            self.dataset = h5py.File(os.path.join(self.data_dir, "preprocessed_data.hdf5"), 'r')

        self.points = torch.tensor(self.dataset["points"][:])
        self.scale = torch.tensor(self.dataset["scale"][:])
        self.direction = torch.tensor(self.dataset["direction"][:])
        self.sentence_embedding_name = [name.decode("utf-8") for name in self.dataset["sentence_embedding_name"][:]]
        self.idx_indicator = self.dataset["idx_indicator"][:]

        # load the language embedding file
        with open(os.path.join(self.data_dir, "sentence_embeddings.pkl"), 'rb') as file:
            self.sentence_embeddings = pickle.load(file)

    def __len__(self):
        return np.sum(self.idx_indicator == 1)

    def find_nth_one_index(self, arr, idx):
        one_indices = np.where(arr == 1)[0]
        
        if idx >= len(one_indices) or idx < 0:
            return f"{idx} is out of range"
        
        return one_indices[idx]

    def __getitem__(self, idx):
        true_idx = self.find_nth_one_index(self.idx_indicator, idx)

        sample = {
            'observation': torch.tensor(self.dataset["video"][true_idx-self.obs_history+1:true_idx+1]), 
            'flow': self.points[true_idx],
            'sentence_embedding': self.sentence_embeddings[self.sentence_embedding_name[true_idx]].float(),
        }

        if self.data_aug is not None:
            for i in range(sample['observation'].shape[0]):
                sample['observation'][i] = self.data_aug(sample['observation'][i])

        if self.calculate_scale_and_direction:
            sample['scale'] = self.scale[true_idx, :, :-1]    # len = flow_horizon - 1
            sample['direction'] = self.direction[true_idx, :, :-1]
        if self.return_future_img:
            sample['future_img'] = torch.tensor(self.dataset["video"][true_idx+1:true_idx+self.flow_horizon])   # len = flow_horizon-1
            if self.data_aug is not None:
                for i in range(sample['future_img'].shape[0]):
                    sample['future_img'][i] = self.data_aug(sample['future_img'][i])
        
        if self.flow_aug:
            sample['flow'] = sample['flow'] + torch.randn_like(sample['flow'])
        
        return sample
