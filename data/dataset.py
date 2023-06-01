import json
import random
from typing import List
import torch 
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np

class TrainDataset(Dataset):

    def __init__(self, json_path: str, name='workout') -> None:
        super(TrainDataset, self).__init__()
        self.json_path = json_path
        self.pose = 'pose'
        self.img_list = self.get_img_list()
        if name == 'yoga':
            self.pose = 'pose'
        elif name == 'workout':
            self.pose = 'label'
        # self.transforms = transforms
        self.label_to_index, self.num_classes = self.classes_to_idx()

    def get_img_list(self):
        with open(self.json_path, 'rb') as f:
            img_list = json.load(f)
        return img_list
    
    def classes_to_idx(self):
        label_to_index = {}
        index = 0
        for item in self.img_list:
            # print(item)
            label = item[self.pose]
            if label not in label_to_index:
                label_to_index[label] = index
                index += 1
        return label_to_index, len(label_to_index)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        a_img = self.img_list[index]
        a_label = a_img[self.pose]
        a_pose = a_img['pose_landmarks']

        a_pose_tensor = self.get_pose_tensor(a_pose)
        # p_pose_tensor = self.get_pose_tensor(p_pose)
        # n_pose_tensor = self.get_pose_tensor(n_pose)

        a_label_index = self.label_to_index[a_label]
        # return torch.concat((a_pose_tensor[:,:2], a_pose_tensor[:,-1].unsqueeze_(-1)), dim=1), a_label_index
        return a_pose_tensor[:,:2], a_label_index
    # def get_img_tensor(self, img_path):
    #     img = Image.open(img_path)
    #     img_tensor = self.transforms(img)
    #     return img_tensor

    def get_pose_tensor(self, pose: List[List[float]]):
        return torch.tensor(pose, dtype= float)
    def get_pose_index(self):
        return self.label_to_index
    
def split_indices(n, val_pct=0.1, seed=99):
    # Determine size of validation set
    n_val = int(val_pct*n*0.5)
    n_test = n_val
    # Set the random seed
    np.random.seed(seed)
    # Create permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val + n_test:], idxs[:n_val], idxs[n_val:n_val+n_test]
