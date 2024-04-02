import pandas as pd
import numpy as np
import torch
import os
from skimage import io
from data_reshape import DataReshape
from typing import List, Tuple, Optional, Dict

'''
This is one of two custom dataloader for the H2O dataset. 
This dataloader will return 32 frames from a single action.
It is intended to train models on the task of action recognition

The datapoints returned have the following shape:
1. hand pose: 32x126 (2 x 21 x 3) so 21 3d points for poth hands for each of the 32 frames
2. object pose: 32x155 (8 + 63 + 84) 8 one hot encoding of position + 21 3d points + 2 hand of 21 contact + 21 distant points. This for each frame
3. image: 32x256x455x3 The chosen images of this action
4. label: An index that can be used as ground truth
'''

class H2ODataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        type: str, 
    ) -> None:
        '''
        type: Type of data, in ['train', 'val', 'test'].
        '''
        self.data_dir = '../data/datasets/H2O'
        self.type = type

        self.pack_save_dir = '../data/datasets/H2O/data/packed_data'

        action_file = 'action_' + self.type + '.txt'
        action_path = os.path.join(self.data_dir, 'h2odataset/action_labels', action_file).replace("\\","/")
        self.actions = pd.read_csv(action_path, delimiter=' ')
        

        assert os.path.isdir(self.data_dir), f"{self.data_dir} does not exist!"
        assert os.path.isdir(self.pack_save_dir), f"{self.pack_save_dir} does not exist!"
        assert type in ['train', 'val', 'test'], "Invalid dataset type!"


                
        if type == 'train':
            hand_file = os.path.join(self.pack_save_dir, 'hand_poses_train.npy').replace("\\","/")
            object_file = os.path.join(self.pack_save_dir, 'obj_poses_train.npy').replace("\\","/")
            image_file = os.path.join(self.pack_save_dir, 'img_train.npy').replace("\\","/")
            label_file = os.path.join(self.pack_save_dir, 'labels_train.npy').replace("\\","/")

        elif type == 'val':
            hand_file = os.path.join(self.pack_save_dir, 'hand_poses_val.npy').replace("\\","/")
            object_file = os.path.join(self.pack_save_dir, 'obj_poses_val.npy').replace("\\","/")
            image_file = os.path.join(self.pack_save_dir, 'img_val.npy').replace("\\","/")
            label_file = os.path.join(self.pack_save_dir, 'labels_val.npy').replace("\\","/")

        elif type == 'test':
            hand_file = os.path.join(self.pack_save_dir, 'hand_poses_test.npy').replace("\\","/")
            object_file = os.path.join(self.pack_save_dir, 'obj_poses_test.npy').replace("\\","/")
            image_file = os.path.join(self.pack_save_dir, 'img_test.npy').replace("\\","/")
            #Does not actually contain any labels
            label_file = os.path.join(self.pack_save_dir, 'labels_test.npy').replace("\\","/")

        if os.path.exists(hand_file):
            self.hand_poses = np.load(hand_file)
            self.obj_poses = np.load(object_file)
            self.img_indices = np.load(image_file)
            self.labels = np.load(label_file)
        else:
            self.hand_poses, self.obj_poses, self.img_indices, self.labels = self.pack_and_pad()
            np.save(hand_file, self.hand_poses)
            np.save(object_file, self.obj_poses)
            np.save(image_file, self.img_indices)
            np.save(label_file, self.labels)
    
    def pack_and_pad(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # pack and pad hand_poses, object labels
        data_reshape = DataReshape(
            self.actions, 
            self.data_dir, 
            self.type, 
        )
        hand_poses, obj_poses, img_indices,  labels = data_reshape.pack_and_pad()
        return hand_poses, obj_poses, img_indices, labels


    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = np.zeros([self.img_indices.shape[1], 256, 455, 3])
        for j in range(self.img_indices.shape[1]):
            num = int(self.img_indices[idx, j])
            image_name = os.path.join(self.data_dir, 'data', self.actions['path'].values[idx], 'cam4/rgb256', str(num).zfill(6) + ".jpg" ).replace("\\","/")
            images[j,:,:] = io.imread(image_name)

        label = self.labels[idx]
        
        hand_pose = torch.from_numpy(self.hand_poses[idx, :, :]).float()
        obj_pose = torch.from_numpy(self.obj_poses[idx, :, :]).float()
        images = torch.from_numpy(images).int()
        return hand_pose, obj_pose, images, label
        
        
    def __len__(self):
        return self.labels.shape[0]
