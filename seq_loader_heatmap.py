import torch
import torch.utils
import numpy as np
import os
import cv2


'''
Loads action sequences from the H2O dataset, 8 frames per action. The dataset first needs to be extracted
-Dataset:
    The Dataset can be downloaded from https://polybox.ethz.ch/index.php/apps/files/?dir=/Shared/3DV&fileid=3656275966 
-Location:
    h2o_root should be the location of seq_8_train seq_8_val and the action labels
-Output:
    Image output will be (3x360x640) (channels x heightx width)
    hand poses ?
    The Label is a single integer between 1 and 36



'''
h2o_root = '/Users/dennisbaumann/3d_vision/data/'


class TrainData(torch.utils.data.DataLoader):
    def __init__(self):
        self.data_path = h2o_root + "seq_8_train/"
        self.img_path = self.data_path + "frames_train(1)/"
        self.hand_path = self.data_path + "poses_hand_train/"
        self.obj_poses = self.data_path + "poses_obj_train/"
        self.num_actions = len(os.listdir(self.hand_path))
        self.labels = np.load(h2o_root + "action_labels_train.npy")

    def __len__(self):
        return self.num_actions
    
    def __getitem__(self, idx):
        img = np.load(self.img_path + format(idx + 1, '03d') + ".npy")
        hand_poses = np.load(self.hand_path + format(idx + 1, '03d') + ".npy")
        obj_poses = np.load(self.obj_poses + format(idx + 1, '03d') + ".npy")
        heatmap = np.load(self.data_path + "heatmaps_train/" + format(idx + 1, '03d') + ".npy")
        label = self.labels[idx]

        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).float()

        return img, hand_poses, label, obj_poses, heatmap
    
    
class ValData(torch.utils.data.DataLoader):
    def __init__(self):
        self.data_path = h2o_root + "seq_8_val/"
        self.img_path = self.data_path + "frames_val(1)/"
        self.hand_path = self.data_path + "poses_hand_val/"
        self.obj_path  = self.data_path + "poses_obj_val/"
        self.num_actions = len(os.listdir(self.hand_path))
        self.labels = np.load(h2o_root + "action_labels_val.npy")

    def __len__(self):
        return self.num_actions
    
    def __getitem__(self, idx):
        img = np.load(self.img_path + format(idx + 1, '03d') + ".npy")
        hand_poses = np.load(self.hand_path + format(idx + 1, '03d') + ".npy")
        obj_poses = np.load(self.obj_path + format(idx + 1, '03d') + ".npy")
        heatmap = np.load(self.data_path + "heatmaps_val/" + format(idx + 1, '03d') + ".npy")
        label = self.labels[idx]
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).float()

        return img, hand_poses, label, obj_poses, heatmap
    



if __name__=="__main__":
    train_dataset = TrainData()

    val_dataset = ValData()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)


    for i, (img, hand_poses, label, obj_poses) in enumerate(train_loader):
        print(label)
        print(obj_poses.shape)
        img = img.numpy()
        for j in range(img.shape[0]):

            tmp = img[:,j]
            tmp = np.asarray(tmp, dtype=np.uint8)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            cv2.imshow("1",tmp)
            cv2.waitKey(0)

        if i == 0:
            break