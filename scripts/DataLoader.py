import torch
import torch.utils
import glob
import numpy as np
import natsort
import cv2 as cv
from torch import nn
import torch.optim as optim
from torchvision.transforms import v2





class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dir):
        self.dir = dir
        self.frames_paths = natsort.natsorted(glob.glob(f"{dir}framesequences_train/*npy"))
        self.label_list = np.loadtxt(f"{dir}action_labels_train.txt")

    def __len__(self):
        return len(self.frames_paths)
    
    def __getitem__(self, idx):
        offset = np.random.randint(4, size=1)[0]
        frames_path = self.frames_paths[idx]
        frames = np.load(frames_path)
        mask = np.zeros(32, dtype=bool)
        mask[[0+offset,4+offset,8+offset,12+offset,16+offset,20+offset,24+offset,28+offset]] = True
        frames = frames[mask,:,:,:]
        self.h_flip(frames=frames)
        frames = np.moveaxis(frames, -1, 0)
        frames = torch.from_numpy(frames).float()
        label = int(self.label_list[idx])
        return frames, label
    
    # 8x256x455x3
    def h_flip(self, frames: np.ndarray):
        prob = np.random.rand(1)
        if prob > 0.5:
            for i in range(frames.shape[0]):
                frames[i,:,:,:] = np.flip(frames[i,:,:,:], axis=1)


 
class ValDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dir):
        self.dir = dir
        self.frames_paths = natsort.natsorted(glob.glob(f"{dir}framesequences_val/*npy"))
        self.label_list = np.loadtxt(f"{dir}action_labels_val.txt")

    def __len__(self):
        return len(self.frames_paths)
    
    def __getitem__(self, idx):
        frames_path = self.frames_paths[idx]
        frames = np.load(frames_path)
        mask = np.zeros(32, dtype=bool)
        mask[[0,4,8,12,16,20,24,28]] = True
        frames = frames[mask,:,:,:]
        frames = np.moveaxis(frames, -1, 0)
        frames = torch.from_numpy(frames).float()
        label = int(self.label_list[idx])
        return frames, label  



if __name__=="__main__":
    train_dataset = TrainDataLoader("C:/Users/alexm/Downloads/Dataset/")

    val_dataset = ValDataLoader("C:/Users/alexm/Downloads/Dataset/")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)


    for i, (data, label) in enumerate(train_loader):

        print(data.shape[1])
        for j in range(data.shape[1]):
            img = data[0,j].numpy()
            img = np.asarray(img, dtype=np.uint8)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imshow("1",img)
            cv.waitKey(0)
        print(label)

        if i == 0:
            break
