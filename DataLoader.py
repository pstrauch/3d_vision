import torch
import torch.utils
import glob
import numpy as np
import natsort
import cv2 as cv
from torch import nn
import torch.optim as optim





class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dir):
        self.dir = dir
        self.frames_paths = natsort.natsorted(glob.glob(f"{dir}framesequences_train/*npy"))
        self.label_list = np.loadtxt(f"{dir}action_labels_train.txt")

    def __len__(self):
        return len(self.frames_paths)
    
    def __getitem__(self, idx):
        frames_path = self.frames_paths[idx]
        frames = np.load(frames_path)
        frames = torch.from_numpy(frames).float()
        label = int(self.label_list[idx])
        return frames, label

 
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
        frames = torch.from_numpy(frames).float()
        label = int(self.label_list[idx])
        return frames, label  




train_dataset = TrainDataLoader("/Users/dennisbaumann/Downloads/Dataset/")

val_dataset = ValDataLoader("/Users/dennisbaumann/Downloads/Dataset/")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)


for i, (data, label) in enumerate(train_loader):

    print(data.shape[1])
    for j in range(data.shape[1]):
        img = data[0,j].numpy()
        img = np.asarray(img, dtype=np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("1",img)
        if cv.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to close the window
            continue
    print(label)
    if i == 0:
        break
cv.destroyAllWindows()  # Close all OpenCV windows