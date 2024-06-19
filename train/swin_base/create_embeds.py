import torch
import torch.utils
import glob
import numpy as np
import natsort
import cv2 as cv
from torch import nn
import torch.optim as optim

from datetime import datetime

from seqloader import TrainData,ValData


from torchvision.models.video.swin_transformer import SwinTransformer3d 
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


def create_embedings():
    train_data = TrainData()
    val_data = ValData()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)

    swin_video = swin3d_t()
    in_features = swin_video._modules['head'].in_features
    out_features = 37
    swin_video._modules['head'] = nn.Linear(in_features, out_features, bias=True)

    swin_video.load_state_dict(torch.load("Weights/model_20240424_235945_10"))

    swin_video._modules['head'] = nn.Dropout(p=0)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    swin_video.to(device)

    swin_video.eval()

    index = 0

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if i % 3 == 2:
                print("train ", i /600 * 8)
            inputs = data[0].to(device)
            outputs = swin_video(inputs)
            out = outputs.to('cpu').numpy()
            for j in range(out.shape[0]):
                np.save("embedings/train/" + str(index + 1) + ".npy", out[j])
                index += 1
                
        index = 0
        for i, vdata in enumerate(val_loader):
            if i % 3 == 2:
                print("val ", i /200 * 8)
            inputs = vdata[0].to(device)
            outputs = swin_video(inputs)
            out = outputs.to('cpu').numpy()
            for j in range(out.shape[0]):
                np.save("embedings/val/" + str(index + 1) + ".npy", out[j])
                index += 1


if __name__=="__main__":
    create_embedings()