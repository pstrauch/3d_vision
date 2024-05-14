import torch
import torch.utils
import numpy as np
import cv2 as cv
from torch import nn
import torch.optim as optim

from datetime import datetime

from  DataLoader import TrainDataLoader,ValDataLoader


from torchvision.models.video.swin_transformer import SwinTransformer3d 
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


### define model, basically any transformer and then we just change the heads
class DualHeatAction():

