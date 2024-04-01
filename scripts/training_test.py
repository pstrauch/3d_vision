#import torch
import sys
sys.path.append('../data/datasets/H2O/data_loader')
from H2O_dataset_action import H2ODataset


def train():
    #hello_world()
    dl = H2ODataset("train")
    a = dl[0]
    c = dl[1]
    b = len(dl)
    some = 0



if __name__ == "__main__":
    train()