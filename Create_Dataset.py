import pandas as pd
import numpy as np
from numpy.typing import NDArray
import cv2 as cv
from tqdm import tqdm
import natsort
import glob
from PIL import Image
import matplotlib.pyplot as plt

def sample_frames(path: str, start_act: int, end_act: int, num_frames: int) -> NDArray:
    
    sequence_len: int = end_act - start_act
    sample_interval: int = int(sequence_len/(num_frames-1))
    current_frame: int = start_act
    frames_idx: list[int] = []

    #print("start_act: ", start_act, "end_act: ", end_act)
    for i in range(num_frames): 
        frames_idx.append(current_frame)
        current_frame += sample_interval
    #print(frames_idx)
    frames: list[NDArray] = []
    for i in range(num_frames):
        img_id: str = f"{frames_idx[i]}"
        img_id = img_id.zfill(6)
        read_path: str = f"D:/{path}/cam4/rgb256/{img_id}.jpg"
        img: Image = Image.open(read_path)
        img = img.convert("RGB")
        img: NDArray[np.uint8] = np.asarray(img, dtype=np.uint8)
        frames.append(img)
    frames: NDArray = np.stack(frames, axis=0)
    return frames


    
    

if __name__ == "__main__":
    df : pd.DataFrame = pd.read_csv("D:/TimeSFormer/action_val.txt", delimiter=" ")
    ids: list[int] = df["id"].to_list()
    paths: list[str] = df["path"].to_list()
    labels: list[int] = df["action_label"].to_list()
    start_acts: list[int] = df["start_act"].to_list()
    end_acts: list[int] = df["end_act"].to_list()
    num_frames: int = 8

    for i in tqdm(range(0, len(ids))):
        id: int = ids[i]
        path: str = paths[i]
        start_act: int = start_acts[i]
        end_act: int = end_acts[i]
        frames = sample_frames(path, start_act, end_act, num_frames)
        np.save(f"D:/framesequences_8_val/{id}.npy", frames)
    
    np.savetxt("D:/labels_val.txt", labels)



