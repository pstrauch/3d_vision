import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import natsort

frames_paths = glob.glob(r"C:\Users\chris\Desktop\3d_vision\3d_vision_data\hand_obj_heatmaps_val\*.npy")
frames_paths = natsort.natsorted(frames_paths)

for i, frames_path in tqdm(enumerate(frames_paths)):
    frames = np.load(frames_path)
    for j in range(frames.shape[0]):
        frame = frames[j]
        #plt.imshow(frame)
        #plt.show()
        np.save(f"C:/Users/chris/Desktop/3d_vision/3d_vision_data_reorganized/heatmaps_val/sequence{i}_frame{j}.npy", frame)
        #frame_2 = np.load(f"C:/Users/chris/Desktop/3d_vision/3d_vision_data_reorganized/frames_train/sequence{i}_frame{j}.npy")
        #plt.imshow(frame_2)
        #plt.show()