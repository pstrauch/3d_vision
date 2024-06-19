# 3D Vision Hand Object interaction Recognition

# Requirements

Our project was developed using python 3.11. Here is a short list of the used packages:
- general python packages such as numpy, matplotlib, pandas, tqdm
- opencv
- pytorch
- transformers
- wandb

## Data

This repository comes without the data that is part of H2O. The dataset can be found at https://taeinkwon.com/projects/h2o/. We further provide the object meshes in this repo.

Since our scripts were run on different environments, data paths may differ. Please adjust the paths in the scripts according to your local environment and file hierarchy! For reference, we provide the hierarchies used for our Swin- and ViT-based frameworks below: 

**Swin-Based Unified Framework**
1. place the raw h2o dataset in data/h2o/h2odataset
2. place the object meshes in data/h2o/object_meshes
3. the extracted data by the scripts are placed in data/h2o/seq_n_mode (see scripts for further description and parameters)

**ViT Self Attention Shaping**
1. clone the h2o git repo into data/dataset/H2O (https://github.com/taeinkwon/h2odataset) (we need the action_labels)
2. download all subjectx_ego_v1_1.tar.gz from https://h2odataset.ethz.ch/data/ (x is either 1,2,3,4) 
3. extract all four files into data/datasets/H2O/data 
4. Rename them to subjectx (for example subject_1_ego to subject1)
5. place the object meshes in data/dataset/H2O/data


# Structure

- The data_extractors scripts are used to extract and reshape the raw data.
- The contact_gt and heatmap_gt scripts are used to generate the ground truth contact and heat maps.
- We supply data loaders for relevant training, val and test input in utils and training folders.
- In the train folder, scripts are provided to train the different methods. 
- Further utility scripts in utils and reprojections are used for generating plots but are not crucial to the project 

> Please note that any paths in the script must be adjusted to your local file hierarchy!
