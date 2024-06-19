# 3D Vision Hand Object interaction Recognition

This repository comes without the data that is part of H2O. The dataset can be found at https://taeinkwon.com/projects/h2o/.

For the data loader to work, the data has to be placed in specific directories. Place it according to the following steps.
1. clone the h2o git repo into data/dataset/H2O (https://github.com/taeinkwon/h2odataset) (we need the action_labels)
2. download all subjectx_ego_v1_1.tar.gz from https://h2odataset.ethz.ch/data/ (x is either 1,2,3,4) 
3. extract all four files into data/datasets/H2O/data 
4. Rename them to subjectx (for example subject_1_ego to subject1)
5. from CASAR/code/H2O/dataset/representation copy object_ply_correct to data/dataset/H2O/data
You should be good to go

# Requirements
python == 3.11
transformers == 4.40.2
wandb

# Structure
we supply data loaders and 
 
# Dataset
-At the first run the data has to be reshaped. This may take some time but you should be updated on the progress in your console. Once it has been created it will be stored locally on your machine.
If you need to change the reshaping, please delete the files in the data/datasets/H2O/packed_data directory. Otherwise, your changes may not take effect.
