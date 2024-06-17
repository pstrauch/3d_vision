import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from datetime import datetime



n_frames_per_seq : int = 8
emb_dim : int = 768
predictor_input_size = emb_dim + n_frames_per_seq*42

h2o_root = './embedings/'
sample_root_train = h2o_root + f'seq_{n_frames_per_seq}_train/'
sample_root_val = h2o_root + f'seq_{n_frames_per_seq}_val/'

n_epochs = 240
lr = 0.001
momentum = 0.8
print_every_iters = 230
n_hid_cp = 1024
n_hidden = 5000

lambda_cp = 0.05


'''
Custom loss function to penalise wrong contact predictions. 
Since points in contact are more rare they are punished stronger to avoid that the network simply predicts no contacts.
'''
def my_loss(output, target):
    loss = torch.sum((output - target)**2 * (1 + 5 * target))
    return loss

'''
The model we use for predictions is a simple MLP with a Dropout layer to slightly help with overfitting
'''
class ActionPredictor(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        
        self.cp = nn.Sequential(nn.Linear(emb_dim, n_hid_cp),
                                 nn.ReLU(),
                                 nn.Linear(n_hid_cp, n_hid_cp),
                                 nn.ReLU(),
                                 nn.Linear(n_hid_cp, n_frames_per_seq*42)

        )
        
        self.fc1 = nn.Sequential(nn.Linear(predictor_input_size, n_hidden),
                                 nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU()                                
        )
        self.fc3 = nn.Sequential(nn.Linear(n_hidden, n_hidden)
        ) 


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        cp = self.cp(x)
        x = torch.cat((x, cp), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, cp

'''
Load the precomputed encodings from the resnet50 and prepare them for training the MLP
Initially only 80% of the training set is used to asses the potential of a model, however before making predictions it is retrained on the hole training set.
'''
class DataTrain(torch.utils.data.DataLoader):
    def __init__(self, mode, h2o_dir, sample_dir):
        assert mode in ['train', 'val'], f'Invalid mode {mode}. Expected: train, val'
        self.sample_dir = sample_dir
        self.emb_dir = sample_dir + f'emb_swin_{mode}/'
        # self.dist_dir = sample_dir + f'distances_{mode}/'
        self.dist_dir = sample_dir + f'cm_{mode}/'
        self.labels = np.load(h2o_dir + f'action_labels_{mode}.npy')
        self.n_actions = self.labels.shape[0]

    def __len__(self):
        return self.n_actions
    
    def __getitem__(self, idx):
        emb = np.load(self.emb_dir + f'{idx+1}.npy')
        dist = np.load(self.dist_dir + f'{(idx+1):03d}.npy').flatten()
        return emb.astype(np.float32), dist.astype(np.float32), int(self.labels[idx]-1)
'''
For training we utilise a setup that is roughly based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
To save on computation time validation is only done once every few iterations.
If a model performes better on the validation set then the previous best it is saved for later use.
'''
class Trainer():
    def __init__(self) -> None:
        data_train = DataTrain('train', h2o_root, sample_root_train)
        data_val = DataTrain('val', h2o_root, sample_root_val)
        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(data_val, batch_size=32, shuffle=False, num_workers=4)

        self.model = ActionPredictor()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        #self.model.load_state_dict(torch.load("Weights/model_20240520_215625_106"))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0

        best_accuracy = 0
        best_accuracy = 0
        for epoch in range(n_epochs):
            if epoch_number == 120:
                print("Freezing contact prediction")
                for param in self.model.cp.parameters():
                    param.requires_grad = False
            print('EPOCH {}:'.format(epoch_number + 1))

            # Set gradient tracking
            self.model.train(True)
            training_loss = self.train_one_epoch(epoch_number)
            print("Training Loss", training_loss)
            running_vlos = 0
            true_classifications = 0
            classifications = 0
            self.model.eval()
            with torch.no_grad():
                for i, (emb, dist,y) in enumerate(self.val_loader):
                    emb, dist, y_gpu = emb.to(self.device), dist.to(self.device), y.to(self.device)
                    voutputs, cp = self.model(emb)
                    _, out = torch.max(voutputs,1)
                    for j in range(out.shape[0]):
                        if out[j] == y[j]:
                            true_classifications += 1
                        classifications += 1

                    vloss = self.loss_fn(voutputs, y_gpu)
                    vloss += my_loss(cp, dist)
                    running_vlos += vloss.item()
                
            accuracy = true_classifications/classifications

            print("Validation Loss", running_vlos/classifications)
            print("Correct predictions: ", true_classifications, " of ", classifications, "Accuracy:", accuracy)
                
                # Track best performance, and save the model's state
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                model_path = 'Weights/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def train_one_epoch(self,epoch_index):
        running_loss = 0.

        last_loss = 0
        for i, (emb, dist,y) in enumerate(self.train_loader):
            emb, dist, y_gpu = emb.to(self.device), dist.to(self.device), y.to(self.device)
            

            self.optimizer.zero_grad()

            outputs, cp = self.model(emb)

            loss = self.loss_fn(outputs, y_gpu)
            loss_cp = my_loss(cp, dist)
            loss = loss + lambda_cp * loss_cp
            loss.backward()

            self.optimizer.step()
            

            running_loss += loss.item()
            
        return running_loss / (i + 1) / 32
    
        
if __name__=="__main__":
    trainer = Trainer()
    trainer.train()
