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
from torch.utils.tensorboard import SummaryWriter



class Trainer():

    def __init__(self):
        train_dataset = TrainDataLoader("C:/Users/alexm/Downloads/Dataset/")
        val_dataset = ValDataLoader("C:/Users/alexm/Downloads/Dataset/")

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=False)

        self.writer = SummaryWriter()




        self.swin_video = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        #print(self.swin_video._modules.keys())
        #print(self.swin_video._modules['patch_embed'])
        in_features = self.swin_video._modules['head'].in_features
        out_features = 37
        self.swin_video._modules['head'] = nn.Linear(in_features, out_features, bias=True)

        #print(self.swin_video)
        self.swin_video.load_state_dict(torch.load("Weights/model_20240424_071100_1"))

        #freeze all but last layer
        '''
        for param in self.swin_video.parameters():
            param.requires_grad = False
        for param in self.swin_video.head.parameters():
            a = param.requires_grad
            param.requires_grad = True
        '''
        
        #self.swin_video.head.0.weight.requires_grad = True
        #self.swin_video.head.0.bias.requires_grad = True
        #for param in self.swin_video._modules['']

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.swin_video.parameters())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.swin_video.to(self.device)
        #print(self.device)



    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0

        EPOCHS = 100

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.swin_video.train(True)
            avg_loss, accuracy_train = self.train_one_epoch(epoch_number)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.swin_video.eval()

            # Disable gradient computation and reduce memory consumption.
            true_classifications = 0
            classifications = 0
            
            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata[0].to(self.device), vdata[1].to(self.device)
                    voutputs = self.swin_video(vinputs)
                    outputs = np.argmax(voutputs.to('cpu').numpy(), axis=1)
                    for j in range(outputs.shape[0]):
                        if outputs[j] == vdata[1][j]:
                            true_classifications += 1
                        classifications += 1
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss
            
            avg_vloss = running_vloss / (i + 1)
            accuracy_val = true_classifications / classifications
            #Add data to tensorboard
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Loss/val", avg_vloss, epoch)
            self.writer.add_scalar("Accuracy/train", accuracy_train, epoch)
            self.writer.add_scalar("Accuracy/val", accuracy_val, epoch)
            self.writer.flush()

            print("Correct predictions: ", true_classifications, " of ", classifications)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'Weights/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.swin_video.state_dict(), model_path)

            epoch_number += 1
        self.writer.close()

    def train_one_epoch(self,epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        true_classifications = 0
        classifications = 0
        for i, data in enumerate(self.train_loader):
            #print("Training ", i , " of ", len(self.train_loader))
            # Every data instance is an input + label pair
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.swin_video(inputs)
            _, cpu_outputs = torch.max(outputs,1)
            for j in range(cpu_outputs.shape[0]):
                if cpu_outputs[j] == data[1][j]:
                    true_classifications += 1
                classifications += 1

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
            if i % 50 == 49:
                last_loss = running_loss / 50 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
            
        return running_loss / (i + 1), true_classifications/classifications


if __name__=="__main__":
    trainer = Trainer()
    trainer.train()