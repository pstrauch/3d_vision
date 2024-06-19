# -*- coding: utf-8 -*-
"""
Code for training a Vision Transformer (ViT) model with self attention shaping 
for action prediction using RGB images and heatmaps as input modalities. 
The model is trained on the H2O dataset, which contains sequences of RGB images
and corresponding hand and object heatmaps. The model is trained using a


This code is intended to be run in Google Colab. To run the code, you need to reformat jupyter notebook.
"""


#****colab dependencies:*******
# !pip install transformers==4.40.2

# !pip install wandb

# from google.colab import drive
# drive.mount('/content/drive')
################
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import json
from tqdm import tqdm
import os

!wandb login

import random
seed = 7
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed)


#path to dataset 
h2o_root = '/content/drive/My Drive/vit_3d/' 



def process_images(rgb_images, hand_heatmaps,obj_heatmaps):
    img_size =496

    images = rgb_images.unsqueeze(0).permute(2, 1, 3, 4, 0).squeeze(-1)  # Reshape to (batch_size * sequence_length, 3, H, W)

    # Resize grayscale images
    crop_size = 360
    start = (images.shape[-1]) // 2
    #start = (grayscale_images.shape[-1]) // 2
    cropped_grayscale = images[:,:, :, int(start - crop_size / 2):int(start + crop_size / 2)]#grayscale_

    resized_grayscale = nn.functional.interpolate(cropped_grayscale, size=(img_size, img_size), mode='nearest').squeeze(0)

    # Resize heatmaps
    crop_size = 720
    start = (hand_heatmaps.shape[-1] - crop_size) // 2
    cropped_hand_heatmaps = hand_heatmaps[:,:, start:start + crop_size]
    cropped_hand_heatmaps =cropped_hand_heatmaps.unsqueeze(1)

    resized_hand_heatmaps = nn.functional.interpolate(cropped_hand_heatmaps, size=(img_size, img_size), mode='nearest').squeeze(0)
    resized_hand_heatmaps = resized_hand_heatmaps.float() # Add channel dimension
    resized_hand_heatmaps = resized_hand_heatmaps.view( 1,8, img_size, img_size).squeeze(0).unsqueeze(1)

    cropped_obj_heatmaps =obj_heatmaps[:,:, start:start + crop_size]
    cropped_obj_heatmaps =cropped_obj_heatmaps.unsqueeze(1)

    resized_obj_heatmaps = nn.functional.interpolate(cropped_obj_heatmaps, size=(img_size, img_size), mode='nearest').squeeze(0)
    resized_obj_heatmaps = resized_obj_heatmaps.float() # Add channel dimension
    resized_obj_heatmaps = resized_obj_heatmaps.view( 1,8, img_size, img_size).squeeze(0).unsqueeze(1)


    #print(resized_grayscale.shape)
    return resized_grayscale, resized_hand_heatmaps, resized_obj_heatmaps


class TrainData(torch.utils.data.DataLoader):
    def __init__(self):
        self.data_path = h2o_root + "seq_8_train/"
        self.img_path = self.data_path + "frames_train(1)/"
        self.hand_path = self.data_path + "poses_hand_train/"
        self.obj_poses = self.data_path + "poses_obj_train/"
        self.num_actions = len(os.listdir(self.hand_path))
        self.labels = np.load(h2o_root + "action_labels_train.npy")

    def __len__(self):
        return self.num_actions

    def __getitem__(self, idx):
        img = np.load(self.img_path + format(idx + 1, '03d') + ".npy")
        hand_heatmap = np.load(self.data_path + "heatmaps_train/" + format(idx + 1, '03d') + ".npy")
        obj_heatmap = np.load(self.data_path + "obj_heatmaps_train/" + format(idx + 1, '03d') + ".npy")
        label = self.labels[idx]

        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).float()
        #hand_heatmap[hand_heatmap > 0] = 255
        hand_heatmap =torch.from_numpy(hand_heatmap/255.0).float()
        obj_heatmap =torch.from_numpy(obj_heatmap/255.0).float()
        img,hand,obj = process_images(img,hand_heatmap,obj_heatmap)
        return img, hand, obj, label


class ValData(torch.utils.data.DataLoader):
    def __init__(self):
        self.data_path = h2o_root + "seq_8_val/"
        self.img_path = self.data_path+ "frames_val(1)/"
        self.hand_path = self.data_path + "poses_hand_val/"
        self.obj_path  = self.data_path + "poses_obj_val/"
        self.num_actions = len(os.listdir(self.hand_path))
        self.labels = np.load(h2o_root + "action_labels_val.npy")

    def __len__(self):
        return self.num_actions

    def __getitem__(self, idx):
        img = np.load(self.img_path + format(idx + 1, '03d') + ".npy")
        hand_poses = np.load(self.hand_path + format(idx + 1, '03d') + ".npy")
        obj_poses = np.load(self.obj_path + format(idx + 1, '03d') + ".npy")
        hand_heatmap = np.load(self.data_path + "heatmaps_val/" + format(idx + 1, '03d') + ".npy")
        obj_heatmap = np.load(self.data_path + "obj_heatmaps_val/" + format(idx + 1, '03d') + ".npy")
        label = self.labels[idx]


        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).float()
        #hand_heatmap[hand_heatmap > 0] = 255
        hand_heatmap =torch.from_numpy(hand_heatmap/255.0).float()
        obj_heatmap =torch.from_numpy(obj_heatmap/255.0).float()
        img,hand,obj = process_images(img,hand_heatmap,obj_heatmap)
        return img, hand, obj, label

class InterleaveHeatmapViTActionPredictionModel(nn.Module):
    def __init__(self, vit_model, num_classes, sequence_length):
        super(InterleaveHeatmapViTActionPredictionModel, self).__init__()
        self.vit_model = vit_model
        self.sequence_length = sequence_length

        self.classifier = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size * sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        sequence_length = self.sequence_length

        # Forward pass through ViT
        outputs = self.vit_model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size * sequence_length, hidden_size)
        attentions = outputs.attentions[-1]  # Get the attention maps from the last layer
        # Process attentions to match the input image resolution
        num_heads = attentions.shape[1]
        num_tokens = attentions.shape[-1] - 1
        attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_heads, num_tokens)

        w_featmap = pixel_values.shape[-2] // self.vit_model.config.patch_size
        h_featmap = pixel_values.shape[-1] // self.vit_model.config.patch_size
        attentions = attentions.reshape(batch_size, num_heads, w_featmap, h_featmap)
        attentions = F.interpolate(attentions, scale_factor=self.vit_model.config.patch_size, mode="nearest")
        attentions = attentions.view(batch_size, num_heads, pixel_values.shape[-2], pixel_values.shape[-1])
        attentions = (attentions - attentions.min()) / (attentions.max() - attentions.min())

        # Split attentions into hand and object heatmaps
        hand_attention = attentions[:,0:10,:,:]
        obj_attention = attentions[:,10:12,:,:]

        #pooling modality
        mean_hand = torch.mean(hand_attention, dim=1)
        mean_obj = torch.mean(obj_attention, dim=1)
        
        # Reshape and concatenate embeddings
        concatenated_embeddings = last_hidden_state.reshape(batch_size // sequence_length, sequence_length * self.vit_model.config.hidden_size)

        # Pass through the classifier
        logits = self.classifier(concatenated_embeddings)

        return logits, mean_hand,mean_obj

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', size=496)
#feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16',size =496)

#vit_model = torch.load(h2o_root + 'model.pth')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer = False)
#
#vit_model = ViTModel.from_pretrained('facebook/dino-vitb16', add_pooling_layer=False)


# Define the action prediction model
num_classes = 37 # Change this to the number of action classes in your dataset
sequence_length = 8
model = InterleaveHeatmapViTActionPredictionModel(vit_model, num_classes, sequence_length)

# Load the dataset
train_dataset = TrainData()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=4,
    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1)),
    generator=torch.Generator().manual_seed(seed))
val_dataset = ValData()
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)




#main training loop
img_size = 496
# Define the optimizer and loss functionf
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

#logging parameters and paths
image_save_dir =  os.path.join(h2o_root, 'evo_new_new_496/')
os.makedirs(image_save_dir, exist_ok=True)
# Initialize variables for best model saving
best_val_loss = float('inf')
save_model_path = h2o_root + 'models__new496/'
os.makedirs(save_model_path, exist_ok=True)

# Training and validation loop
num_epochs = 15
alpha = 1.0  # Initial weight for classification loss
beta = 30.0   # Initial weight for heatmap loss
gamma = 10.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_class_loss =0.0
    running_hand_mse = 0.0
    running_obj_mse = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):

        optimizer.zero_grad()
        imgs,hand_heatmap, obj_heatmap,label = batch
        imgs =imgs.squeeze(0)
        hand_heatmap = hand_heatmap.squeeze(0)
        obj_heatmap = obj_heatmap.squeeze(0)
        pixel_values = feature_extractor(images=imgs, return_tensors="pt").pixel_values
        pixel_values =  pixel_values.to(device)#pixel_values.to(device)
        label = label.to(device)

        # Forward pass
        logits, hand_attention, obj_attention = model(pixel_values)#,free_attention
        loss_class = criterion(logits, label)
        running_class_loss += loss_class.item()

        hand_heatmap = hand_heatmap.squeeze(1).to(device)
        obj_heatmap = obj_heatmap.squeeze(1).to(device)

        #heatmap stuff
        obj_mse = F.mse_loss(obj_attention, obj_heatmap)
        hand_mse = F.mse_loss(hand_attention,hand_heatmap)
        running_hand_mse += obj_mse.item()
        running_obj_mse += hand_mse.item()

        loss =  alpha * loss_class + beta* hand_mse + gamma * obj_mse

        running_loss += loss.item()
        loss.backward()
        optimizer.step()



    avg_train_loss = running_loss / len(train_loader)
    avg_class_loss = running_class_loss / len(train_loader)
    avg_hand_mse = running_hand_mse / len(train_loader)
    avg_obj_mse = running_obj_mse / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f} ,class: {avg_class_loss:.4f},hand:{avg_hand_mse:.4f},obj:{avg_obj_mse:.4f}")

    # Log training loss to wandb
    wandb.log({"train_loss": avg_train_loss, "class_loss": avg_class_loss,"hand_mse":avg_hand_mse, "obj_mse": avg_obj_mse, "epoch": epoch + 1})

    # Validation loop
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    val_loss = 0.0
    val_class_loss = 0.0
    val_hand_mse = 0.0
    val_obj_mse = 0.0


    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")):
            imgs,hand_heatmap, obj_heatmap,label = batch
            imgs =imgs.squeeze(0)
            hand_heatmap = hand_heatmap.squeeze(0)
            obj_heatmap = obj_heatmap.squeeze(0)
            pixel_values = feature_extractor(images=imgs, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            label = label.to(device)


            # Forward pass
            logits, hand_attention, obj_attention = model(pixel_values)#,free_attention
            loss_class = criterion(logits, label)
            val_class_loss += loss_class.item()

            hand_heatmap = hand_heatmap.squeeze(1).to(device)
            obj_heatmap = obj_heatmap.squeeze(1).to(device)

            #heatmap stuff
            obj_mse = F.mse_loss(obj_attention, obj_heatmap)
            hand_mse = F.mse_loss(hand_attention,hand_heatmap)

            val_hand_mse += obj_mse.item()
            val_obj_mse += hand_mse.item()
            loss =  alpha * loss_class + beta* hand_mse + gamma * obj_mse
            val_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()


            hand_attention = hand_attention.unsqueeze(1)
            obj_attention = obj_attention.unsqueeze(1)
            #free_attention = free_attention.unsqueeze(1)



            # Log images, heatmaps, and attention masks to wandb
            if batch_idx in [0]:#,11,15,63,73
                save_batch_dir = os.path.join(image_save_dir, f'image_{batch_idx + 1}')
                os.makedirs(save_batch_dir, exist_ok=True)
                # Store images, heatmaps, and attentions for logging
                # Log first eight images
                fig, axes = plt.subplots(5, 8, figsize=(20, 10))
                for j in range(8):

                    img_np = pixel_values[j].permute(1, 2, 0).cpu().numpy()  # Transpose to (height, width, channels)
                    img_save = imgs[j].permute(1, 2, 0).cpu().numpy()
                    axes[0,j].imshow(img_np)
                    axes[0,j].set_title(f'Input Image {j+1}')
                    axes[0,j].axis('off')

                    heat_np = hand_heatmap[j].cpu().numpy()  # Transpose to (height, width, channels)
                    axes[1,j].imshow(heat_np, cmap='hot', interpolation='nearest')
                    axes[1,j].set_title(f'hand_heatmap {j+1}')
                    axes[1,j].axis('off')

                    obj_np = obj_heatmap[j].cpu().numpy()  # Attention map

                    axes[2, j].imshow(obj_np, cmap='hot', interpolation='nearest')
                    axes[2, j].set_title('obj heat Map')
                    axes[2, j].axis('off')

                    hand_attention_np = hand_attention[j].squeeze(0).cpu().numpy()  # Attention map

                    axes[3, j].imshow(hand_attention_np, cmap='hot', interpolation='nearest')
                    axes[3, j].set_title('hand attention')
                    axes[3, j].axis('off')

                    obj_attention_np = obj_attention[j].squeeze(0).cpu().numpy()  # Attention map

                    axes[4, j].imshow(obj_attention_np, cmap='hot', interpolation='nearest')
                    axes[4, j].set_title('obj attention')
                    axes[4, j].axis('off')
                    plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_original_{j}.png'), img_save / 255)
                    plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_gt_hand_{j}.png'), heat_np, cmap='hot')
                    plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_gt_obj_{j}.png'), obj_np, cmap='hot')
                    plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_hand_{j}.png'), hand_attention_np, cmap='hot')
                    plt.imsave(os.path.join(save_batch_dir, f'{epoch+1}_obj_{j}.png'), obj_attention_np, cmap='hot')
                   
                action_prediction = predicted[0].item()
                ground_truth = label[0].item()
                fig.suptitle(f'Action Prediction: {action_prediction}, Ground Truth: {ground_truth}')
                wandb.log({"val_image_{}": wandb.Image(fig)})
                plt.close(fig)

    accuracy = correct_predictions / total_predictions
    avg_val_class_loss = val_class_loss / len(val_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_hand_mse = val_hand_mse / len(val_loader)
    avg_val_obj_mse = val_obj_mse / len(val_loader)

    print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f},val_class_loss:{avg_val_class_loss:.4f},val_hand_mse:{avg_val_hand_mse:.4f} ,val_obj_mse:{avg_val_obj_mse:.4f} ")

    # Log validation metrics to wandb
    wandb.log({"val_accuracy": accuracy, "val_loss": avg_val_loss,"val_class_loss": avg_val_class_loss,"val_hand_mse": avg_val_hand_mse,"val_obj_mse": avg_obj_mse, "epoch": epoch + 1})

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = save_model_path + f"{epoch:02d}_mean_dual_10_2_attention_496_1_30_10_action_prediction_model_{avg_val_loss:.2f}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {avg_val_loss:.4f}")

# Save the final trained model
final_model_path = h2o_root + "final_concatenated_action_prediction_model.pth"
torch.save(model.state_dict(), final_model_path)



# Test the model on the test set
class TestData(torch.utils.data.DataLoader):
    def __init__(self):
        self.img_path = h2o_root + "framesequences_8_test/"
        self.num_actions = int(len(os.listdir(self.img_path)))

    def __len__(self):
        return self.num_actions

    def __getitem__(self, idx):
        img = np.load(self.img_path + format(idx + 1) + ".npy")
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).float()

        return img


# Load the test dataset
test_dataset = TestData()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
save_model_path = h2o_root + 'models_496/'

model_files = [os.path.join(save_model_path, f) for f in os.listdir(save_model_path) if f.endswith('.pth')]
models = []
print(sorted(model_files))
for model_file in sorted(model_files):
    model = InterleaveHeatmapViTActionPredictionModel(vit_model, num_classes, sequence_length)
    model.load_state_dict(torch.load(model_file))
    models.append(model)

#evaluate the model on the test set
json_base = save_model_path + 'action_results/'

os.makedirs(json_base, exist_ok=True)
print(len(models))

for i,m in enumerate(models):
    m.to(device)
    m.eval()
    epoch =0
    num_epochs = 1
    img_size =224
    predictions ={}
    predictions["modality"] = "training: rgb + heatmaps, test: rgb"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"testdation Epoch {epoch + 1}/{num_epochs}")):
            img = batch
            images = img.permute(2, 1, 3, 4, 0).squeeze(-1) # Reshape to (batch_size * sequence_length, 3, H, W)
            # Define the crop size and preprocess images
            crop_size = 360
            start = (images.shape[-1]) // 2
            cropped_images = images[:, :, :, int(start-crop_size/2):int(start+crop_size/2)]
            cropped_images = nn.functional.interpolate(cropped_images, size=(img_size, img_size), mode='nearest')
            pixel_values = feature_extractor(images=cropped_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Forward pass
            logits, hand,obj = m(pixel_values)#,obj


            _, predicted = torch.max(logits.data, 1)

            predictions[f'{batch_idx + 1}'] = predicted[0].item()

            action_prediction = predicted[0].item()

    #write results to json file
    with open(json_base + f'action_labels{i}.json', 'w') as json_file:
        json.dump(predictions, json_file)


