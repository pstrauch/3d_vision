# Required imports
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from seq_loader import ValData, TrainData
import os


# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', size=640)

# Define the feature extractor function for images and heatmaps
def preprocess_with_heatmap(images):
    # Apply the feature extractor transformations to the images
    encoding = feature_extractor(images=images, return_tensors="pt")
    pixel_values = encoding.pixel_values

    # Apply the same transformations to the heatmaps
    # def transform_heatmap(image, size):
    #     transform = transforms.Compose([
    #         transforms.Resize(size),
    #         transforms.ToTensor()
    #     ])
    #     return transform(image)
    
    # transformed_heatmaps = torch.stack([transform_heatmap(hm, (640, 640)) for hm in heatmaps])
    
    return pixel_values#, transformed_heatmaps


# Initialize dataset and dataloader
train_dataset = ValData()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load the pretrained ViT model
model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)

# Define the forward pass function to get attention maps
def forward(pixel_values):
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

    attentions = outputs.attentions[-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    w_featmap = pixel_values.shape[-2] // model.config.patch_size
    h_featmap = pixel_values.shape[-1] // model.config.patch_size
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
    print(attentions.shape)
    mean_attention = torch.mean(attentions, dim=0)

    return mean_attention

# Define the custom attention loss function
def custom_attention_loss(attentions, heatmaps):
    print(attentions.max(), attentions.min())

    print(heatmaps.max(), heatmaps.min())
    assert attentions.shape == heatmaps.shape, "Attention maps and heatmaps must have the same shape."
    loss = F.mse_loss(attentions, heatmaps)
    return loss
# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        img, hand_poses, label, obj_poses,heatmaps = batch
        images = img[:,:, 7, :, :]
        

        # Define the crop size
        crop_size = 360

        # Calculate the starting point for cropping
        start = (images.shape[-1] - crop_size) // 2

        # Crop the images
        cropped_images = images[:, :, :, start:start+crop_size]
        crop_size = 720
        start = (heatmaps.shape[-1] - crop_size) // 2
        heatmaps =  heatmaps[:,7, :, :]
        cropped_heatmaps = heatmaps[:, :, start:start+crop_size]
        cropped_heatmaps = cropped_heatmaps
        resized_heatmaps = nn.functional.interpolate(cropped_heatmaps.unsqueeze(1), size=(640, 640), mode='nearest').squeeze(1)
        # Convert to float
        resized_heatmaps = resized_heatmaps.float()

        # Normalize to [0, 1]
        resized_heatmaps = (resized_heatmaps - resized_heatmaps.min()) / (resized_heatmaps.max() - resized_heatmaps.min())
        
        pixel_values = preprocess_with_heatmap(cropped_images, resized_heatmaps)
        
        # Get attention maps
        attention= forward(pixel_values)
        attention = attention.float()
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        loss = custom_attention_loss(attention, resized_heatmaps.squeeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_vit_model.pth")
