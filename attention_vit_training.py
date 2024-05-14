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
def preprocess_with_heatmap(images, heatmaps):
    # Apply the feature extractor transformations to the images
    encoding = feature_extractor(images=images, return_tensors="pt")
    pixel_values = encoding.pixel_values

    # Apply the same transformations to the heatmaps
    def transform_heatmap(image, size):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        return transform(image)
    
    transformed_heatmaps = torch.stack([transform_heatmap(hm, (640, 640)) for hm in heatmaps])
    
    return pixel_values, transformed_heatmaps

# Initialize dataset and dataloader
train_dataset = ValData()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load the pretrained ViT model
model = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)

# Define the forward pass function to get attention maps
def forward(pixel_values):
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)
    attentions = outputs.attentions[-1]  # Attention maps from the last layer
    nh = attentions.shape[1]  # Number of heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # Reshape to keep only output patch attention
    return attentions

# Define the custom attention loss function
def custom_attention_loss(attentions, heatmaps):
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
        heatmaps =  heatmaps[:,7, :, :]
        # Preprocess images and heatmaps
        pixel_values, transformed_heatmaps = preprocess_with_heatmap(images, heatmaps)
        
        # Get attention maps
        attentions = forward(pixel_values)
        
        # Compute custom attention loss
        loss = custom_attention_loss(attentions, transformed_heatmaps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_vit_model.pth")
