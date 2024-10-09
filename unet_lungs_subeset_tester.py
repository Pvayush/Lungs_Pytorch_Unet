import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from skimage.transform import resize
import random


#Check if device has GPU/CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Classes

#Dataset Class
class LungCTDataset(Dataset): #this is to create a custom dataset to handle loading and preprocessing of images and mask

    def __init__(self, images_dir, masks_dir,  patch_size=(64, 128, 128), transform = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patch_size = patch_size
        self.transform = transform

        self.images_filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.masks_filenames = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
    
    def __len__(self):
        return len(self.images_filenames)
    
    def __getitem__(self, idx): #purpose of this function is to retrieve a single sample using an index
        image_path = os.path.join(self.images_dir, self.images_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_filenames[idx])

        image = np.load(image_path)
        mask = np.load(mask_path)

        image_patch, mask_patch = self.extract_random_patch(image, mask)

        # Normalize the image
        image_patch = (image_patch - np.mean(image_patch)) / np.std(image_patch)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).float().unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask_patch).long()

        print(f"Unique values in mask before processing: {np.unique(mask_patch)}")
        

        return image_tensor, mask_tensor
    
    def extract_random_patch(self, image, mask): #create random patch so it doesn't take too much memory for cpu
        depth, height, width = image.shape
        p_depth, p_height, p_width = self.patch_size

        # Check if image dimensions are smaller than patch size
        if depth < p_depth or height < p_height or width < p_width:
            # Resize image and mask to patch size
            image_resized = resize(image, self.patch_size, mode='constant', preserve_range=True)
            mask_resized = resize(mask, self.patch_size, order=0, mode='constant', preserve_range=True, anti_aliasing=False)
            return image_resized, mask_resized

        # Randomly select patch indices
        d_start = random.randint(0, depth - p_depth)
        h_start = random.randint(0, height - p_height)
        w_start = random.randint(0, width - p_width)

        # Extract patches
        image_patch = image[d_start:d_start + p_depth, h_start:h_start + p_height, w_start:w_start + p_width]
        mask_patch = mask[d_start:d_start + p_depth, h_start:h_start + p_height, w_start:w_start + p_width]

        return image_patch, mask_patch

#Convlution Operation Class

class ConvolutionOperation(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionOperation, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        return out


#Encoder Class

class EncoderBlock(nn.Module): #Simplifies Image/Reduces size
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_op = ConvolutionOperation(in_channels, out_channels)
        self.max_pool = nn.MaxPool3d(kernel_size=2)
            
    def forward(self, x):
        enc = self.conv_op(x)
        enc_pool = self.max_pool(enc)
        return enc, enc_pool


#Decoder Class

class DecoderBlock(nn.Module):#brings it back to original size
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_op = ConvolutionOperation(out_channels * 2, out_channels)
            
    def forward(self, x, skip):
        upsample = self.conv_transpose(x)
        concat = torch.cat((upsample, skip), dim=1)
        out = self.conv_op(concat)
        return out

#Unet Class
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvolutionOperation(512, 1024)

        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Output layer
        self.conv_final = nn.Conv3d(64, out_channels, kernel_size=1)
        self.activation = nn.Softmax(dim=1)  # Use dim=1 for channel-wise softmax

    def forward(self, x):
        # Encoder
        enc1, enc_pool1 = self.enc1(x)
        enc2, enc_pool2 = self.enc2(enc_pool1)
        enc3, enc_pool3 = self.enc3(enc_pool2)
        enc4, enc_pool4 = self.enc4(enc_pool3)

        # Bottleneck
        bottleneck = self.bottleneck(enc_pool4)

        # Decoder
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        # Output
        out = self.conv_final(dec1)
        out = self.activation(out)

        return out

#Functions

#Dice Loss Function


def dice_loss(pred, target, smooth=1e-5, background=False): #too gauge similarity between predicted and ground truth
    num_classes = pred.size(1)
    target_one_hot = F.one_hot(target, num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

    if not background:
        pred = pred[:, 1:, ...]
        target_one_hot = target_one_hot[:, 1:, ...]

    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target_one_hot.contiguous().view(target_one_hot.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    dice_loss = 1 - dice_score.mean()

    return dice_loss


def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Training function on a small subset
def train_on_subset():
    # Check if device has GPU/CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = UNet3D(in_channels=1, out_channels=3)
    model = model.to(device)

    # Reset model weights
    reset_model_weights(model)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Paths to your data directories
    images_dir = 'C:/Users/pvayu/OneDrive/Documents/Code/Lungs_Pytorch_Unet/images_preprocessed/images_preprocessed'
    masks_dir = 'C:/Users/pvayu/OneDrive/Documents/Code/Lungs_Pytorch_Unet/LR_segmentations/LR_segmentations'

    print(f"Images Directory: {images_dir}")
    print(f"Masks Directory: {masks_dir}")

    patch_size = (64, 128, 128)

    # Create the dataset
    dataset = LungCTDataset(images_dir, masks_dir, patch_size=patch_size)

    # Select a small subset
    subset_size = 5
    train_subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # Create a DataLoader for the subset
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=1,        # Batch size can remain 1
        shuffle=True,
        num_workers=1
    )

    # Number of epochs for testing
    num_test_epochs = 20

    # To record loss
    epoch_losses = []

    # Training loop
    for epoch in range(num_test_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in train_subset_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Outputs with softmax applied

            # Compute Dice loss
            loss = dice_loss(outputs, masks, background=False)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_subset_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_test_epochs}], Loss: {avg_loss:.4f}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_test_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs (Small Subset)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.xticks(range(1, num_test_epochs + 1))
    plt.show()

    # Optionally, save the model trained on the subset
    torch.save(model.state_dict(), 'unet3d_subset_trained_model.pth')
    print("Model trained on subset and saved as 'unet3d_subset_trained_model.pth'.")


#Main Execution
if __name__ == "__main__":
  
    train_on_subset()