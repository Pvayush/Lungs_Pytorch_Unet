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
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


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

        return out

#Functions
criterion_ce = nn.CrossEntropyLoss()
criterion_dice = DiceLoss(to_onehot_y=True, softmax=True)

#Main Execution
if __name__ == "__main__":
  
    # Initialize the model
    model = UNet3D(in_channels=1, out_channels=3)
    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Paths to my data directories
    images_dir = 'C:/Users/pvayu/OneDrive/Documents/Code/Lungs_Pytorch_Unet/images_preprocessed/images_preprocessed'
    masks_dir = 'C:/Users/pvayu/OneDrive/Documents/Code/Lungs_Pytorch_Unet/LR_segmentations/LR_segmentations'

    print(f"Images Directory: {images_dir}")
    print(f"Masks Directory: {masks_dir}")


    patch_size = (64, 128, 128)

    # Create the dataset
    dataset = LungCTDataset(images_dir, masks_dir, patch_size = patch_size)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 1 
    num_workers = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Training loop
    num_epochs = 50

    # To record loss
    epoch_losses = []
  

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  
            # Compute Dice loss

            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_ce + loss_dice 
        
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs (Small Subset)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))
    plt.show()

    dice_metric = DiceMetric(to_onehot_y=True, include_background=True, reduction="mean")

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            dice_metric(y_pred=outputs, y= masks)

        dice_score = dice_metric.aggregate().item()
        print(f"Validation Dice Score: {dice_score:.4f}")
        dice_metric.reset()

    # Save the model
    torch.save(model.state_dict(), 'unet3d_model.pth')

    




