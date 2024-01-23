import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, folder_X, folder_Y, resized_height=270, resized_width=540, augment=True):
        self.folder_X = folder_X
        self.folder_Y = folder_Y
        self.resize_height = resized_height
        self.resize_width = resized_width
        self.augment = augment
        self.image_files_X = sorted([f for f in os.listdir(folder_X) if os.path.isfile(os.path.join(folder_X, f))])
        self.image_files_Y = sorted([f for f in os.listdir(folder_Y) if os.path.isfile(os.path.join(folder_Y, f))])
        # Define a transform to convert the images to grayscale, resize, convert to tensor, and normalize
        self.transform = transforms.Compose([
            transforms.Lambda(crop_upper_arch),  # Crop
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((resized_height, resized_width), interpolation=transforms.InterpolationMode.NEAREST),  # Resize the image
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    def __len__(self):
        return len(self.image_files_X)

    def __getitem__(self, idx):
        img_name_X = os.path.join(self.folder_X, self.image_files_X[idx])
        img_name_Y = os.path.join(self.folder_Y, self.image_files_Y[idx])

        image_X = Image.open(img_name_X)
        image_Y = Image.open(img_name_Y)

        image_X = self.transform(image_X) / 255.0
        image_Y = self.transform(image_Y)

        if self.augment:
            aug_X, aug_Y = self.augment_data(image_X, image_Y)
            return aug_X, aug_Y
        else:
            return image_X, image_Y

    def augment_data(self, x, y):
        if isinstance(x, torch.Tensor):
            x = np.ascontiguousarray(x.permute(1, 2, 0).cpu().numpy())
        if isinstance(y, torch.Tensor):
            y = np.ascontiguousarray(y.permute(1, 2, 0).cpu().numpy())

        aug1 = A.HorizontalFlip(p=0.5)
        aug2 = A.Rotate(limit=15, p=0.5)

        # Apply first augmentation
        augmented = aug1(image=x, mask=y)
        image_X_aug1 = torch.from_numpy(augmented["image"]).permute(2, 0, 1)
        image_Y_aug1 = torch.from_numpy(augmented["mask"]).permute(2, 0, 1)

        # Apply second augmentation to the output of the first
        augmented = aug2(image=image_X_aug1.permute(1, 2, 0).cpu().numpy(),
                        mask=image_Y_aug1.permute(1, 2, 0).cpu().numpy())
        image_X_aug2 = torch.from_numpy(augmented["image"]).permute(2, 0, 1)
        image_Y_aug2 = torch.from_numpy(augmented["mask"]).permute(2, 0, 1)

        # Add the doubly augmented pair to the list
        return image_X_aug2, image_Y_aug2

# Example usage
train_dataset = CustomImageDataset(folder_X='/content/drive/MyDrive/Demarcation Line/train_x', folder_Y='/content/drive/MyDrive/Demarcation Line/train_y')
val_dataset = CustomImageDataset(folder_X='/content/drive/MyDrive/Demarcation Line/validation_x', folder_Y='/content/drive/MyDrive/Demarcation Line/validation_y')
test_dataset = CustomImageDataset(folder_X='/content/drive/MyDrive/Demarcation Line/test_x', folder_Y='/content/drive/MyDrive/Demarcation Line/test_y')

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
