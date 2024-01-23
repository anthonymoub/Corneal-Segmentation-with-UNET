import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import wandb  
from model import UNET  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNET().to(device)

criterion_weighted = nn.BCELoss(weight=torch.tensor(40).to(device))

# Learning rate
lr = 0.0001

# Choose an optimizer - Adam.
optimizer = optim.Adam(model.parameters(), lr=lr)

# Number of epochs
num_epochs = 200

# Batch size
batch_size = 4

# Decide the number of samples to train on - TODO: check size of total data
num_samples_to_test = 500
train_subset, _ = random_split(train_dataset, [num_samples_to_test, len(train_dataset) - num_samples_to_test])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

best_val_loss = float('inf')
best_model_path = 'best_model.pth'

# WanDB Configuration
config = wandb.config
config.learning_rate = 0.00001
config.epochs = num_epochs
config.batch_size = batch_size

# Declare directory to export model
drive_path = '/content/drive/My Drive/Demarcation Line/Model Checkpoints/'
os.makedirs(drive_path, exist_ok=True)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode.
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = torch.sigmoid(model(inputs))  # add sigmoid
        targets = (targets >= 0.5).float()
        loss = criterion_weighted(outputs, targets)
        wandb.log({"Train Loss": loss.item()})
        loss.backward()
        optimizer.step()

    model.eval()  # Set model to evaluation mode.
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            inputs, targets = val_batch[0].to(device), val_batch[1].to(device)
            outputs = torch.sigmoid(model(inputs))
            loss = criterion_weighted(outputs, targets)
            val_loss += loss.item()
            wandb.log({"Val Loss": val_loss / len(val_loader)})

    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}')

    # Check if the current validation loss is the best one
    if val_loss < best_val_loss:
        print(f'Saving new best model at epoch {epoch + 1} with Val Loss: {val_loss}')
        best_val_loss = val_loss  # Update best validation loss

        # Save the model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': best_val_loss
        }

        torch.save(checkpoint, 'best_model_checkpoint.pth')

