import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import wandb  
import matplotlib.pyplot as plt  
from model import UNET 
from data import train_dataset , train_loader, test_dataset, test_loader

# Importing a model version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET().to(device)
drive_path = '/content/drive/MyDrive/Demarcation Line/Model Checkpoints/'
checkpoint_filename = os.path.join(drive_path, f'best_model_checkpoint_weight30.pth')
checkpoint = torch.load(checkpoint_filename)
model.load_state_dict(checkpoint['state_dict'])

# Loss
criterion_weighted = nn.BCELoss(weight=torch.tensor(20).to(device))

opt_LR = 0.02

# Initialize optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=opt_LR)
optimizer_sgd = optim.SGD(model.parameters(), lr=opt_LR, momentum=0.9)
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=opt_LR)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=opt_LR, alpha=0.99,
                                  momentum=0.9,
                                  eps=1e-8,
                                  weight_decay=0)
optimizer_adamw = optim.AdamW(model.parameters(), lr=opt_LR, betas=(0.9, 0.999),
                              eps=1e-8,
                              weight_decay=0.01,
                              amsgrad=False)

# Start with Adam optimizer
optimizer = optimizer_adam
current_optimizer = "Adam"
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
lr_adjust_count = 0
patience = 5
patience_counter = 0
start_epoch = 0  # Assuming starting from the beginning, update as needed
num_epochs = 200  # Set the number of epochs
batch_size = 5  # batch size

# Use a subset for quick testing
num_samples_to_test = 622
train_subset, _ = random_split(train_dataset, [num_samples_to_test, len(train_dataset) - num_samples_to_test])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(start_epoch, num_epochs):
    model.train()  # Set model to training mode.
    # Training steps
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = torch.sigmoid(model(inputs)).to(device)  # Add sigmoid extra
        targets = (targets >= 0.5).float()  # TODO - note and comment this
        loss = criterion_weighted(outputs, targets)
        wandb.log({"Train Loss": loss.item()})
        loss.backward()
        optimizer.step()

    model.eval()  # Set model to evaluation mode.
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            inputs, targets = val_batch[0].to(device), val_batch[1].to(device)
            outputs = torch.sigmoid(model(inputs)).to(device)
            loss = criterion_weighted(outputs, targets)
            val_loss += loss.item()
            wandb.log({"Val Loss": val_loss / len(val_loader)})

    # Validation steps...
    val_loss /= len(val_loader)

    # Print losses every 2 epochs
    if (epoch + 1) % 1 == 0:  # Every 2 epochs
        print(f'Epoch {epoch+1}: Train Loss: {loss.item() / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # Learning Rate Scheduling and Optimizer Switching
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        lr_adjust_count = 0
        patience_counter = 0
        # Checkpoint Saving Logic
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': best_val_loss
        }
        checkpoint_path = os.path.join(drive_path, f'best_model_checkpoint_w20.pth')
        # Save the model checkpoint to the selected path
        torch.save(checkpoint, checkpoint_path)
        print(f'Saving new best model at epoch {epoch+1} with Val Loss: {val_loss:.4f}')
    else:
        patience_counter += 1
        # Reduce LR if val loss does not decrease for 2 epochs

        if patience_counter > 2:
            if lr_adjust_count < 1:  # First, try reducing LR
                scheduler.step(val_loss)
                lr_adjust_count += 1
            else:
                if current_optimizer == "Adam":
                    optimizer = optimizer_sgd
                    current_optimizer = "SGD"
                elif current_optimizer == "SGD":
                    optimizer = optimizer_adagrad
                    current_optimizer = "Adagrad"
                elif current_optimizer == "Adagrad":
                    optimizer = optimizer_rmsprop
                    current_optimizer = "RMSprop"
                elif current_optimizer == "RMSprop":
                    optimizer = optimizer_adamw
                    current_optimizer = "AdamW"
                else:  # Default back to Adam if the cycle completes
                    optimizer = optimizer_adam
                    current_optimizer = "Adam"

                # Reset counters
                lr_adjust_count = 0
                patience_counter = 0
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
                print(f"Switching to {current_optimizer} optimizer at Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        else:
            scheduler.step(val_loss)
