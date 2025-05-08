import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from poker_bot.hand_strength_estimator.model import PokerMLP, PokerDataset, BoundedMaxErrorLoss

# Paths to data files
train_file = "data/all_easier_poker_hand_exact_strength.pkl"
model_save_path = "models/poker_mlp.pth"

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare the data
full_train_dataset = PokerDataset(train_file, target_key="hand_strength")
train_size = 100000
validation_size = len(full_train_dataset) - train_size

# Randomly split the training data
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

# Move entire datasets to GPU
X_train, y_train = zip(*[sample for sample in tqdm(train_dataset, desc="Loading Training Data")])
X_train = torch.stack(X_train).to(device)
y_train = torch.stack(y_train).to(device)

X_val, y_val = zip(*[sample for sample in tqdm(val_dataset, desc="Loading Validation Data")])
X_val = torch.stack(X_val).to(device)
y_val = torch.stack(y_val).to(device)

# Initialize the model, optimizer, and loss function
model = PokerMLP(dropout_rate=0.15).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
criterion = BoundedMaxErrorLoss(alpha=0.95, delta=1.0).to(device)

# Training loop
epochs = 1000
batch_size = 8192

for epoch in range(epochs):

    if epoch > 500:
        criterion.alpha = 0.99

    # Shuffle training data
    indices = torch.randperm(X_train.size(0), device=device)
    total_train_loss = 0
    total_train_mse = 0
    all_train_errors = []

    # Training phase
    model.train()
    for i in range(0, X_train.size(0), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate total loss and MSE
        total_train_loss += loss.item() * X_batch.size(0)
        total_train_mse += mean_squared_error(y_batch.detach().cpu().numpy(), predictions.detach().cpu().numpy()) * X_batch.size(0)

        # Track maximum training error for this batch
        all_train_errors.append(abs(y_batch - predictions).max().item())

    avg_train_loss = total_train_loss / X_train.size(0)
    avg_train_mse = total_train_mse / X_train.size(0)
    max_train_error = max(all_train_errors)

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_val_mse = 0
    all_val_errors = []

    with torch.no_grad():
        for i in range(0, X_val.size(0), batch_size):
            X_batch = X_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]

            predictions = model(X_batch)

            # Accumulate total loss and MSE
            batch_loss = criterion(predictions, y_batch).item()
            total_val_loss += batch_loss * X_batch.size(0)
            total_val_mse += mean_squared_error(y_batch.detach().cpu().numpy(), predictions.detach().cpu().numpy()) * X_batch.size(0)

            # Track maximum validation error for this batch
            all_val_errors.append(abs(y_batch - predictions).max().item())

    avg_val_loss = total_val_loss / X_val.size(0)
    avg_val_mse = total_val_mse / X_val.size(0)
    max_val_error = max(all_val_errors)

    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Training Loss: {avg_train_loss:.6f}, Training MSE: {avg_train_mse:.6f}, Max Training Error: {max_train_error:.6f}, "
          f"Validation Loss: {avg_val_loss:.6f}, Validation MSE: {avg_val_mse:.6f}, Max Validation Error: {max_val_error:.6f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
