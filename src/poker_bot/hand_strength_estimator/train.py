import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from model import PokerMLP, PokerDataset, BoundedMaxErrorLoss

# Paths to data files
train_file = "data/poker_hand_estimated_strength_1M_0.pkl"
test_file = "data/poker_hand_exact_strength_25000.pkl"
model_save_path = "models/poker_mlp.pth"

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare the data
full_train_dataset = PokerDataset(train_file, target_key="estimated_hand_strength")
validation_size = 25000
train_size = len(full_train_dataset) - validation_size

# Randomly split the training data
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

test_dataset = PokerDataset(test_file, target_key="hand_strength")

# Move entire datasets to GPU
X_train, y_train = zip(*[sample for sample in tqdm(train_dataset, desc="Loading Training Data")])
X_train = torch.stack(X_train).to(device)
y_train = torch.stack(y_train).to(device)

X_val, y_val = zip(*[sample for sample in tqdm(val_dataset, desc="Loading Validation Data")])
X_val = torch.stack(X_val).to(device)
y_val = torch.stack(y_val).to(device)

X_test, y_test = zip(*[sample for sample in tqdm(test_dataset, desc="Loading Test Data")])
X_test = torch.stack(X_test).to(device)
y_test = torch.stack(y_test).to(device)

# Initialize the model, optimizer, and loss function
model = PokerMLP(dropout_rate=0.0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=100 * (X_train.size(0) // 32768), pct_start=0.1, anneal_strategy='linear', final_div_factor=10)
criterion = BoundedMaxErrorLoss(alpha=0.95, delta=1.0).to(device)

# Training loop
epochs = 1000
batch_size = 131072

for epoch in range(epochs):
    # Shuffle training data
    indices = torch.randperm(X_train.size(0), device=device)
    total_train_loss = 0

    # Training phase
    model.train()
    all_train_errors = []
    for i in range(0, X_train.size(0), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        # Track maximum training error for this batch
        all_train_errors.append(abs(y_batch - predictions).max().item())

    avg_train_loss = total_train_loss / (X_train.size(0) // batch_size)
    avg_train_error = np.mean(all_train_errors)
    max_train_error = max(all_train_errors)

    # Validation phase
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)
        val_loss = criterion(predictions, y_val).item()
        val_mse = mean_squared_error(y_val.cpu().numpy(), predictions.cpu().numpy())
        val_max_error = abs(y_val - predictions).max().item()

    # Testing phase
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test).item()
        test_mse = mean_squared_error(y_test.cpu().numpy(), predictions.cpu().numpy())
        test_max_error = abs(y_test - predictions).max().item()

    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, Max Training Error: {max_train_error:.6f}, Validation Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}, Validation MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}, Validation Max Error: {val_max_error:.6f}, Test Max Error: {test_max_error:.6f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
