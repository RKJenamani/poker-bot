import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from model import PokerMLP, PokerDataset

# Paths to data files
train_file = "data/poker_hand_estimated_strength_1M.pkl"
test_file = "data/poker_hand_exact_strength_25000.pkl"
model_save_path = "models/poker_mlp.pth"

# Prepare the data
train_dataset = PokerDataset(train_file, target_key="estimated_hand_strength")
test_dataset = PokerDataset(test_file, target_key="hand_strength")

batch_size = 4096
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
model = PokerMLP(dropout_rate=0.2)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
criterion = nn.MSELoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    
    # Testing phase
    model.eval()
    y_true = []
    y_pred = []
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Testing"):
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_test_loss += loss.item()
            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions.numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_mse = mean_squared_error(y_true, y_pred)
    max_error = max(abs(true - pred) for true, pred in zip(y_true, y_pred)).item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, Test MSE: {test_mse:.6f}, Max Error: {max_error:.6f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
