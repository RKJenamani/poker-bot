import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from treys import Card

# Independent one-hot encoding for rank and suit
def independent_one_hot_encode(cards):
    encoded = []
    for card in cards:
        # Extract rank and suit correctly
        rank_int = (card >> 8) & 0xF  # Extract the rank (2-14)
        suit_int = (card >> 12) & 0x3  # Extract the suit (0-3)

        # One-hot encode the rank (13 possible ranks)
        rank_vec = [0] * 13
        rank_vec[rank_int - 2] = 1  # Rank 2 is the smallest (index 0)

        # One-hot encode the suit (4 possible suits)
        suit_vec = [0] * 4
        suit_vec[suit_int] = 1

        # Combine the rank and suit encodings
        encoded.extend(rank_vec + suit_vec)
    
    # Return as a single 1D tensor
    return torch.tensor(encoded, dtype=torch.float32)

# Poker dataset
class PokerDataset(Dataset):
    def __init__(self, data_file, target_key="estimated_hand_strength"):
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)
        self.target_key = target_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        cards = sample["private_cards"] + sample["visible_public_cards"]
        X = independent_one_hot_encode(cards)
        y = torch.tensor([sample[self.target_key]], dtype=torch.float32)
        return X, y

class PokerMLP(nn.Module):
    def __init__(self, dropout_rate=0.15):
        super(PokerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(68, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Use Kaiming initialization (good for ReLU activations)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)
    
def load_model(model_path):
    model = PokerMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

class BoundedMaxErrorLoss(nn.Module):
    def __init__(self, alpha=0.95, delta=1.0):
        super(BoundedMaxErrorLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.huber_loss = nn.HuberLoss(delta=delta)

    def forward(self, predictions, targets):
        # Regular huber loss for general fit
        base_loss = self.huber_loss(predictions, targets)

        # Max error within bounds
        bounded_errors = torch.clamp(predictions - targets, min=-1.0, max=1.0)
        max_error = bounded_errors.abs().max()

        # Combine the two losses
        return self.alpha * max_error + (1 - self.alpha) * base_loss