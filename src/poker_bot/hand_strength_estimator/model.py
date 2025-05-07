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

# Define the MLP model with regularization
class PokerMLP(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(PokerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(102, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
    
def load_model(model_path):
    model = PokerMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
