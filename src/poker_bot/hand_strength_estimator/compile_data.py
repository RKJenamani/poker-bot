

# combine together 18 pickle file into one pickle file

import os
import pickle

file_source = "data/poker_hand_estimated_strength_1M_"

combined_data = []
for i in range(5):
    print(f"Processing file {i}...")
    file_path = f"{file_source}{i}.pkl"
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    combined_data.extend(data)

# Save the combined dataset to a new pickle file
output_file = "data/poker_hand_estimated_strength_5M.pkl"
with open(output_file, "wb") as f:
    pickle.dump(combined_data, f)
print(f"Combined dataset saved to {output_file}. Total samples: {len(combined_data)}")