import os
import shutil
from sklearn.model_selection import train_test_split

# Define the path to your main data folder
data_folder = "/content/drive/MyDrive/DataBank_initial"

# Define the path to store the training and testing data
train_folder = "/content/drive/MyDrive/DataBank/Train"
test_folder = "/content/drive/MyDrive/DataBank/Test"

# Define your split ratio
split_ratio = 0.8

# Iterate through each subfolder in your data folder
for subfolder in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder)
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder}")
        # List all files in the subfolder
        files = os.listdir(subfolder_path)
        # Split the files into training and testing sets
        train_files, test_files = train_test_split(files, test_size=(1 - split_ratio))

        # Create directories to store the split data if they don't exist
        os.makedirs(os.path.join(train_folder, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, subfolder), exist_ok=True)

        # Copy training files to the training folder
        for file in train_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(train_folder, subfolder))
        print(f"Successfully copied {len(train_files)} files to {train_folder}/{subfolder}")

        # Copy testing files to the testing folder
        for file in test_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(test_folder, subfolder))
        print(f"Successfully copied {len(test_files)} files to {test_folder}/{subfolder}")

print("Splitting and copying completed successfully!")
