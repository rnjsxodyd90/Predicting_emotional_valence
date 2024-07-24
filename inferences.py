import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
import csv

# Hyperparameters
ACTIVATION = 'relu'  
KERNEL_SIZE = 5  
POOLING = 'avg'  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test data from pickle files
def load_test_data(directory, target_length=61500, n_mels=64, sr=8000, hop_length=512, n_fft=1024):
    features = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                utterance = data["audio_data"]

                # Ensure the utterance is the target length
                if len(utterance) < target_length:
                    utterance = np.pad(utterance, (0, target_length - len(utterance)), 'constant')
                else:
                    utterance = utterance[:target_length]

                # Convert numpy array to torch tensor
                waveform = torch.tensor(utterance).float().unsqueeze(0)

                # Create Mel spectrogram
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length
                )(waveform)

                features.append(mel_spectrogram.squeeze(0).numpy()) 
                file_names.append(file_name)
    return np.array(features), file_names

test_features, test_file_names = load_test_data('test2')
print(f"Test feature shape: {test_features.shape}")

# Dataset and DataLoader
class AudioDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

test_dataset = AudioDataset(test_features)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CNNNetwork(nn.Module):
    def __init__(self, activation='relu', kernel_size=3, pooling='max'):
        super(CNNNetwork, self).__init__()
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.pooling = nn.MaxPool2d(kernel_size=2) if pooling == 'max' else nn.AvgPool2d(kernel_size=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            self.activation,
            self.pooling
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            self.activation,
            self.pooling
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            self.activation,
            self.pooling
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            self.activation,
            self.pooling
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 1)  # This will be adjusted dynamically

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x.squeeze()

    def set_linear_layer(self, input_size):
        self.linear = nn.Linear(input_size, 1)


def initialize_model(activation='relu', kernel_size=3, pooling='max'):
    model = CNNNetwork(activation, kernel_size, pooling).to(device)
    sample_input = torch.randn((1, 1, 64, 121)).to(device)
    with torch.no_grad():
        sample_output = model.conv1(sample_input)
        sample_output = model.conv2(sample_output)
        sample_output = model.conv3(sample_output)
        sample_output = model.conv4(sample_output)
        flattened_size = sample_output.view(1, -1).size(1)
        model.set_linear_layer(flattened_size)
    return model

model = initialize_model(ACTIVATION, KERNEL_SIZE, POOLING)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference function
def run_inference(test_loader, model):
    model.to(device)
    predictions = []

    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Running Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().tolist())

    return predictions

test_predictions = run_inference(test_loader, model)

# Save test predictions to a CSV file
def save_test_predictions(predictions, file_names):
    prediction_file = 'test_predictions.csv'
    with open(prediction_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Label'])
        for file_name, prediction in zip(file_names, predictions):
            writer.writerow([file_name, prediction])

save_test_predictions(test_predictions, test_file_names)

print("Inference completed and predictions saved to 'test_predictions.csv'.")
