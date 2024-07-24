import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
import csv
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data from pickle files
def load_data(directory, target_length=61500, n_mels=64, sr=8000, hop_length=512, n_fft=1024):
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                utterance, valence = data["audio_data"], data["valence"]

                # Ensure the utterance is the target length
                if len(utterance) < target_length:
                    # Pad with zeros
                    utterance = np.pad(utterance, (0, target_length - len(utterance)), 'constant')
                else:
                    # Trim the utterance
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
                labels.append(valence)
    return np.array(features), np.array(labels)

train_features, train_labels = load_data('train')
test_features, test_labels = load_data('test')

print(f"Train feature shape: {train_features.shape}")
print(f"Test feature shape: {test_features.shape}")

# Dataset and DataLoader
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) 
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = AudioDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = AudioDataset(test_features, test_labels)
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
    sample_input = torch.randn((1, 1, 64, 121)).to(device)  # Adjust the dimensions based on your input shape
    with torch.no_grad():
        sample_output = model.conv1(sample_input)
        sample_output = model.conv2(sample_output)
        sample_output = model.conv3(sample_output)
        sample_output = model.conv4(sample_output)
        flattened_size = sample_output.view(1, -1).size(1)
        model.set_linear_layer(flattened_size)
    return model

# Training and validation function
def train_model(train_loader, val_loader, activation='relu', kernel_size=3, pooling='max', lr=0.0001, num_epochs=50):
    model = initialize_model(activation, kernel_size, pooling)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    best_val_loss = float('inf')
    best_epoch = -1

    train_losses_over_epochs = []
    val_losses_over_epochs = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        train_losses_over_epochs.append(train_loss)
        print(f"Epoch {epoch+1} - Training Loss: {train_loss:.4f}")

        model.eval()
        val_losses = []
        val_predictions = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_predictions.extend(outputs.cpu().numpy().tolist())

        val_loss = np.mean(val_losses)
        val_losses_over_epochs.append(val_loss)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation performance so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')

        # Save validation predictions with corresponding .pkl file names
        save_validation_predictions(val_predictions, epoch+1)

        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses_over_epochs, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses_over_epochs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.show()

def save_validation_predictions(predictions, epoch):
    directory = 'test' 
    prediction_file = f'val_predictions_epoch_{epoch}.csv'
    with open(prediction_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'predicted_valence'])
        for file_name, prediction in zip(sorted(os.listdir(directory)), predictions):
            if file_name.endswith('.pkl'):
                writer.writerow([file_name, prediction])

train_model(train_loader, test_loader, activation='relu', kernel_size=5, pooling='avg', lr=0.0001, num_epochs=50)
