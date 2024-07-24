import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

# Check for CUDA availability
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

                # Convert numpy array to torch tensor and ensure it is 2D (1 x time)
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

# Custom estimator to integrate with sklearn
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_epochs=50, batch_size=4, lr=0.001, activation='relu', kernel_size=3, pooling='max'):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.kernel_size = kernel_size
        self.pooling = pooling

    def fit(self, X, y):
        self.model = initialize_model(self.activation, self.kernel_size, self.pooling)
        self.model.to(device)
        dataset = AudioDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            train_losses = []
            for inputs, labels in tqdm(loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

        return self

    def predict(self, X):
        self.model.eval()
        dataset = AudioDataset(X, np.zeros(X.shape[0]))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                if outputs.dim() == 0:
                    predictions.append(outputs.cpu().item()) 
                else:
                    predictions.extend(outputs.cpu().numpy().tolist())
        return np.array(predictions)

# Grid search
param_grid = {
    'num_epochs': [10, 20, 50],
    'batch_size': [2, 4, 8, 16],
    'lr': [0.001, 0.0005, 0.0001, 0.00005, 0.00001],
    'activation': ['relu', 'sigmoid'],
    'kernel_size': [3, 5],
    'pooling': ['max', 'avg']
}

regressor = PyTorchRegressor()
grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(train_features, train_labels)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", -grid_search.best_score_)
