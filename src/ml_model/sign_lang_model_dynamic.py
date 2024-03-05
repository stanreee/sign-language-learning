import torch
import cv2
import torch.nn as nn
import numpy as np

### IDS:
## 0: no, 1: where, 2: future, 9: j, 25: z

class SignLangModelDynamic(nn.Module):
    def __init__(self, num_hands, num_neurons, num_layers, name=""):
        super(SignLangModelDynamic, self).__init__()
        self.num_hands = num_hands
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.name = name

        # 3D Convolutional Layers
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)  # Adjust output size based on your task
    
    def forward(self, x):
        # Input: (batch_size, in_channels, frames, landmarks, coordinates)
        # Example input shape: (batch_size, 3, 30, 21, 3)

        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Reshape for fully connected layers
        x = x.view(-1, 64 * 4 * 4 * 4)

        # Fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        return x
        return x

# class SignLangModelDynamic(nn.Module):
#     def __init__(self, num_hands, num_neurons, num_layers, name=""):
#         super().__init__()
#         self.name = name
#         self.num_neurons = num_neurons
#         self.num_layers = num_layers

#         dim = 18 if num_hands == 1 else 10
#         input_dim = dim * 63 * num_hands
        
#         self.layers = nn.ModuleDict()
#         self.layers["input"] = nn.Linear(in_features=input_dim, out_features=num_neurons)

#         for i in range(self.num_layers):
#             self.layers[f"hidden_{i}"] = nn.Linear(in_features=num_neurons, out_features=num_neurons)
        
#         self.layers["output"] = nn.Linear(in_features=num_neurons, out_features=26)

#         self.relu = nn.ReLU()
#         self.soft = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.layers["input"](x)

#         for i in range(self.num_layers):
#             x = self.relu(self.layers[f"hidden_{i}"](x))

#         return self.soft(self.layers["output"](x))