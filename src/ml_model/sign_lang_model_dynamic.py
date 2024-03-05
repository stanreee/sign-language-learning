import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### IDS:
## 0: no, 1: where, 2: future, 9: j, 25: z

# class SignLangModelDynamic(nn.Module):
#     def __init__(self, num_hands, num_neurons, num_layers, name=""):
#         super(SignLangModelDynamic, self).__init__()
#         self.num_hands = num_hands
#         self.num_neurons = num_neurons
#         self.num_layers = num_layers
#         self.name = name

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.fc1 = nn.Linear(560, 26)

#         self.soft = nn.LogSoftmax(dim=1)
    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.soft(self.fc1(x))

#         return x

class SignLangModelDynamic(nn.Module):
    def __init__(self, num_hands, num_neurons, num_layers, name=""):
        super().__init__()
        self.name = name
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        dim = 18 if num_hands == 1 else 10
        input_dim = 30 * 63 * num_hands
        
        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(in_features=input_dim, out_features=num_neurons)

        for i in range(self.num_layers):
            self.layers[f"hidden_{i}"] = nn.Linear(in_features=num_neurons, out_features=num_neurons)
        
        self.layers["output"] = nn.Linear(in_features=num_neurons, out_features=26)

        self.relu = nn.ReLU()
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layers["input"](x)

        for i in range(self.num_layers):
            x = self.relu(self.layers[f"hidden_{i}"](x))

        return self.soft(self.layers["output"](x))