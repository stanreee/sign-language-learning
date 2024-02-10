import torch
import cv2
import torch.nn as nn
import numpy as np

class SignLangModel(nn.Module):
    def __init__(self, num_hands, name):
        super().__init__()
        self.name = name
        self.fc1 = nn.Linear(42 * num_hands, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 42)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x =  self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x