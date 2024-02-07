import torch
import cv2
import torch.nn as nn
import numpy as np

### IDS:
## 0: no, 1: where, 2: future, 9: j, 25: z

class SignLangModelDynamic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18 * 42, 42)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(42, 30)
        self.fc3 = nn.Linear(30, 26)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x =  self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.soft(x)
        return x