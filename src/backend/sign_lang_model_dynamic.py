import torch.nn as nn
import torch.nn.functional as F

class SignLangModelDynamic(nn.Module):
    def __init__(self, num_hands, num_neurons, num_layers, name=""):
        """
            Creates a sign language neural network model for dynamic signs.

            Parameters:
            num_hands (int): Number of hands for this model.
            num_neurons (int): Number of neurons per hidden layer.
            num_layers (int): Number of hidden layers.
            name (string): Name of model.
        """
        super().__init__()
        self.name = name
        self.num_neurons = num_neurons
        self.num_layers = num_layers

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