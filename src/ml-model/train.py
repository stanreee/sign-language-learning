import os
import pandas as pd
from sign_lang_model import SignLangModel
from sign_lang_model_dynamic import SignLangModelDynamic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

cur_dir = os.getcwd()

# TRAINING_DATA_PATH = cur_dir + "/datasets/train/csv/data.csv"
TRAINING_DATA_PATH = cur_dir + "/gather/datasets/static.csv"
DYNAMIC_TRAINING_DATA_PATH = cur_dir + "/gather/datasets/dynamic.csv"
NUM_EPOCHS = 48

training_data = pd.read_csv(TRAINING_DATA_PATH)
dynamic_training_data = pd.read_csv(DYNAMIC_TRAINING_DATA_PATH)

target = training_data[training_data.columns[0]].values
features = training_data.drop(training_data.columns[0], axis=1).values

dynamic_target = dynamic_training_data[dynamic_training_data.columns[0]].values
dynamic_features = dynamic_training_data.drop(dynamic_training_data.columns[0], axis=1).values

ytrain = torch.from_numpy(target).float()
ytrain = ytrain.type(torch.LongTensor)
xtrain = torch.from_numpy(features).float()

ytrain_dynamic = torch.from_numpy(dynamic_target).float()
ytrain_dynamic = ytrain_dynamic.type(torch.LongTensor)
xtrain_dynamic = torch.from_numpy(dynamic_features).float()

dataset = TensorDataset(xtrain, ytrain)
datasetDynamic = TensorDataset(xtrain_dynamic, ytrain_dynamic)

features_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
features_loader_dynamic = DataLoader(datasetDynamic, batch_size=4, shuffle=True, drop_last=True)

# dataiter = iter(features_loader)
# landmarks, labels = next(dataiter)

model = SignLangModel()
dynamic_model = SignLangModelDynamic()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer_dynamic = optim.Adam(dynamic_model.parameters(), lr=0.001)

torch.manual_seed(42)

model.eval()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(features_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx == len(features_loader) - 1:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(features_loader)}, Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), cur_dir + '/simple_classifier.pth')

print("STATIC MODEL TRAINED")

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(features_loader_dynamic):
        # Zero the gradients
        optimizer_dynamic.zero_grad()

        # Forward pass
        outputs = dynamic_model(data)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer_dynamic.step()

        if batch_idx == len(features_loader_dynamic) - 1:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(features_loader_dynamic)}, Loss: {loss.item():.4f}')

# Save the trained model
torch.save(dynamic_model.state_dict(), cur_dir + '/simple_dynamic_classifier.pth')

print("DYNAMIC MODEL TRAINED")


