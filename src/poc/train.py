import os
import pandas as pd
from sign_lang_model import SignLangModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

cur_dir = os.getcwd()

TRAINING_DATA_PATH = cur_dir + "/src/poc/datasets/train/csv/data.csv"
NUM_EPOCHS = 48

training_data = pd.read_csv(TRAINING_DATA_PATH)

target = training_data[training_data.columns[0]].values
features = training_data.drop(training_data.columns[0], axis=1).values

ytrain = torch.from_numpy(target).float()
ytrain = ytrain.type(torch.LongTensor)
xtrain = torch.from_numpy(features).float()

dataset = TensorDataset(xtrain, ytrain)

features_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# dataiter = iter(features_loader)
# landmarks, labels = next(dataiter)

model = SignLangModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
torch.save(model.state_dict(), cur_dir + '/src/poc/simple_classifier.pth')


