import os
import pandas as pd
from sign_lang_model import SignLangModel
from sign_lang_model_dynamic import SignLangModelDynamic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_features_loader(TRAINING_DATA_PATH):
    training_data = pd.read_csv(TRAINING_DATA_PATH)

    target = training_data[training_data.columns[0]].values
    features = training_data.drop(training_data.columns[0], axis=1).values

    ytrain = torch.from_numpy(target).float()
    ytrain = ytrain.type(torch.LongTensor)
    xtrain = torch.from_numpy(features).float()

    dataset = TensorDataset(xtrain, ytrain)

    features_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    return features_loader

def zip_feature_loaders(training_data_paths):
    feature_loaders = []
    for object in training_data_paths:
        dataPath, model = object
        feature_loaders.append((get_features_loader(dataPath), model))
    return feature_loaders

cur_dir = os.getcwd()

model = SignLangModel(1, "static_one_hand")
dynamic_model = SignLangModelDynamic(1, "dynamic_one_hand")

TRAINING_PATHS = [
    (cur_dir + "/gather/datasets/static.csv", model),
    (cur_dir + "/gather/datasets/dynamic.csv", dynamic_model),
]
NUM_EPOCHS = 48

features_loaders = zip_feature_loaders(TRAINING_PATHS)

for idx, object in enumerate(features_loaders):
    features_loader, model = object
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    torch.manual_seed(42)
    model.eval()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(features_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx == len(features_loader) - 1:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(features_loader)}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), cur_dir + '/trained_models/' + str(model.name) + '.pth')
    print("Model saved to", cur_dir + "/trained_models/" + str(model.name) + ".pth")


