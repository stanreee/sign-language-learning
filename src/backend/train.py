import os
import pandas as pd
from sign_lang_model import SignLangModel
from sign_lang_model_dynamic import SignLangModelDynamic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time

"""
    idea of training:

    for each model, we will vary the number of layers and neruons in each layer and choose the best combination of the two that results in the
    best accuracy on the test dataset
"""

# BATCH_SIZE = 2
NUM_EPOCHS = 48

# create feature loaders on specified training data sets
def get_features_loader(TRAINING_DATA_PATH, BATCH_SIZE):
    """
        Creates train and test data loaders for the specified training data path.

        Parameters:
        TRAINING_DATA_PATH (str): File path of the training data.
        BATCH_SIZE (int): Size of individual batches for each data loader.

        Returns:
        Tuple of the training data loader and the test data loader.
    """
    data = pd.read_csv(TRAINING_DATA_PATH)

    train, test = train_test_split(data, test_size=0.2)

    target = train[train.columns[0]].values
    features = train.drop(train.columns[0], axis=1).values

    ytrain = torch.from_numpy(target).float()
    ytrain = ytrain.type(torch.LongTensor)
    xtrain = torch.from_numpy(features).float()

    target = test[test.columns[0]].values
    features = test.drop(test.columns[0], axis=1).values

    ytest = torch.from_numpy(target).float()
    ytest = ytest.type(torch.LongTensor)
    xtest = torch.from_numpy(features).float()

    dataset = TensorDataset(xtrain, ytrain)
    test_dataset = TensorDataset(xtest, ytest)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(len(train_loader), len(test_loader))
    return (train_loader, test_loader)

def collect_features_loaders(data_path, num_hands, model_name, model, num_neuron_range, batch_size):
    """
        Compiles data in the data path specified into data loaders and creates models with varying number of layers and neurons.

        Parameters:
        data_path (str): File path of the training data.
        num_hands (int): Number of hands for this model.
        model_name (str): Name of model.
        model (SignLangModel or SignLangModelDynamic class): Specific neural network model to be used.
        num_neuron_range (array): Array of different number of neurons.
        batch_size (int): Size of batches.
        
        Returns:
            Tuple consisting of
                - train and test data loaders
                - a list of models with varying number of neurons per layer and number of layers
                - name of model
    """
    train_loader, test_loader = get_features_loader(data_path, batch_size)
    models = []
    num_layers = np.arange(1, 3)
    num_neurons = num_neuron_range

    for i in range(len(num_neurons)):
        for j in range(len(num_layers)):
            m = model(num_hands, num_neurons[i], num_layers[j], model_name)
            models.append(m)
    return (train_loader, test_loader, models, model_name)

def train(train_loader, test_loader, model, lr=0.0001, num_epochs=NUM_EPOCHS):
    """
        Trains a specific model.

        Parameters:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        model (instance of a SignLangModel or SignLangModelDynamic object): Model to train.
        lr (float): Learning rate for the training process.
        num_epochs (int): Number of rounds of training.

        Returns:
        Tuple consisting of accuracy on training and test datasets.
    """
    train_acc, test_acc = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(42)
    model.eval()

    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            pred = torch.argmax(outputs, axis=1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # if i == len(train_loader) - 1:
            #     print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_acc.append(
            100 * torch.mean((pred == targets).float()).item()
        )

        # print(len(train_loader), len(test_loader))

        data, targets = next(iter(test_loader))
        pred = torch.argmax(model(data), axis=1)
        test_acc.append(
            100 * torch.mean((pred == targets).float()).item()
        )

    return train_acc[-1], test_acc[-1]

cur_dir = os.getcwd()

print("Creating feature loaders...")

features_loaders = []

# features_loaders.append(collect_features_loaders(
#     cur_dir + "/gather/datasets/static.csv",
#     1,
#     "static_one_hand",
#     SignLangModel,
#     np.arange(128, 144, 8),
#     32
# ))

# features_loaders.append(collect_features_loaders(
#     cur_dir + "/gather/datasets/dynamic.csv",
#     1,
#     "dynamic_one_hand",
#     SignLangModelDynamic,
#     np.arange(320, 384, 32),
#     2
# ))

### UNCOMMENT BOTH OF THESE FOR TWO HAND DYNAMIC

features_loaders.append(collect_features_loaders(
    cur_dir + "/gather/datasets/dynamic_two_1.csv",
    1,
    "dynamic_two_1",
    SignLangModelDynamic,
    np.arange(320, 384, 32),
    2
))

features_loaders.append(collect_features_loaders(
    cur_dir + "/gather/datasets/dynamic_two_2.csv",
    1,
    "dynamic_two_2",
    SignLangModelDynamic,
    np.arange(320, 384, 32),
    2
))

print("Feature loaders created. Training models...")
for idx, object in enumerate(features_loaders):
    train_loader, test_loader, models, model_name = object
    train_accuracies = []
    test_accuracies = []
    cur_max_accuracy = -1
    model_to_save = None
    accuracies = None

    for model in models:
        start_time = time.time()
        print(f"Training model '{model_name}' with {model.num_layers} layers and {model.num_neurons} neurons...")
        train_acc, test_acc = train(train_loader, test_loader, model)
        train_accuracies.append({
            "num_layers": model.num_layers,
            "num_neurons": model.num_neurons,
            "accuracy": train_acc
        })
        test_accuracies.append({
            "num_layers": model.num_layers,
            "num_neurons": model.num_neurons,
            "accuracy": test_acc
        })
        if test_acc > cur_max_accuracy:
            model_to_save = model
            accuracies = (train_acc, test_acc)
        elapsed = time.time() - start_time
        print("Model trained, took", elapsed, "seconds. Test accuracy of this train:", test_acc)

    print("train accuracies:")
    print(train_accuracies)

    print("test accuracies:")
    print(test_accuracies)
    
    if model_to_save:
        torch.save(model_to_save, cur_dir + "/trained_models/" + str(model.name) + ".pt")
        print(f"Saving model {model_to_save.name} with {model.num_layers} layers and {model.num_neurons} neurons with train and test accuracies of {accuracies}.")