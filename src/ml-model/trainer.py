import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset, TensorDataset
import itertools
from sign_lang_model import SignLangModel
import os

# load the data model
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

TRAINING_DATA_PATH = '/Users/stan/Desktop/GitHub/sfwr-eng-year-4/sign-language-learning/src/ml-model/SIGN_LANG_DATASET/sign_mnist_train.csv'
TESTING_DATA_PATH = '/Users/stan/Desktop/GitHub/sfwr-eng-year-4/sign-language-learning/src/ml-model/SIGN_LANG_DATASET/sign_mnist_test.csv'

##
## PROCESSING DATA SETS TO BE TRAINED ON CONVOLUTIONAL NEURAL NETWORK (CNN)
##

training_data = pd.read_csv(TRAINING_DATA_PATH)

testing_data = pd.read_csv(TESTING_DATA_PATH)

def adjust_class_labels(label):
    if label >= 10:
        label -= 1
    return label

# we want to apply adjust_class_labels on all labels because the 9th label index (J) can only be detected using gestures
training_data["label"] = training_data["label"].apply(adjust_class_labels)
testing_data["label"] = testing_data["label"].apply(adjust_class_labels)

# get the labels (1st column)
target = training_data["label"].values
# get the features for that specific label (rest of row, excluding the label column)
features = training_data.drop("label", axis=1).values

# do the same for the testing dataset
target_test = testing_data["label"].values
features_test = testing_data.drop("label", axis=1).values

# reshape data for neural network?
features = features.reshape(-1, 1, 28, 28)
features_test = features_test.reshape(-1, 1, 28, 28)

# rescaling features
features_scaled = features / 255
features_test_scaled = features_test / 255

# convert numpy arrays to Torch tensors (tensors are essentially multidimensional arrays that can be run on the GPU)
# i guess converting to a tensor from a numpy array allows to run on the GPU resulting in this being faster?
y_train = torch.from_numpy(target).float()
x_train = torch.from_numpy(features_scaled).float()

y_test = torch.from_numpy(target_test).float()
x_test = torch.from_numpy(features_test_scaled).float()

# initialize TensorDataset for training and testing
sign_lang_dataset = TensorDataset(x_train, y_train)

testing_dataset = TensorDataset(y_test, x_test)

training_dataset = Subset(
    sign_lang_dataset,
    torch.arange(10000, len(sign_lang_dataset))
)

validation_dataset = Subset(sign_lang_dataset, torch.arange(10000))

# this is a way of visualizing the training dataset
fig = plt.figure(figsize=(15,6))
# iterate through the data in the training set
for i, (data, label) in itertools.islice(enumerate(training_dataset), 10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # data is a multidimensional array that represents the darkness of the pixel
    ax.imshow(data.numpy().reshape(28, 28), cmap='gray_r')
    ax.set_title(f'True label = {int(label)}', size=15)
plt.suptitle("Training dataset examples", fontsize=20)
plt.tight_layout()
plt.show()

# creating data loaders for the convolutional NN

torch.manual_seed(42)

batch_size = 64

training_dataloader = DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=batch_size,
    shuffle=False
)

##
## BUILDING THE CNN WITH THE PROCESSED DATA SETS
##

model = SignLangModel()
classifier = model.model

classifier.eval()

# training the network

def train(
        model,
        loss_func,
        optimizer,
        training_dataloader,
        validation_dataloader,
        epochs=10,
        enable_logging=False,
):
    loss_history_train = [0] * epochs
    accuracy_history_train = [0] * epochs
    loss_history_valid = [0] * epochs
    accuracy_history_valid = [0] * epochs

    for epoch in range(epochs):
        # training mode
        model.train()

        for x_batch, y_batch in training_dataloader:
            # generate the predictions
            model_predictions = model(x_batch)

            # compute the losses
            loss = loss_func(model_predictions, y_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(model_predictions, dim=1) == y_batch).float()
            accuracy_history_train[epoch] += is_correct.sum().cpu()
        
        loss_history_train[epoch] /= len(training_dataloader.dataset)
        accuracy_history_train[epoch] /= len(training_dataloader.dataset)

        # evaluation mode
        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in validation_dataloader:
                # make predictions on the x_batch, compare it with the y_batch?
                model_predictions = model(x_batch)

                loss = loss_func(model_predictions, y_batch.long())

                loss_history_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(model_predictions, dim=1) == y_batch).float()
                accuracy_history_valid[epoch] += is_correct.sum().cpu()
        
        loss_history_valid[epoch] /= len(validation_dataloader.dataset)
        accuracy_history_valid[epoch] /= len(validation_dataloader.dataset)

        # Logging the training process
        if enable_logging:
            print(
                "Epoch {}/{}\n"
                "train_loss = {:.4f}, train_accuracy = {:.4f} | "
                "valid_loss = {:.4f}, valid_accuracy = {:.4f}".format(
                epoch + 1,
                epochs,
                loss_history_train[epoch], 
                accuracy_history_train[epoch],
                loss_history_valid[epoch],
                accuracy_history_valid[epoch],
                )
            )
        
    return (
        model, 
        loss_history_train, 
        accuracy_history_train,
        loss_history_valid,
        accuracy_history_valid,
    )

# using this as the loss function
loss_func = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

torch.manual_seed(42)

(
    classifier,
    loss_history_train,
    accuracy_history_train,
    loss_history_valid,
    accuracy_history_valid
) = train(
    model=classifier,
    loss_func=loss_func,
    optimizer=optimizer,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    epochs=8,
    enable_logging=True,
)

torch.save(classifier.state_dict(), "/Users/stan/Desktop/GitHub/sfwr-eng-year-4/sign-language-learning/src/ml-model/model/model.pth")
