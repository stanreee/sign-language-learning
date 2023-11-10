import torch
import cv2
import torch.nn as nn
import numpy as np

class SignLangModel:
    def __init__(self) -> None:
        self.model = torch.hub.load( \
                      'ultralytics/yolov5', \
                      'yolov5s', \
                      pretrained=True)
        # self.model = nn.Sequential()
        # # 1st convolutional layer
        # self.model.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
        # # activation for 1st convolutional layer
        # self.model.add_module("relu1", nn.ReLU())
        # # max pooling layer 1
        # self.model.add_module("pool1", nn.MaxPool2d(kernel_size=2))

        # # 2nd convolutional layer (in_channels for 2nd layer = out channels for 1st layer)
        # self.model.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
        # # activation for 2nd conv. layer
        # self.model.add_module("relu2", nn.ReLU())
        # # max pooling layer 2
        # self.model.add_module("pool2", nn.MaxPool2d(kernel_size=2))

        # # not sure what the below does, will look into the specifics of a convolutional neural network to gain a better understanding

        # # flatten layer
        # self.model.add_module("flatten", nn.Flatten())

        # # fully connected layer
        # self.model.add_module('fc1', nn.Linear(3136, 1024))
        # # FC layer's activation
        # self.model.add_module("relu3", nn.ReLU())

        # # dropout
        # self.model.add_module("dropout", nn.Dropout(p=0.5))

        # # fully connected layer 2
        # self.model.add_module('fc2', nn.Linear(1024, 24))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def score_frame(self, frame):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        results = self.model(image)
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        return labels, cord
    
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2: 
                continue
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
            cv2.rectangle(frame, \
                        (x1, y1), (x2, y2), \
                        bgr, 2) #Plot the boxes
            cv2.putText(frame,\
                        classes[labels[i]], \
                        (x1, y1), \
                        label_font, 0.9, bgr, 2) #Put a label over box.
            return frame