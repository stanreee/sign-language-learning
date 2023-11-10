import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from sign_lang_model import SignLangModel
import os

CONFIDENCE = 0.5

# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
slm = SignLangModel()
# slm.load_model("/Users/stan/Desktop/GitHub/sfwr-eng-year-4/sign-language-learning/src/ml-model/model/model.pth")

onnx_model_path = "/Users/stan/Desktop/GitHub/sfwr-eng-year-4/sign-language-learning/src/ml-model/model/onnx"
onnx_model_name = "model.onnx"

os.makedirs(onnx_model_path, exist_ok=True)
full_model_path = os.path.join(onnx_model_path, onnx_model_name)

# generate model input
generated_input = Variable(
    torch.randn(1, 3, 224, 224)
)
# model export into ONNX format
torch.onnx.export(
    slm.model,
    generated_input,
    full_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("cannot open camera")
    exit()

width = 400
height = 400

slm.model.eval()

while True:
    # get the frames
    ret, frame = cap.read()
    copy = frame.copy()

    results = slm.score_frame(frame)
    frame = slm.plot_boxes(results, frame)
    
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()