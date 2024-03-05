#Import necessary libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from static_model import StaticModel
from dynamic_model import DynamicModel
from util import process_features, get_features, normalize_landmark_history, landmark_history_preprocess
import torch
import cv2
import numpy as np
import base64
import os
import mediapipe as mp
import json
#Initialize the Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

static = StaticModel()
dynamic_model = DynamicModel()

max_frames = 300
cur_frames = 0

cur_dir = os.getcwd()
static.load_state_dict(torch.load(cur_dir + "/simple_classifier.pth"))
dynamic_model.load_state_dict(torch.load(cur_dir + "/simple_dynamic_classifier.pth"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

frames_path = cur_dir + "/frames/"

@socketio.on('dynamic')
def dynamic(message):
    landmark_history = message['landmarkHistory']
    landmark_history = normalize_landmark_history(landmark_history)
    compressed = landmark_history_preprocess(landmark_history)

    tensor = torch.from_numpy(np.array(compressed))
    tensor = tensor.to(torch.float32)

    results = dynamic_model(tensor[None, ...])

    result_arr = results.detach().numpy()
    result = np.argmax(result_arr)

    data = {}
    data['result'] = str(result)
    serialized = json.dumps(data)
    socketio.emit("dynamic", serialized)

@socketio.on('stream')
def stream(message):
    features = message['features']
    
    if len(features) >= 21:
        # process landmark features by normalizing coordinates
        features = process_features(features)
        
        # convert to tensor for ml model
        tensor = torch.from_numpy(np.array(features))
        tensor = tensor.to(torch.float32)

        # run through static model
        results = static(tensor[None, ...])

        # get the highest confidence result
        result_arr = results.detach().numpy()
        result = np.argmax(result_arr)

        # return result
        print("result: " + chr(result + 65))
        data = {}
        # data['frame'] = frame
        data['result'] = str(chr(result + 65))
        serialized = json.dumps(data)

        socketio.emit("stream", serialized)
    


if __name__ == "__main__":
    app.run(debug=False)