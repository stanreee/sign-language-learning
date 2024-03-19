#Import necessary libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from server_util import process_features, get_features, normalize_landmark_history, landmark_history_preprocess
from recognition_model import RecognitionModel
import torch
import cv2
import numpy as np
import base64
import os
import mediapipe as mp
import json
from id_mapping import id_map

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

#Initialize the Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# static = StaticModel()
# dynamic_model = DynamicModel()

dynamic_model = RecognitionModel([parent + "/trained_models/dynamic_one_hand.pt"], "dynamic")
static = RecognitionModel([parent + "/trained_models/static_one_hand.pt"], "static")
dynamic_model_two = RecognitionModel([parent + "/trained_models/dynamic_two_1.pt", parent + "/trained_models/dynamic_two_2.pt"], "dynamic_2")

max_frames = 300
cur_frames = 0

cur_dir = os.getcwd()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

frames_path = cur_dir + "/frames/"

@socketio.on('dynamic')
def dynamic(message):
    landmark_history = message['landmarkHistory']
    reflect = message['reflect']
    hands = message['numHands']
    results = dynamic_model.evaluate(landmark_history, reflect) if hands == 1 else dynamic_model_two.evaluate(landmark_history, reflect)

    result, confidence = results[0], results[1]

    print(landmark_history)
    print(result)

    data = {}
    data['result'] = str(id_map(result, model="dynamic", hands=hands))
    data['confidence'] = str(confidence)
    serialized = json.dumps(data)
    socketio.emit("dynamic", serialized)

@socketio.on('stream')
def stream(message):
    features = message['features']
    reflect = message['reflect']
    
    if len(features) >= 21:
        # run through static model
        results = static.evaluate(features, reflect)

        result, confidence = results[0], results[1]
        
        data = {}
        data['result'] = str(id_map(result, model="static", hands=1))
        data['confidence'] = str(confidence)
        serialized = json.dumps(data)

        socketio.emit("stream", serialized)
    


if __name__ == "__main__":
    app.run(debug=False)