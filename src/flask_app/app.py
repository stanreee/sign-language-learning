#Import necessary libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from sign_lang_model import SignLangModel
from util import process_features
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

model = SignLangModel()

max_frames = 300
cur_frames = 0

cur_dir = os.getcwd()
model.load_state_dict(torch.load(cur_dir + "/simple_classifier.pth"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

frames_path = cur_dir + "/frames/"

@socketio.on('stream')
def stream(message):
    # print(message)
    frame = message['image']
    encoded = message['image'].split("base64,")[1]
    image = np.fromstring(base64.b64decode(encoded, validate=True), np.uint8)

    # print(image)

    cv2frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(cv2frame, cv2.COLOR_BGR2RGB)
    # print(rgb_frame)
    results = hands.process(rgb_frame)
    
    ## image processing (move to module later?)
    features = []

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        for point in landmarks.landmark:
            features.append([point.x, point.y])
    
    if len(features) >= 21:
        features = process_features(features)
        
        tensor = torch.from_numpy(np.array(features))
        tensor = tensor.to(torch.float32)

        # print(tensor)

        results = model(tensor[None, ...])

        result_arr = results.detach().numpy()
        result = np.argmax(result_arr)

        print("result: " + chr(result + 65))
        # break
        data = {}
        data['frame'] = frame
        data['result'] = str(chr(result + 65))
        serialized = json.dumps(data)
        # print(serialized)

        socketio.emit("stream", serialized)
    


if __name__ == "__main__":
    app.run(debug=False)