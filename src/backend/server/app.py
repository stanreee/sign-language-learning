#Import necessary libraries
from flask import Flask
from flask_socketio import SocketIO
import os
import json
from id_mapping import id_map

from server_util import get_features

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

from recognition_model import RecognitionModel

#Initialize the Flask app
app = Flask(__name__)

# Initialize socket
socketio = SocketIO(app, cors_allowed_origins="*")

dynamic_model = RecognitionModel([parent + "/trained_models/dynamic_one_hand.pt"], "dynamic")
static = RecognitionModel([parent + "/trained_models/static_one_hand.pt"], "static")
dynamic_model_two = RecognitionModel([parent + "/trained_models/dynamic_two_1.pt", parent + "/trained_models/dynamic_two_2.pt"], "dynamic_2")

max_frames = 300
cur_frames = 0

cur_dir = os.getcwd()

frames_path = cur_dir + "/frames/"

@socketio.on('dynamic')
def dynamic(message):
    """
        Socket endpoint for dynamic signs. 

        Expects the input to be of shape:
        {
            landmarkHistory: array of 30 frames of landmarks
            reflect: boolean
            hands: int
        }
    """
    landmark_history = message['landmarkHistory']
    reflect = message['reflect']
    hands = message['numHands']
    results = dynamic_model.evaluate(landmark_history, reflect) if hands == 1 else dynamic_model_two.evaluate(landmark_history, reflect)

    result, confidence = results[0], results[1]

    data = {}
    data['result'] = str(id_map(result, model="dynamic", hands=hands))
    data['confidence'] = str(confidence if confidence is not None else 0)
    serialized = json.dumps(data)
    socketio.emit("dynamic", serialized)

@socketio.on('stream')
def stream(message):
    """
        Socket endpoint for static signs.

        Expects the input to be of shape:
        {
            features: array consisting of landmark coordinates
            reflect: boolean
        }
    """
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