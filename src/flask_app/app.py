#Import necessary libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import base64
import os
#Initialize the Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

max_frames = 300
cur_frames = 0

cur_dir = os.getcwd()

frames_path = cur_dir + "/frames/"

@socketio.on('stream')
def stream(message):
    # print(message)
    frame = message['image']
    socketio.emit("stream", frame)
    # encoded = message['image'].split("base64,")[1]
    # image = base64.b64decode(encoded, validate=True)
    # if frame < max_frames:
    #     print(frame)
    #     fh = open(frames_path + "image_" + str(frame) + ".png", "wb")
    #     fh.write(image)
    #     fh.close()


if __name__ == "__main__":
    app.run(debug=True)