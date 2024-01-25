#Import necessary libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
#Initialize the Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('stream')
def stream(message):
    print(message)
    s = message['stream']
    print(s)

if __name__ == "__main__":
    app.run(debug=True)