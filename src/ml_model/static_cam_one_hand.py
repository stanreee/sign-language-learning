import cv2
import mediapipe as mp
import numpy as np
from util import process_features
from sign_lang_model import SignLangModel
from sign_lang_model_dynamic import SignLangModelDynamic
import os
import torch
from sklearn.decomposition import PCA
from recognition_model import RecognitionModel

cur_dir = os.getcwd()

static = RecognitionModel(cur_dir + "/trained_models/static_one_hand.pt", 1, "static")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    reflect = False

    # data to be sent by frontend:
    #   - landmark data (or landmark history for dynamic signs)
    #   - left or right hand (if only one hand)
    #   - number of hands
    #   - type of sign (dynamic or static)
    # -- GET LANDMARK DATA -- ** frontend responsibility **
    if results.multi_hand_landmarks:
        handedness = results.multi_handedness[0].classification[0].label # label
        if len(results.multi_hand_landmarks) == 1:
            if handedness.lower() == 'right':
                reflect = True
        landmarks = results.multi_hand_landmarks[0]
        for point in landmarks.landmark:
            x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            features.append([point.x, point.y, point.z])

    if len(features) >= 21:
        # reflect should be true if there is only one hand and that hand is the right hand
        result, confidence = static.evaluate(features, reflect)
        print(chr(result + 65), confidence)

    cv2.imshow('Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

