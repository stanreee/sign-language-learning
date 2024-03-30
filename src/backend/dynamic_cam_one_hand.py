import cv2
import mediapipe as mp
import numpy as np
from sign_lang_model_dynamic import SignLangModelDynamic
import os
import torch
from sklearn.decomposition import PCA 
from recognition_model import RecognitionModel
import time

cur_dir = os.getcwd()

dynamic = RecognitionModel([cur_dir + "/trained_models/dynamic_one_hand.pt"], "dynamic")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

landmark_history = []
frameNum = 0
prev_input = None

capturing = False

FPS_INFO = False

cap = cv2.VideoCapture(1)
startTime = time.time()
# deltaTime = 0
prevTime = time.time()
while cap.isOpened():
    curTime = time.time()
    deltaTime = curTime - prevTime
    totalDeltaTime = curTime - startTime
    prevTime = curTime
    frameNum += 1

    if frameNum > 100 and FPS_INFO:
        print("FPS:", str(1 / deltaTime), "AVG FPS:", str(frameNum / totalDeltaTime))

    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    if capturing and results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        for point in landmarks.landmark:
            x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            features.append([point.x, point.y, point.z])
        if len(features) >= 21:
            landmark_history.append(features)
            if len(landmark_history) > 30:
                landmark_history.pop(0)
                result, confidence, input_vector = dynamic.evaluate(landmark_history, False, prev_input)
                prev_input = input_vector
                print(result, confidence)
                capturing = False
                landmark_history = []
    
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        capturing = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    