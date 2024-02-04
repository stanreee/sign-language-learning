import cv2
import mediapipe as mp
import numpy as np
from util import process_features, landmark_history_preprocess
from sign_lang_model_dynamic import SignLangModelDynamic
import os
import torch
from sklearn.decomposition import PCA 

cur_dir = os.getcwd()

model = SignLangModelDynamic()
model.load_state_dict(torch.load(cur_dir + "/simple_dynamic_classifier.pth"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

model.eval()

landmark_history = []
frameNum = 0

capturing = False

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    if capturing:
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            for point in landmarks.landmark:
                x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                features.append([point.x, point.y])
        if len(features) >= 21:
            features = process_features(features)
            landmark_history.append(features)
            if len(landmark_history) > 30:
                compressed = landmark_history_preprocess(landmark_history)

                tensor = torch.from_numpy(np.array(compressed))
                tensor = tensor.to(torch.float32)

                results = model(tensor[None, ...])

                result_arr = results.detach().numpy()
                result = np.argmax(result_arr)
                print(result)
                capturing = False
                landmark_history = []
    
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        capturing = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    