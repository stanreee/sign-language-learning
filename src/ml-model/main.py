import cv2
import mediapipe as mp
import numpy as np
from util import process_features
from sign_lang_model import SignLangModel
from sign_lang_model_dynamic import SignLangModelDynamic
import os
import torch
from sklearn.decomposition import PCA

cur_dir = os.getcwd()

model = SignLangModel()
model.load_state_dict(torch.load(cur_dir + "/simple_classifier.pth"), strict=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

model.eval()

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    reflect = False

    if results.multi_hand_landmarks:
        handedness = results.multi_handedness[0].classification[0].label # label
        if len(results.multi_hand_landmarks) == 1:
            if handedness.lower() == 'right':
                reflect = True
            # for i in range(21):
            #     features.append([0, 0]) # appending dummy data if only one hand
        for hand in results.multi_hand_landmarks:
            landmarks = hand
            for point in landmarks.landmark:
                x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                features.append([point.x, point.y])

    if len(features) >= 21:
        features = process_features(features, reflect)

        # print(features)
        
        tensor = torch.from_numpy(np.array(features))
        tensor = tensor.to(torch.float32)

        # print(tensor)

        results = model(tensor[None, ...])

        result_arr = results.detach().numpy()
        result = np.argmax(result_arr)
        confidence = 2**results[0][result].item()

        # if confidence >= 0.95:
        print(chr(result + 65), confidence)
        # else:
        #     print("NOT CONFIDENT:", chr(result + 65), confidence)
        # # break

    cv2.imshow('Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

