import cv2
import numpy as np
from sklearn.decomposition import PCA
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from util import process_features, process_landmark_history, normalize_features

def extract_features(frame, hands, num_hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    reflect = False
    failed = False

    if results.multi_hand_landmarks:
        if num_hands == 1:
            if len(results.multi_hand_landmarks) == 1:
                handedness = results.multi_handedness[0].classification[0].label # label
                landmarks = results.multi_hand_landmarks[0]
                for point in landmarks.landmark:
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    features.append([point.x, point.y, point.z])
                if handedness.lower() == 'right':
                    reflect = True
        else:
            for i in range(len(results.multi_hand_landmarks)):
                landmarks = results.multi_hand_landmarks[i]
                for point in landmarks.landmark:
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    features.append([point.x, point.y, point.z])
            if len(results.multi_hand_landmarks) < 2:
                failed = True

    return (features, reflect, failed)