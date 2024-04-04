import os

import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def get_features(landmarks):
    print(landmarks)
    features = []
    for point in landmarks:
        features.append([point['x'], point['y'], point['z']])

    return features