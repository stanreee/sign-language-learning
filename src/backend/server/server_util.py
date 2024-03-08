import numpy as np
from sklearn.decomposition import PCA
import copy
import os

import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from util import process_features, landmark_history_preprocess, normalize_landmark_history

def get_features(landmarks):
    print(landmarks)
    features = []
    for point in landmarks:
        features.append([point['x'], point['y'], point['z']])

    return features