import numpy as np
from sklearn.decomposition import PCA

def get_features(landmarks):
    features = []
    for point in landmarks:
        features.append([point['x'], point['y']])

    return features

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
#   - normalize landmarks on the maximum coordinate magnitude to improve consistency
#   referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main

"""
    inputs:
        features:
            - a 2x21 dimension array consisting of X and Y coordinates of each of the 21 hand landmarks
    outputs:
        - a flattened 1x42 dimension array consisting of normalized X and Y coordinates of the 21 hand landmarks
"""
def process_features(features):
    base_x, base_y = 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0:
            base_x = feature[0]
            base_y = feature[1]
        feature[0] = base_x - feature[0]
        feature[1] = base_y - feature[1]

    features = np.array(features).flatten()

    max_val = max(list(map(abs, features)))

    def normalize(n):
        return n / max_val
    
    features = list(map(normalize, features))

    return features

"""
    compresses landmark_history array using principal 
    component analysis (PCA) into a 1x756 dimension array to be fed into ML model
    inputs:
        landmark_history:
            - 30x42 dimensional array (30 frames, 42 landmark points for each frame, each having been processed by process_features)
    outputs:
        compressed:
            - 1x756 dimensional array to be fed into ML model
"""
def landmark_history_preprocess(landmark_history):
    pca = PCA(n_components=18)
    pca.fit(landmark_history)

    compressed = []
    for i in range(pca.n_components_):
        compressed += np.ndarray.tolist(pca.components_[i])

    return compressed