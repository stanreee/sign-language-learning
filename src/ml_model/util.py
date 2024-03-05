import numpy as np
from sklearn.decomposition import PCA
import copy

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
#   - normalize landmarks on the maximum coordinate magnitude to improve consistency
#   referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main
def process_features(features, reflect, base_coords=None):
    base_x, base_y, base_z = 0, 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0:
            base_x = feature[0] if not base_coords else base_coords[0]
            base_y = feature[1] if not base_coords else base_coords[1]
            base_z = feature[2] if not base_coords else base_coords[2]
        feature[0] = base_x - feature[0] if not reflect else feature[0] - base_x
        feature[1] = base_y - feature[1]
        feature[2] = base_z - feature[2]
    
    # if len(features) < 42:
    #     for i in range(21):
    #         features.append([0, 0]) # append dummy data if only one hand

    features = np.array(features).flatten()

    max_val = max(list(map(abs, features)))

    def normalize(n):
        return n / max_val
    
    features = list(map(normalize, features))

    return features

"""
    compresses landmark_history array using principal component analysis (PCA) into a 1x756 dimension array to be fed into ML model
    inputs:
        landmark_history:
            - 30x42 dimensional array (30 frames, 42 landmark points for each frame)
    outputs:
        compressed:
            - 1x756 dimensional array to be fed into ML model
"""
def landmark_history_preprocess(landmark_history, num_hands):
    dim = 18 if num_hands == 1 else 10
    pca = PCA(n_components=dim)
    pca.fit(landmark_history)

    compressed = []
    for i in range(pca.n_components_):
        compressed += np.ndarray.tolist(pca.components_[i])

    return compressed

def normalize_landmark_history(landmark_history, reflect):
    landmark_history_copy = copy.deepcopy(landmark_history)
    base_coords = landmark_history[0][0]
    for i in range(len(landmark_history)):
        features = landmark_history[i]
        processed = process_features(features, reflect, base_coords)
        landmark_history_copy[i] = processed
    return landmark_history_copy