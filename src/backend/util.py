import numpy as np
from sklearn.decomposition import PCA
import copy

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
#   - normalize landmarks on the maximum coordinate magnitude to improve consistency
#   referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main
def process_features(features, reflect, base_coords=None, shouldNormalize=False):
    base_x, base_y, base_z = 0, 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0 and base_z == 0:
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
    
    if shouldNormalize:
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
    # dim = 18 if num_hands == 1 else 10
    # pca = PCA(n_components=dim)
    # pca.fit(landmark_history)

    compressed = []
    coords = [[], [], []]
    for frame in landmark_history:
        for i in range(len(frame)):
            landmark = frame[i]
            coords[i%3].append(landmark)
    
    for i in range(len(coords)):
        max_val = max(list(map(abs, coords[i])))

        def normalize(n):
            return n / max_val
        
        coords[i] = list(map(normalize, coords[i]))
        compressed += coords[i]

    # compressed = []
    # for i in range(len(landmark_history)):
    #     compressed += landmark_history[i]

    return compressed

def normalize_landmark_history(landmark_history, reflect, num_hands):
    landmark_history_copy = copy.deepcopy(landmark_history)
    # base_coords = landmark_history[0][0]
    base_coords = [[], []]
    for i in range(len(landmark_history)):
        features = landmark_history[i]
        all_hand_features = []
        for hand in range(num_hands):
            hand_features = features[0:21] if hand == 0 else features[21:]
            if not base_coords[hand]:
                base_coords[hand] = hand_features[0].copy()
            all_hand_features.extend(process_features(hand_features, reflect, base_coords[hand], shouldNormalize=False))
        landmark_history_copy[i] = all_hand_features

    return landmark_history_copy