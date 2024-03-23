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
    
    if shouldNormalize:
        features = np.array(features).flatten()

        max_val = max(list(map(abs, features)))

        def normalize(n):
            return n / max_val

        features = list(map(normalize, features))

    return features

# def landmark_history_preprocess_old(landmark_history, num_hands):
#     # dim = 18 if num_hands == 1 else 10
#     # pca = PCA(n_components=dim)
#     # pca.fit(landmark_history)

#     print(len(landmark_history), len(landmark_history[0]))

#     compressed = []
#     coords = [
#         [[], [], []],
#         [[], [], []]
#     ]
#     all_coordinates = [[], []]

#     for idx, frame in enumerate(landmark_history):
#         for hand in range(num_hands):
#             hand_features = frame[0:21] if hand == 0 else frame[21:]
#             for landmarks in hand_features:
#                 all_coordinates[hand].extend(landmarks)
    

#     for i in range(num_hands):
#         max_val = max(list(map(abs, all_coordinates[i])))

#         def normalize(n):
#             return n / max_val
        
#         # print(all_coordinates[i])
        
#         all_coordinates[i] = list(map(normalize, all_coordinates[i]))

#     for i in range(num_hands):
#         compressed += all_coordinates[i]

#     # compressed = []
#     # for i in range(len(landmark_history)):
#     #     compressed += landmark_history[i]

#     return compressed

def landmark_history_preprocess(landmark_history):
    # compressed = np.ndarray.flatten(landmark_history).tolist()
    compressed = []
    for frame in landmark_history:
        compressed.extend(np.ndarray.flatten(np.array(frame)).tolist())

    max_val = max(list(map(abs, compressed)))

    def normalize(n):
        return n / max_val
    
    compressed = list(map(normalize, compressed))

    # print(compressed)

    return compressed

# def normalize_landmark_history_old(landmark_history, reflect, num_hands):
#     # print(landmark_history)
#     landmark_history_copy = copy.deepcopy(landmark_history)
#     # base_coords = landmark_history[0][0]
#     base_coords = [[], []]
#     for i in range(len(landmark_history)):
#         features = landmark_history[i]
#         all_hand_features = []
#         for hand in range(num_hands):
#             hand_features = features[0:21] if hand == 0 else features[21:]
#             # print(len(hand_features))
#             if not base_coords[hand]:
#                 base_coords[hand] = hand_features[0].copy()
#                 print("old", base_coords[hand])
#                 if hand > 0:
#                     base_coords[hand] = np.ndarray.tolist(np.subtract(base_coords[0], base_coords[hand]))
#             all_hand_features.extend(process_features(hand_features, reflect, base_coords[hand], shouldNormalize=False))
#         landmark_history_copy[i] = all_hand_features

#     print(landmark_history_copy)

#     landmark_history_copy = landmark_history_preprocess_old(landmark_history_copy, num_hands)

#     return landmark_history_copy

def normalize_landmark_history(landmark_history, reflect, base_coords=None):
    landmark_history_copy = copy.deepcopy(landmark_history)

    # base = base_coords if base_coords else landmark_history[0][0].copy()
    base = landmark_history[0][0].copy()

    for i in range(len(landmark_history)):
        features = landmark_history[i]
        landmark_history_copy[i] = process_features(features, reflect, base, shouldNormalize=False)
    
    landmark_history_copy = landmark_history_preprocess(landmark_history_copy)

    return (landmark_history_copy, base)