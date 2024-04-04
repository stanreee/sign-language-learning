import numpy as np
import copy

def normalize_features(features):
    """
        Normalizes landmark features on the maximum value of the array.

        Parameters:
        features (array): Array of landmark coordinates.
    """
    max_val = max(list(map(abs, features)))

    def normalize(n):
        return n / max_val

    features = list(map(normalize, features))

    return features

def process_features(features, reflect, base_coords=None):
    """
        Repositions landmark features relative to the base_coords landmark. If no base_coords specified, the first landmark is designated as
        the base coordinate.

        Referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main

        Parameters:
        features (array): Array consisting of landmark coordinates.
        reflect (boolean): Reflects the coordinates along the y-axis. Necessary for users who are left handed.
        base_coords (array): If specified, this coordinate will be the base coordinate, and all coordinates in the features array will be shifted
        relative to this.

        Returns:
        Array of processed features.
    """
    base_x, base_y, base_z = 0, 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0 and base_z == 0:
            base_x = feature[0] if not base_coords else base_coords[0]
            base_y = feature[1] if not base_coords else base_coords[1]
            base_z = feature[2] if not base_coords else base_coords[2]
        feature[0] = base_x - feature[0] if not reflect else feature[0] - base_x
        feature[1] = base_y - feature[1]
        feature[2] = base_z - feature[2]

    return features

def normalize_landmark_history(landmark_history):
    """
        Flattens landmark history array into a 1-dimensional array, then normalizes it.

        Parameters:
        landmark_history (array): A 30x21 size array, consisting of 30 frames of 21 landmarks.

        Returns: 
        Compressed 1-dimensional array.
    """
    compressed = []
    for frame in landmark_history:
        compressed.extend(np.ndarray.flatten(np.array(frame)).tolist())

    compressed = normalize_features(compressed)

    return compressed

def process_landmark_history(landmark_history, reflect):
    """
        Processes landmark history by repositioning all coordinates relative to the first landmark of the first frame, then flattens
        and normalizes the data.

        Parameters:
        landmark_history (array): A 30x21 size array, consisting of 30 frames of 21 landmarks.

        Returns:
        Tuple consisting of the processed landmark history array and the base coordinate used.
    """
    landmark_history_copy = copy.deepcopy(landmark_history)

    base = landmark_history[0][0].copy()

    for i in range(len(landmark_history)):
        features = landmark_history[i]
        landmark_history_copy[i] = process_features(features, reflect, base)
    
    landmark_history_copy = normalize_landmark_history(landmark_history_copy)

    return (landmark_history_copy, base)