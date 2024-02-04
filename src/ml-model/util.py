import numpy as np
from sklearn.decomposition import PCA

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
#   - normalize landmarks on the maximum coordinate magnitude to improve consistency
#   referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main
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

def landmark_history_preprocess(landmark_history):
    landmark_history.pop(0)
    pca = PCA(n_components=18)
    pca.fit(landmark_history)

    compressed = []
    for i in range(pca.n_components_):
        compressed += np.ndarray.tolist(pca.components_[i])

    return compressed