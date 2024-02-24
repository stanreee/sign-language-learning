import cv2
import numpy as np
from sklearn.decomposition import PCA

def extract_features(frame, hands, num_hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    reflect = False

    if results.multi_hand_landmarks:
        if num_hands == 1:
            if len(results.multi_hand_landmarks) == 1:
                handedness = results.multi_handedness[0].classification[0].label # label
                landmarks = results.multi_hand_landmarks[0]
                for point in landmarks.landmark:
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    features.append([point.x, point.y])
                if handedness.lower() == 'right':
                    reflect = True
        else:
            for i in range(len(results.multi_hand_landmarks)):
                landmarks = results.multi_hand_landmarks[i]
                for point in landmarks.landmark:
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    features.append([point.x, point.y])
            if len(results.multi_hand_landmarks) < 2:
                for i in range(21):
                    features.append([0, 0])

    return (features, reflect)

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
#   - normalize landmarks on the maximum coordinate magnitude to improve consistency
#   referenced from https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main
def process_features(features, reflect):
    base_x, base_y = 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0:
            base_x = feature[0]
            base_y = feature[1]
        feature[0] = base_x - feature[0] if not reflect else feature[0] - base_x
        feature[1] = base_y - feature[1]

    features = np.array(features).flatten()

    max_val = max(list(map(abs, features)))

    def normalize(n):
        return n / max_val
    
    features = list(map(normalize, features))

    return features