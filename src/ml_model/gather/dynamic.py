from classifier import Classifier
import os
import csv
from sklearn.decomposition import PCA
import numpy as np
from gather_util import extract_features, process_features, landmark_history_preprocess

class DynamicClassifier(Classifier):
    def __init__(self, hands) -> None:
        super().__init__("dynamic", hands)
        self.base_coords = [[], []]
        pass

    def save(self, data, id, num_hands):
        # reduce frame input dimensions from 30 to 18 (one hand) or 10 (two hands)
        compressed = landmark_history_preprocess(data, num_hands)
        dataToSave = [id] + compressed
        
        # save to file
        fileName = str(self.name)
        if num_hands > 1: fileName += "_2"
        cur_dir = os.curdir
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(dataToSave)
        print("Frames saved for id", id, ".")
    
    def capture(self, frame, frameNum, data):
        features, reflect, failed = extract_features(frame, self.hands, self.num_hands)
        if len(features) >= 21 if self.num_hands == 1 else 42 and not failed:
            all_hand_features = []
            for hand in range(self.num_hands):
                hand_features = features[0:21] if hand == 0 else features[21:]
                # designate first landmark of the first frame as the base landmark
                # for each frame, all landmark coordinates will be relative to thsi base landmark
                if not self.base_coords[hand]:
                    self.base_coords[hand] = hand_features[0].copy()
                all_hand_features += process_features(hand_features, reflect, self.base_coords[hand], normalize=False)
            # if not self.base_coords:
            #     self.base_coords = features[0]
            # features = process_features(features, reflect, self.base_coords)
            data.append(all_hand_features)
        else:
            data, frameNum = self.forceEndCapture()
        if frameNum >= self.FRAME_CAP:
            data, frameNum = self.endCapture(data, frameNum)
            self.base_coords = [[], []]
        return (data, frameNum)