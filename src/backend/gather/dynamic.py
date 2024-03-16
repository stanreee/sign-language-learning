from classifier import Classifier
import os
import csv
from sklearn.decomposition import PCA
import numpy as np
from gather_util import extract_features, process_features, landmark_history_preprocess, normalize_landmark_history

class DynamicClassifier(Classifier):
    def __init__(self, hands) -> None:
        super().__init__("dynamic", hands)
        self.base_coords = [[], []]
        pass

    def save(self, data, id, num_hands):
        print(len(data), len(data[0]))
        compressed = normalize_landmark_history(data, False, num_hands)
        print(len(compressed))
        dataToSave = [id] + compressed
        
        # save to file
        fileName = str(self.name)
        if num_hands > 1: fileName += "_2"
        cur_dir = os.curdir
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(dataToSave)
        print("Frames saved for id", str(id) + ".")
    
    def capture(self, frame, frameNum, data):
        features, reflect, failed = extract_features(frame, self.hands, self.num_hands)
        if len(features) >= 21 if self.num_hands == 1 else 42 and not failed:
            data.append(features)
        # else:
        #     data, frameNum = self.forceEndCapture()
        if len(data) >= self.FRAME_CAP:
            data, frameNum = self.endCapture(data, frameNum)
            self.base_coords = [[], []]
        return (data, frameNum)