import cv2
from classifier import Classifier
from util import extract_features, process_features

class StaticClassifier(Classifier):

    def __init__(self, hands) -> None:
        super().__init__("static", hands)
        pass

    def capture(self, frame) -> None:
        features = extract_features(frame, self.hands)
        if len(features) >= 21:
            features = process_features(features)
            return features, frame
        self.capturing = False
        return [], frame
        
    
        