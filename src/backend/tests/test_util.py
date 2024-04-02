import pytest
import os
import copy
import numpy as np

import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from util import normalize_features, process_features

@pytest.fixture
def sample_flattened_features():
    features_1 = [1.0, 2.0, 3.0, -4.0]
    features_2 = [4.0, -6.0, 3.0, -1.0]
    return features_1, features_2

@pytest.fixture
def sample_unflattened_features():
    features_1 = [[1.0, 2.0, 5.0], [3.0, 4.0, -2.0], [5.0, 6.0, -3.0]]
    features_2 = [[-1.0, 1.0, 0.0], [2.0, 1.0, -3.0], [-3.0, 2.0, 1.0]]
    return features_1, features_2

def test_normalize_features(sample_flattened_features):
    features_1, features_2 = sample_flattened_features
    normalized = normalize_features(features_1)
    assert normalized == [0.25, 0.5, 0.75, -1]
    normalized = normalize_features(features_2)
    assert normalized == [4.0/6.0, -6.0/6.0, 3.0/6.0, -1.0/6.0]

def test_process_features(sample_unflattened_features):
    features_1, features_2 = sample_unflattened_features
    processed = process_features(copy.deepcopy(features_1), False)
    assert processed == [[0.0, 0.0, 0.0], [-2.0, -2.0, 7.0], [-4.0, -4.0, 8.0]]
    processed = process_features(copy.deepcopy(features_1), True)
    assert processed == [[0.0, 0.0, 0.0], [2.0, -2.0, 7.0], [4.0, -4.0, 8.0]]
    processed = process_features(copy.deepcopy(features_1), False, [1.0, 2.0, 3.0])
    assert processed == [[0.0, 0.0, -2.0], [-2.0, -2.0, 5.0], [-4.0, -4.0, 6.0]]
    processed = process_features(copy.deepcopy(features_2), False)
    assert processed == [[0.0, 0.0, 0.0], [-3.0, 0.0, 3.0], [2.0, -1.0, -1.0]]