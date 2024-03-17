STATIC = {
    1: {
        index: chr(65 + index) for index in range(26)
    },
    2: {}
}

DYNAMIC = {
    1: {
        0: 'no',
        1: 'where',
        2: 'future',
        3: 'yes',
        4: 'need',
        5: 'thank you',
        6: 'hello',
        7: 'home',
        8: 'please',
        9: 'J',
        10: 'who',
        11: 'why',
        25: 'Z',
    },
    2: {
        0: 'spaghetti',
        1: 'how are you',
        2: 'help',
        3: 'friend',
        4: 'family',
        5: 'when',
        6: 'what',
    }
}

def id_map(id, model, hands):
    MAPPING = STATIC if model == "static" else DYNAMIC
    return MAPPING[hands][id]