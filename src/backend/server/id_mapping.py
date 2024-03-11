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
        9: 'J',
        25: 'Z',
    },
    2: {
        0: 'spaghetti',
    }
}

def id_map(id, model, hands):
    MAPPING = STATIC if model == "static" else DYNAMIC
    return MAPPING[hands][id]