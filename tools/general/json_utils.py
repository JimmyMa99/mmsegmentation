

import json

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value