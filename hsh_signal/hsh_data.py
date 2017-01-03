import numpy as np
import json
from datetime import datetime
import time


class PrettyFloat(float):
    def __repr__(self):
        # fix time floats to microsecond precision in JSON encoding sent to server.
        # for now, amplitudes are too precise. We could encode timestamps and amplitudes separately but that's overkill.
        return '%.6f' % self


def pretty_floats(obj):
    if isinstance(obj, float) or isinstance(obj, np.float32):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return pretty_floats(obj.tolist())
        elif isinstance(obj, datetime):
            return int(time.mktime(obj.timetuple()))
        else:
            return json.JSONEncoder.default(self, obj)

    def encode(self, obj):
        if isinstance(obj, (float, np.float32, dict, list, tuple)):
            return json.JSONEncoder.encode(self, pretty_floats(obj))
        return json.JSONEncoder.encode(self, obj)
