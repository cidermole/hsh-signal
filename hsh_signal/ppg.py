from brueser.brueser2 import brueser_beatdetect
from .heartseries import HeartSeries
import numpy as np


def ppg_beatdetect(ppg):
    ibeats, ibis = brueser_beatdetect(ppg.x, ppg.fps)

    # remove beats too close to each other
    deltas = np.diff(np.insert(np.array(ibeats), 0, 0) / float(ppg.fps))
    idupes = np.where(deltas <= 0.3)[0]
    ibeats = np.delete(ibeats, idupes)

    ibeats = ibeats[np.arange(len(ibeats))]

    return HeartSeries(ppg.x, ibeats, fps=ppg.fps, lpad=ppg.lpad)
