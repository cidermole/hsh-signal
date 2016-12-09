from brueser.brueser2 import brueser_beatdetect
from .heartseries import HeartSeries


def ppg_beatdetect(ppg):
    ibeats, ibis = brueser_beatdetect(ppg.x, ppg.fps)
    return HeartSeries(ppg.x, ibeats, fps=ppg.fps, lpad=ppg.lpad)
