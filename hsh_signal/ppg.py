from brueser.brueser2 import brueser_beatdetect
from .heartseries import HeartSeries
from hsh_beatdet import beatdet_getrr_v1, beatdet_getrr_v2, beatdet_getrr_v2_fracidx, beatdet
#from .signal import evenly_resample
import numpy as np


def ppg_beatdetect(ppg, debug=False):
    return ppg_beatdetect_brueser(ppg, debug)

def ppg_beatdetect_getrr(ppg, type='fracidx', debug=False):
    # regular | fracidx
    # XXX: getrr() fps must be 30.0 currently, there is a subtle getrr() bug otherwise
    assert np.abs(ppg.fps - 30.0) < 1e-3
    fps = ppg.fps
    data = np.vstack((ppg.t, ppg.x)).T

    if type == 'fracidx':
        return beatdet(data, beatdet_getrr_v2_fracidx)
    else:
        return beatdet(data, beatdet_getrr_v2)
    #ibi, filtered, idxs, tbeats = beatdet(data, beatdet_getrr_v2)

def ppg_beatdetect_brueser(ppg, debug=False):
    ibeats, ibis = brueser_beatdetect(ppg.x, ppg.fps)

    # remove beats too close to each other
    deltas = np.diff(np.insert(np.array(ibeats), 0, 0) / float(ppg.fps))
    if debug:
        print 'ibeats', ibeats
        print 'deltas', deltas
    idupes = np.where(deltas <= 0.3)[0]
    if debug: print 'idupes', idupes
    ibeats = np.delete(ibeats, idupes)
    if debug: print 'ibeats after delete', ibeats

    ibeats = ibeats[np.arange(len(ibeats))]

    return HeartSeries(ppg.x, ibeats, fps=ppg.fps, lpad=ppg.lpad)


    ppg_dt, rel_fps_error = analyze_ecg_ppg_base(ecg, ppg, step=50e-3, debug=False)
    clock_bias = (1.0 + rel_fps_error)
    ppg_new = HeartSeries(ppg.x, ppg.ibeats, ppg.fps * clock_bias)