from brueser.brueser2 import brueser_beatdetect
from .heartseries import HeartSeries
#from .signal import evenly_resample
import numpy as np

def ppg_beatdetect(ppg, debug=False):
    return ppg_beatdetect_brueser(ppg, debug)

def ppg_beatdetect_getrr(ppg, debug=False):
    """UNTESTED!"""
    import sys
    srv_path = '/home/david/Info/ownCloud/heartshield/heartshield-server-backend'
    if srv_path not in sys.path:
        sys.path.append(srv_path)
        sys.path.append(srv_path + '/ML')
    from ppg_beatdetector import getrr
    # XXX: getrr() fps must be 30.0 currently, there is a subtle getrr() bug otherwise
    assert np.abs(ppg.fps - 30.0) < 1e-3
    fps = ppg.fps
    data = np.vstack((ppg.t, ppg.x)).T
    ibi, filtered, idx = getrr(data, fps)

    return HeartSeries(ppg.x, idx, fps=ppg.fps, lpad=ppg.lpad)

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
