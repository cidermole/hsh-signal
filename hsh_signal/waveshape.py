import matplotlib.pyplot as plt
import numpy as np
from hsh_signal.signal import localmax, lowpass, lowpass_fft
from scipy.interpolate import interp1d


def ppg_wave_foot(ppg_raw_l, ppg_l):
    """
    :param ppg_raw_l: PPG signal, evenly spaced
    :param ppg_l: highpass filtered, beat detected PPG signal
    """
    #
    # move beats to the wave foot
    #

    # highpass (`ppg()`) -> lowpass -> localmax
    #
    # to do: this is not methodologically thought through.
    # to do: testing (does it work cleanly under noisy conditions?)
    #
    ratio=10  # use same ratio for all!
    ppg_smoothed = ppg_l.upsample(ratio)
    ppg_smoothed.x = lowpass_fft(ppg_smoothed.x, ppg_smoothed.fps, cf=6.0, tw=0.5)

    # fill `ileft_min` with index of next local minimum to the left
    # for noise robustness, use some smoothing before
    #local_min = localmax(ppg.x)
    local_min = localmax(ppg_smoothed.x)
    ileft_min = np.arange(len(local_min)) * local_min
    for i in range(1, len(ileft_min)):
        if ileft_min[i] == 0:
            ileft_min[i] = ileft_min[i - 1]

    ppg_smoothed.ibeats = ileft_min[ppg_smoothed.ibeats.astype(int)]
    ppg_smoothed.tbeats = (ppg_smoothed.ibeats - ppg_smoothed.lpad) / float(ppg_smoothed.fps)

    #
    # hack to get beats onto ppg_raw
    #
    ppg_u = ppg_l.upsample(ratio)
    ppg_raw_tmp = ppg_raw_l.upsample(ratio)
    ppg_raw_u = ppg_l.upsample(ratio)
    ppg_raw_u.x = ppg_raw_tmp.x
    ppg_raw_u.ibeats = ppg_smoothed.ibeats
    ppg_raw_u.tbeats = ppg_smoothed.tbeats

    ppg_baselined = beat_baseline(ppg_raw_u, ppg_u)
    return ppg_baselined


def beat_baseline(ppg_feet, ppg_beats):
    """
    :param ppg_feet: raw ppg with beats located at the wave foot.
    :param ppg_beats: ppg with beats located on the edge as usual.
    """
    ppg = ppg_feet
    xu = np.zeros(len(ppg.x))
    fps = ppg.fps
    ibeats = ppg.ibeats.astype(int)

    # where long periods without beats, we need support from raw average (~baseline)
    ibis = np.diff(ppg.tbeats)
    median_ibi = np.median(ibis)
    long_ibis = np.where(ibis > 1.5 * median_ibi)[0]
    #print 'long_ibis', long_ibis
    ib_start = ppg.ibeats[long_ibis]
    ib_end = ppg.ibeats[np.clip(long_ibis + 1, 0, len(ppg.ibeats))]
    if len(ib_end) < len(ib_start): ib_end = np.insert(ib_end, len(ib_end), len(ppg.x) - 1)
    # iterate over long holes and fill them with fake ibis
    # to do: actually use raw average baseline, not some random value at a picked index
    extra_ibeats = []
    for s, e in zip(ib_start, ib_end):
        #print 's,e', s, e, ppg.t[s], ppg.t[e]
        extra_ibeats += list(np.arange(s + fps * median_ibi, e - 0.5 * median_ibi * fps, fps * median_ibi).astype(int))

    #print 'extra_ibeats', extra_ibeats, 'extra_tbeats', ppg.t[extra_ibeats]

    ibeats = np.array(sorted(list(ibeats) + list(extra_ibeats)))
    xu[ibeats] = ppg.x[ibeats]

    lf = interp1d(ibeats, xu[ibeats])
    xul = lf(np.arange(ibeats[0], ibeats[-1]))
    xul = np.pad(xul, (ibeats[0], len(xu) - ibeats[-1]), mode='constant')
    # to do: should we lowpass filter here, to get rid of all higher freq components?

    f_cf = 1.8
    f_tw = f_cf / 2.0
    xu_filtered = lowpass_fft(xul, ppg.fps, cf=f_cf, tw=f_tw)  # * scaling  # * ratio

    if False:
        plt.plot(ppg.t, np.clip(xu_filtered, a_min=220, a_max=270), c='b')
        plt.plot(ppg.t, np.clip(ppg.x, a_min=200, a_max=270), c='r')
        plt.show()

    ppg_detrended = ppg.copy()
    ppg_detrended.x = ppg.x - xu_filtered
    ppg_detrended.tbeats, ppg_detrended.ibeats = ppg_beats.tbeats, ppg_beats.ibeats

    return ppg_detrended
    # return HeartSeries(xu, ppg.ibeats, ppg.fps, ppg.lpad)
