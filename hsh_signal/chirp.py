import numpy as np
import matplotlib.pyplot as plt

from hsh_signal.heartseries import Series
from hsh_signal.signal import lowpass_fft, filter_fft_ff, localmax, localmax_climb


def chirp(T, f1, f2, fps=48000):
    # T in samples, NOT sec
    t = np.arange(T) / float(fps)
    f = f1 + (t / t[-1]) * (f2 - f1)
    sig = np.cos(2*np.pi*f*t)
    return sig


def cross_corr(x1, x2):
    # x1 is the shorter chirp
    assert len(x1) <= len(x2)
    pad = len(x1)//2
    x1p, x2p = np.pad(x1, (pad, pad), mode='constant'), np.pad(x2, (2*pad, 2*pad-1), mode='constant')
    x1pi = x1p[::-1]
    #corr = np.convolve(x1p, x2pi, mode='valid')
    corr = filter_fft_ff(x2p, x1pi)
    # length = max(M, N) - min(M, N) + 1
    # since we are padded up by len(x1), len(corr) == len(x2)
    assert len(corr) == len(x2)

    #print 'len(x1), len(x2), len(x1p), len(x2p), len(corr)', len(x1), len(x2), len(x1p), len(x2p), len(corr)

    # normalize
    return corr / np.sqrt(np.sum(x1**2) * np.sum(x2**2))



class AudioChirpDetector(object):
    def __init__(self, track, debug=False):
        self.debug = debug
        self.track = track
        chrp = chirp(0.5*track.fps, 100, 8000, fps=track.fps)[:int(0.05*track.fps)+1]
        # ensure chrp_bit is even length (assumed by later code)
        self.chirp = chrp[:len(chrp)-len(chrp)%2]

        self.times, self.idxs = [], []

    def chirp_times(self):
        track, chirp = self.track, self.chirp

        SMOOTHED_CORR_THRESHOLD = 0.08
        CORR_THRESHOLD = 0.35

        corr = cross_corr(chirp, track.x)

        smoothed_corr = lowpass_fft(np.abs(corr), track.fps, cf=100.0, tw=20.0) / np.max(np.abs(corr))

        if self.debug:
            plt.plot(track.t, corr / np.max(np.abs(corr)), label='corr')
            plt.plot(track.t, smoothed_corr, label='smoothed_corr')
            plt.legend()
            plt.show()

        peak_locs = np.where(
            localmax(corr) &
            (smoothed_corr > SMOOTHED_CORR_THRESHOLD) &
            (corr / np.max(np.abs(corr)) > CORR_THRESHOLD)
        )[0]

        # look in +/- 5 ms of the smoothed_corr peak for real corr peak
        corr_peak_locs = np.unique(localmax_climb(corr, peak_locs, hwin=int(track.fps * 0.005)))

        times = track.t[corr_peak_locs]

        self.times, self.idxs = times, corr_peak_locs

        return self.times, self.idxs


def audio_chirp_times(sig, fps):
    track = Series(sig, fps=fps)
    cd = AudioChirpDetector(track, debug=False)
    times, idxs = cd.chirp_times()
    times = times[:4]
    print 'audio_chirp_times() times', times
    #assert len(times) == 4

    dtimes = np.diff(times)

    assert np.abs(dtimes[0] - 0.5) < 1e-3 and np.abs(dtimes[2] - 0.5) < 1e-3  # sometimes last chirp missing?! but if it does, signal's crap.
    return times
