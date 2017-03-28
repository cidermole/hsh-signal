from __future__ import division

from .filter import *
from .signal import highpass_fft
from .heartseries import Series, HeartSeries
from .ecg import scrub_ecg, NoisyECG

import wave
import numpy as np


class AlivecorFilter(SourceBlock):
    """Demodulates an AliveCor ECG transmitted via FM audio signal."""
    def __init__(self, fps):
        super(AlivecorFilter, self).__init__()
        self.sampling_rate = fps

        self.prefilter = Bandpass(low_cutoff_freq=17000, high_cutoff_freq=21000, transition_width=500, sampling_rate=self.sampling_rate)
        self.hilbert = Hilbert()
        # roughly center on the observed 18.8 kHz carrier
        self.pll = PLL(loop_bw=1500, max_freq=21000, min_freq=17000, sampling_rate=self.sampling_rate)
        self.lowpass = Lowpass(cutoff_freq=100, transition_width=5, sampling_rate=self.sampling_rate)
        self.bandreject = Bandreject(low_cutoff_freq=40, high_cutoff_freq=60, transition_width=3, sampling_rate=self.sampling_rate)
        #Logger.debug('lowpass taps={}, bandreject taps={}'.format(self.lowpass._ntaps, self.bandreject._ntaps))

        self.delay = self.prefilter.delay + self.hilbert.delay + self.lowpass.delay + self.bandreject.delay

        connect(self.prefilter, self.hilbert, self.pll, self.lowpass, self.bandreject)  #, self - but no.

    def connect(self, consumer):
        # redirect the output
        self.bandreject.connect(consumer)

    def put(self, x):
        self.prefilter.put(x)


def decode_alivecor(signal, fps=48000, debug=False):
    alivecor = AlivecorFilter(fps)
    signal_padded = np.pad(signal, (0, alivecor.delay), mode='constant')  # pad with trailing zeros to force returning complete ECG
    mic = ChunkDataSource(data=signal_padded, batch_size=179200, sampling_rate=fps)
    ecg = DataSink()
    #mic.connect(alivecor)
    #alivecor.connect(ecg)
    connect(mic, alivecor, ecg)

    # to do: use apply_filter() instead of stuff below

    # push through all the data
    prev_t = time.time()
    mic.start()
    while not mic.finished():
        if time.time() > prev_t + 1.0 and debug:
            print 'progress: {} %'.format(mic.progress())
            prev_t = time.time()
        mic.poll()
    mic.stop()

    return ecg.data[alivecor.delay:]  # cut off leading filter delay (contains nonsense output)


def load_raw_audio(file_name):
    """Returns (sampling_rate, samples) where samples is an array of floats"""
    wf = wave.open(file_name)
    nframes = wf.getnframes()
    buf = wf.readframes(nframes)
    arr = np.frombuffer(buf, dtype=np.int16)
    arr = arr.reshape((nframes, wf.getnchannels())).T

    assert(wf.getsampwidth() == 2)  # for np.int16 to hold
    #assert(wf.getframerate() == 48000)  # Android recs 48 kHz?!

    raw_audio = arr[0] / float(2**15)  # assuming 16-bit wav file, left if stereo
    return raw_audio, wf.getframerate()


def beatdet_alivecor(signal, fps=48000, lpad_t=0):
    """decode, scrub, and beatdetect AliveCor."""
    #print 'ecg_fps=', ecg_fps
    ecg_raw = decode_alivecor(signal, fps=fps)
    ecg_dec_fps = 300
    ecg = ecg_raw[::int(fps/ecg_dec_fps)]
    ecg = highpass_fft(ecg, fps=ecg_dec_fps)
    ecg = Series(ecg, fps=ecg_dec_fps, lpad=lpad_t*ecg_dec_fps)

    scrubbed = scrub_ecg(ecg)
    ne = NoisyECG(scrubbed)
    ecg2 = HeartSeries(scrubbed.x, ne.beat_idxs, fps=ecg.fps, lpad=ecg.lpad)
    return ecg2


def beatdet_ecg(series):
    ne = NoisyECG(series)
    return HeartSeries(series.x, ne.beat_idxs, fps=series.fps, lpad=series.lpad)
