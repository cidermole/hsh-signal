from __future__ import division

from .filter import *


class ChunkDataSource(SourceBlock):
    """
    Fake Microphone signal source for testing. Provides a wav as audio.
    """
    def __init__(self, data, batch_size, sampling_rate=44100):
        super(ChunkDataSource, self).__init__()
        self.sampling_rate = sampling_rate
        self._data = data
        self._batch_size = batch_size
        self._i = 0

    def poll(self):
        """Call this regulary in order to trigger the callback."""
        # currently called with 30 fps in kivy -> could compute batch_size via sampling_rate
        #Logger.debug('FakeMic.poll()')
        before = time.time()
        self.put(self._data[self._i:self._i+self._batch_size])
        after = time.time()
        #Logger.debug('FakeMic: poll() took {} sec'.format(after-before))
        self._i += self._batch_size

    def progress(self):
        return float(self._i) / len(self._data) * 100.0

    def start(self): pass
    def stop(self): pass

    def finished(self):
        return self._i >= len(self._data)


class AlivecorFilter(SourceBlock):
    """Demodulates an AliveCor ECG transmitted via FM audio signal."""
    def __init__(self, mic):
        super(AlivecorFilter, self).__init__()
        self.sampling_rate = mic.sampling_rate

        self.mic = mic
        self.hilbert = Hilbert()
        # roughly center on the observed 18.8 kHz carrier
        self.pll = PLL(loop_bw=1500, max_freq=22000, min_freq=16000, sampling_rate=self.sampling_rate)
        self.lowpass = Lowpass(cutoff_freq=100, transition_width=5, sampling_rate=self.sampling_rate)
        self.bandreject = Bandreject(low_cutoff_freq=40, high_cutoff_freq=60, transition_width=3, sampling_rate=self.sampling_rate)
        #Logger.debug('lowpass taps={}, bandreject taps={}'.format(self.lowpass._ntaps, self.bandreject._ntaps))

        connect(self.mic, self.hilbert, self.pll, self.lowpass, self.bandreject, self)


def decode_alivecor(signal, debug=False, fps=48000):
    mic = ChunkDataSource(data=signal, batch_size=179200, sampling_rate=fps)
    alivecor = AlivecorFilter(mic)
    ecg = DataSink()
    alivecor.connect(ecg)

    # push through all the data
    prev_t = time.time()
    mic.start()
    while not mic.finished():
        if time.time() > prev_t + 1.0 and debug:
            print 'progress: {} %'.format(mic.progress())
            prev_t = time.time()
        mic.poll()
    mic.stop()

    return ecg.data

