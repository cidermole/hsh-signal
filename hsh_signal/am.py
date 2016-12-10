from __future__ import division

from .filter import *


class AMFilter(SourceBlock):
    """Demodulates an AM audio signal."""
    def __init__(self, fps):
        super(AMFilter, self).__init__()
        self.sampling_rate = fps

        f_carrier = 10000
        f_low, f_high = f_carrier - 1000, f_carrier + 1000

        self.prefilter = Bandpass(low_cutoff_freq=f_low, high_cutoff_freq=f_high, transition_width=500, sampling_rate=self.sampling_rate)
        self.hilbert = Hilbert()
        self.pll = PLL(loop_bw=100, max_freq=f_high, min_freq=f_low, sampling_rate=self.sampling_rate)
        self.lowpass = Lowpass(cutoff_freq=100, transition_width=5, sampling_rate=self.sampling_rate)

        self.delay = self.prefilter.delay + self.hilbert.delay + self.lowpass.delay

        connect(self.prefilter, self.hilbert, self.pll, self.lowpass)

    def connect(self, consumer):
        # redirect the output
        self.lowpass.connect(consumer)

    def put(self, x):
        self.prefilter.put(x)
