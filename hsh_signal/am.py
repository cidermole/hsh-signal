from __future__ import division

from .filter import *


class _AMDemod(FilterBlock):
    def __init__(self, loop_bw, max_freq, min_freq, sampling_rate, ntaps=65):
        """
        real ---> |              _AM               | ---> real
                   -> Hilbert -> PLL -> Multiply ->
                              -------->

        You must use a low-pass afterwards, since this still contains frequency components from both sides.
        """
        super(_AMDemod, self).__init__()
        self.hilbert = Hilbert(ntaps)
        self.pll = PLL(loop_bw, max_freq, min_freq, sampling_rate)
        self.delay = self.hilbert.delay

    def batch(self, x):
        """batch-process an array and return array of output values"""
        analytic = self.hilbert.batch(x)
        carrier = self.pll.batch_vco(analytic)
        demod = -np.real(carrier * analytic)
        return demod


class AMFilter(SourceBlock):
    """Demodulates an AM audio signal."""
    def __init__(self, fps, low_cutoff_freq, high_cutoff_freq, transition_width, min_freq=None, max_freq=None, loop_bw=None):
        super(AMFilter, self).__init__()
        self.sampling_rate = fps

        if min_freq is None: min_freq = low_cutoff_freq
        if max_freq is None: max_freq = high_cutoff_freq
        if loop_bw is None: loop_bw = (max_freq - min_freq)

        self.prefilter = Bandpass(low_cutoff_freq=low_cutoff_freq, high_cutoff_freq=high_cutoff_freq, transition_width=transition_width, sampling_rate=self.sampling_rate)
        self.demod = _AMDemod(loop_bw=loop_bw, max_freq=max_freq, min_freq=min_freq, sampling_rate=self.sampling_rate)
        self.lowpass = Lowpass(cutoff_freq=100, transition_width=5, sampling_rate=self.sampling_rate)

        self.delay = self.prefilter.delay + self.demod.delay + self.lowpass.delay

        connect(self.prefilter, self.demod, self.lowpass)

    def connect(self, consumer):
        # redirect the output
        self.lowpass.connect(consumer)

    def put(self, x):
        self.prefilter.put(x)
