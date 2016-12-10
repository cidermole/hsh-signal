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
    def __init__(self, fps):
        super(AMFilter, self).__init__()
        self.sampling_rate = fps

        f_carrier = 10000

        self.prefilter = Bandpass(low_cutoff_freq=f_carrier-100, high_cutoff_freq=f_carrier+100, transition_width=20, sampling_rate=self.sampling_rate)
        self.demod = _AMDemod(loop_bw=100, max_freq=f_carrier+1000, min_freq=f_carrier-1000, sampling_rate=self.sampling_rate)
        self.lowpass = Lowpass(cutoff_freq=100, transition_width=5, sampling_rate=self.sampling_rate)

        self.delay = self.prefilter.delay + self.demod.delay + self.lowpass.delay

        connect(self.prefilter, self.demod, self.lowpass)

    def connect(self, consumer):
        # redirect the output
        self.lowpass.connect(consumer)

    def put(self, x):
        self.prefilter.put(x)
