# distutils: language = c++

#from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.complex cimport complex
cimport numpy as np
import numpy as np
from math import pi

ctypedef np.complex64_t complex64_t


cdef extern from "<gnuradio/analog/pll_freqdet_cf.h>" namespace "gr::analog":
    ctypedef vector[const void*] gr_vector_const_void_star
    ctypedef vector[void*] gr_vector_void_star
    ctypedef complex[float] gr_complex

    cdef cppclass pll_freqdet_cf:
        int work(int noutput_items,
			      gr_vector_const_void_star &input_items,
			      gr_vector_void_star &output_items)

cdef extern from "<gnuradio/analog/pll_freqdet_cf.h>" namespace "gr::analog::pll_freqdet_cf":
    # static method from class pll_freqdet_cf
    #shared_ptr[pll_freqdet_cf] make(float loop_bw, float max_freq, float min_freq)
    pll_freqdet_cf *make(float loop_bw, float max_freq, float min_freq)

cdef class PLL:
    """
    Implements a PLL which locks to the input frequency and outputs an estimate of that frequency. Useful for FM Demod.

    Input stream 0: complex Output stream 0: float

    This PLL locks onto a [possibly noisy] reference carrier on the input and outputs an estimate of that frequency
    in radians per sample.

    The loop bandwidth determines the lock range and should be set around sampling_rate * [pi/200; 2pi/100].

    see http://gnuradio.org/doc/doxygen/classgr_1_1analog_1_1pll__freqdet__cf.html#details
    """

    cdef int _i
    #cdef shared_ptr[pll_freqdet_cf] _ptr
    cdef pll_freqdet_cf *_ptr

    def __cinit__(self, loop_bw, max_freq, min_freq, sampling_rate):
        """
        Make PLL block that outputs the tracked signal's frequency.

        :param loop_bw:  loop bandwidth, determines the lock range and should be set around sampling_rate * [pi/200; 2pi/100].
        :param max_freq: maximum frequency cap (Hz)
        :param min_freq: minimum frequency cap (Hz)
        :param sampling_rate: sampling rate (Hz)
        """
        # internal settings are in terms of radians per sample, not Hz.
        # see https://en.wikipedia.org/wiki/Normalized_frequency_%28unit%29
        self._ptr = make(loop_bw * 2.0*pi / sampling_rate, max_freq * 2.0*pi / sampling_rate, min_freq * 2.0*pi / sampling_rate)

    def __dealloc__(self):
        del self._ptr

    def filter_cf(self, x):
        """
        FM demodulator (complex -> float).
        :param x: array of complex numbers (analytic input signal)
        :returns: demodulated frequency values, as array of float numbers (I think these are in radians per sample -- normalize/convert back to Hz if required)
        """

        # wrap to the exact data types required by _work():
        pll_in = np.array(x, dtype=np.complex64)
        pll_out = np.zeros(len(pll_in), dtype=np.float32)
        self._work(pll_in, pll_out)
        return pll_out

    def _work(self, input_items, output_items):
        """
        Outputs the tracked signal's frequency.

        :param input_items: complex[float] array
        :param output_items: float array
        :return: len(output_items)

        Example:

        In [12]: input=np.array([1+1j,1+2j], dtype=np.complex64)
        In [13]: output=np.zeros(2, dtype=np.float32)
        In [15]: pll.work(input, output)
        """
        cdef complex64_t[:] input_view = input_items
        cdef float[:] output_view = output_items
        cdef vector[const void*] input
        cdef vector[void*] output
        input.push_back(&input_view[0])
        output.push_back(&output_view[0])
        #return self._ptr.get().work(len(output_items), input, output)
        return self._ptr.work(len(output_items), input, output)
