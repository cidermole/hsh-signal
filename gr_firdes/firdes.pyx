# distutils: language = c++

"""
Finite Impulse Response (FIR) filter design functions.

see http://gnuradio.org/doc/doxygen/classgr_1_1filter_1_1firdes.html
"""

from firdes cimport high_pass_2 as c_high_pass_2
from firdes cimport low_pass_2 as c_low_pass_2
from firdes cimport band_pass_2 as c_band_pass_2
from firdes cimport band_reject_2 as c_band_reject_2
from firdes cimport hilbert as c_hilbert
import numpy as np


cdef class WinType:
    WIN_NONE = -1           #: don't use a window
    WIN_HAMMING = 0         #: Hamming window; max attenuation 53 dB
    WIN_HANN = 1            #: Hann window; max attenuation 44 dB
    WIN_BLACKMAN = 2        #: Blackman window; max attenuation 74 dB
    WIN_RECTANGULAR = 3     #: Basic rectangular window
    WIN_KAISER = 4          #: Kaiser window; max attenuation a function of beta, google it
    WIN_BLACKMAN_hARRIS = 5 #: Blackman-harris window
    WIN_BLACKMAN_HARRIS = 5 #: alias to WIN_BLACKMAN_hARRIS for capitalization consistency
    WIN_BARTLETT = 6        #: Barlett (triangular) window
    WIN_FLATTOP = 7         #: flat top window; useful in FFTs


def high_pass_2(gain, sampling_freq, cutoff_freq, transition_width, attenuation_dB, window = WinType.WIN_HAMMING, beta = 6.76):
    """
    Use "window method" to design a high-pass FIR filter.  The
    normalized width of the transition band and the required stop band
    attenuation is what sets the number of taps required.  Narrow --> more
    taps More attenuation --> more taps. The window type determines
    maximum attentuation and passband ripple.

    :param gain                overall gain of filter (typically 1.0)
    :param sampling_freq       sampling freq (Hz)
    :param cutoff_freq         beginning of transition band (Hz)
    :param transition_width    width of transition band (Hz)
    :param attenuation_dB      required stopband attenuation (dB)
    :param window              one of firdes::win_type
    :param beta		           parameter for Kaiser window

    :return a numpy.array() of filter taps (aka h[x], aka impulse response)
    """
    cdef vector[float] c_taps = c_high_pass_2(gain, sampling_freq, cutoff_freq, transition_width, attenuation_dB, window, beta)
    cdef char *c_str = <char *> &c_taps[0]
    cdef Py_ssize_t length = c_taps.size() * sizeof(float)
    np_taps = np.fromstring(c_str[0:length], dtype=np.float32)
    return np_taps


def low_pass_2(gain, sampling_freq, cutoff_freq, transition_width, attenuation_dB, window = WinType.WIN_HAMMING, beta = 6.76):
    """
    Use "window method" to design a low-pass FIR filter.  The
    normalized width of the transition band and the required stop band
    attenuation is what sets the number of taps required.  Narrow --> more
    taps More attenuation --> more taps. The window type determines
    maximum attentuation and passband ripple.

    :param gain                overall gain of filter (typically 1.0)
    :param sampling_freq       sampling freq (Hz)
    :param cutoff_freq         beginning of transition band (Hz)
    :param transition_width    width of transition band (Hz)
    :param attenuation_dB      required stopband attenuation (dB)
    :param window              one of firdes::win_type
    :param beta		           parameter for Kaiser window

    :return a numpy.array() of filter taps (aka h[x], aka impulse response)
    """
    cdef vector[float] c_taps = c_low_pass_2(gain, sampling_freq, cutoff_freq, transition_width, attenuation_dB, window, beta)
    cdef char *c_str = <char *> &c_taps[0]
    cdef Py_ssize_t length = c_taps.size() * sizeof(float)
    np_taps = np.fromstring(c_str[0:length], dtype=np.float32)
    return np_taps


def band_pass_2(gain, sampling_freq, low_cutoff_freq, high_cutoff_freq, transition_width, attenuation_dB, window = WinType.WIN_HAMMING, beta = 6.76):
    """
    Use "window method" to design a band-pass FIR filter.  The
    normalized width of the transition band and the required stop band
    attenuation is what sets the number of taps required.  Narrow --> more
    taps More attenuation --> more taps. The window type determines
    maximum attentuation and passband ripple.

    :param gain                overall gain of filter (typically 1.0)
    :param sampling_freq       sampling freq (Hz)
    :param low_cutoff_freq     center of transition band (Hz)
    :param high_cutoff_freq    center of transition band (Hz)
    :param transition_width    width of transition band (Hz)
    :param attenuation_dB      required stopband attenuation (dB)
    :param window              one of firdes::win_type
    :param beta		           parameter for Kaiser window

    :return a numpy.array() of filter taps (aka h[x], aka impulse response)
    """
    cdef vector[float] c_taps = c_band_pass_2(gain, sampling_freq, low_cutoff_freq, high_cutoff_freq, transition_width, attenuation_dB, window, beta)
    cdef char *c_str = <char *> &c_taps[0]
    cdef Py_ssize_t length = c_taps.size() * sizeof(float)
    np_taps = np.fromstring(c_str[0:length], dtype=np.float32)
    return np_taps


def band_reject_2(gain, sampling_freq, low_cutoff_freq, high_cutoff_freq, transition_width, attenuation_dB, window = WinType.WIN_HAMMING, beta = 6.76):
    """
    Use "window method" to design a band-reject FIR filter.  The
    normalized width of the transition band and the required stop band
    attenuation is what sets the number of taps required.  Narrow --> more
    taps More attenuation --> more taps. The window type determines
    maximum attentuation and passband ripple.

    :param gain                overall gain of filter (typically 1.0)
    :param sampling_freq       sampling freq (Hz)
    :param low_cutoff_freq     center of transition band (Hz)
    :param high_cutoff_freq    center of transition band (Hz)
    :param transition_width    width of transition band (Hz)
    :param attenuation_dB      required stopband attenuation (dB)
    :param window              one of firdes::win_type
    :param beta		           parameter for Kaiser window

    :return a numpy.array() of filter taps (aka h[x], aka impulse response)
    """
    cdef vector[float] c_taps = c_band_reject_2(gain, sampling_freq, low_cutoff_freq, high_cutoff_freq, transition_width, attenuation_dB, window, beta)
    cdef char *c_str = <char *> &c_taps[0]
    cdef Py_ssize_t length = c_taps.size() * sizeof(float)
    np_taps = np.fromstring(c_str[0:length], dtype=np.float32)
    return np_taps


def hilbert(ntaps, window = WinType.WIN_RECTANGULAR, beta = 6.76):
    """
    design a Hilbert Transform Filter

    :param ntaps   number of taps, must be odd
    :param window  one kind of firdes::win_type
    :param beta	   parameter for Kaiser window

    :return a numpy.array() of filter taps (aka h[x], aka impulse response)
    """
    cdef vector[float] c_taps = c_hilbert(ntaps, window, beta)
    cdef char *c_str = <char *> &c_taps[0]
    cdef Py_ssize_t length = c_taps.size() * sizeof(float)
    np_taps = np.fromstring(c_str[0:length], dtype=np.float32)
    return np_taps
