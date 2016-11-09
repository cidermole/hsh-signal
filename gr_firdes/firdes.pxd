from libcpp.vector cimport vector

cdef extern from "<gnuradio/filter/firdes.h>" namespace "gr::filter::firdes":
    cpdef enum win_type:
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

    # static methods from class firdes
    vector[float] high_pass_2(double gain, double sampling_freq, double cutoff_freq, double transition_width, double attenuation_dB, win_type window, double beta)
    vector[float] low_pass_2(double gain, double sampling_freq, double cutoff_freq, double transition_width, double attenuation_dB, win_type window, double beta)
    vector[float] band_reject_2(double gain, double sampling_freq, double low_cutoff_freq, double high_cutoff_freq, double transition_width, double attenuation_dB, win_type window, double beta)
