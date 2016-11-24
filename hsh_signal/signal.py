from __future__ import division

import numpy as np


def bpm2hz(f_bpm):
    return f_bpm / 60.0


def hilbert_fc(x):
    """Hilbert Transform: recover analytic signal (float -> complex)"""

    # gnuradio.filter.firdes.hilbert(ntaps=65)
    hilbert_taps_65 = (0.0, -0.020952552556991577, 0.0, -0.022397557273507118, 0.0, -0.024056635797023773, 0.0, -0.02598116546869278, 0.0, -0.028240399435162544, 0.0, -0.030929960310459137, 0.0, -0.0341857448220253, 0.0, -0.03820759803056717, 0.0, -0.04330194741487503, 0.0, -0.04996378347277641, 0.0, -0.05904810503125191, 0.0, -0.07216990739107132, 0.0, -0.09278988093137741, 0.0, -0.1299058347940445, 0.0, -0.21650972962379456, 0.0, -0.6495291590690613, 0.0, 0.6495291590690613, 0.0, 0.21650972962379456, 0.0, 0.1299058347940445, 0.0, 0.09278988093137741, 0.0, 0.07216990739107132, 0.0, 0.05904810503125191, 0.0, 0.04996378347277641, 0.0, 0.04330194741487503, 0.0, 0.03820759803056717, 0.0, 0.0341857448220253, 0.0, 0.030929960310459137, 0.0, 0.028240399435162544, 0.0, 0.02598116546869278, 0.0, 0.024056635797023773, 0.0, 0.022397557273507118, 0.0, 0.020952552556991577, 0.0)

    #x = np.array(x)
    xi = np.convolve(x, hilbert_taps_65, mode='same')  # mode: for now, consistent with gnuradio
    return x + xi * 1j


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(2**int(m_i))


def filter_fft_ff(sig, taps):
    """
    Applies a filter to a signal, implementing the overlap-save method (chunk up the signal)
    see https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method

    This is equivalent to:  np.convolve(x, taps, mode='valid')  but more efficient for long signals
    """

    assert(len(sig) > len(taps))  # expect a long signal
    # ^ we could try reversing sig and taps in this case,
    # but the caller would be surprised to get a longer return value than expected

    h = taps
    M = len(taps)
    overlap = M - 1
    N = nextpow2(4*overlap)
    step_size = N - overlap
    # TODO: keep filter's FFT around (cache it) instead of recomputing it every time. e.g. currently 15 ms (out of 50 ms) just for Bandreject's H
    H = np.fft.fft(h, N)
    x = np.concatenate([sig, np.zeros(N)])  # end padding, so the last step_size batch will surely cover the end
    y = np.zeros(len(x))
    for pos in range(0, len(x)+1 - N, step_size):
        #print(pos)
        yt = np.fft.ifft(np.fft.fft(x[pos:pos+N], N) * H, N)
        y[pos:pos+step_size] = yt[M-1:N].real
    # cut back the end padding, and the overlap region where taps hang out of the signal
    #y = y[:-N-M//2]  # what is wrong with this line??
    y = y[0:len(sig)-len(taps)+1]

    # if latency is an issue, see "block convolver" (Bill Gardner)
    # here: http://dsp.stackexchange.com/questions/2537/do-fft-based-filtering-methods-add-intrinsic-latency-to-a-real-time-algorithm

    return y


# copied heartbeat_localmax() from heartshield-kivy-app/utils.py, added some more docstring
def localmax(d):
    """
    Calculate the local maxima of a (heartbeat) signal vector
    (based on script from https://github.com/compmem/ptsa/blob/master/ptsa/emd.py).

    :param d: signal vector
    :returns: boolean vector with same length as 'd', True values indicating local maxima
    """

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = np.r_[-np.inf,d,-np.inf]
    diffScore=np.diff(np.sign(np.diff(diffvec)))

    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions=np.r_[np.nonzero(d[0:-1]!=d[1:])[0],len(d)-1]
    runLengths = np.diff(np.r_[-1, runEndingPositions])
    runStarts=runEndingPositions-runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths>1]
    realRunStops = runEndingPositions[runLengths>1]
    realRunLengths = runLengths[runLengths>1]

    # save only the runs that are local maxima
    maxRuns=(diffScore[realRunStarts]==-1) & (diffScore[realRunStops]==-1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    maxRunMiddles=np.round(realRunStarts[maxRuns]+realRunLengths[maxRuns]/2.)-1

    # get all the maxima
    maxima=(diffScore==-2)
    maxima[maxRunMiddles.astype(np.int32)] = True

    return maxima


def localmax_pos(x):
    """:returns: list of tuple(position, peak): [(t, x_t), ...]"""
    m = localmax(x)
    return [(t, x_t) for (t, x_t) in enumerate(x) if m[t]]


def evenly_resample(times, series, target_fps=30.0):
    L = (times[-1] - times[0])*target_fps
    even_times = np.linspace(times[0], times[-1], L)
    series2 = np.interp(even_times, times, series)
    data = np.vstack((np.reshape(even_times, -1, 1), np.reshape(series2, -1, 1))).T
    return data

