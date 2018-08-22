from __future__ import division

import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.interpolate import interp1d


def bpm2hz(f_bpm):
    return f_bpm / 60.0
def hz2bpm(f_hz):
    return f_hz * 60.0


def gauss(x, t_mu, t_sigma):
    a = 1.0 / (t_sigma * np.sqrt(2 * np.pi))
    y = a * np.exp(-0.5 * (x - t_mu)**2 / t_sigma**2)
    return y


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


def grid_resample(times, series, target_fps=30.0):
    """like `evenly_resample()`, but with precise grid spacing through tail padding."""
    # add tail padding:
    times = np.hstack((times, [times[-1] + times[-1] - times[-2]]))
    series = np.hstack((series, [series[-1]]))
    # the trail padding above ensures precise grid spacing until the last sample.
    # Fractional index interpolation beyond the original end may leave a trailing constant value in the result output.

    # note: the original `evenly_resample()` result was never intended to be used on its own, decoupled from its timestamps.

    L = (times[-1] - times[0]) * target_fps
    even_times = np.linspace(times[0], times[-1], L, endpoint=False)
    series2 = np.interp(even_times, times, series)
    data = np.vstack((np.reshape(even_times, -1, 1), np.reshape(series2, -1, 1))).T
    return data

    #  Example 3:1 extrapolation from [a, b, c]:
    #
    #  [ a . . b . . c . . (c) ]
    #    _ _ _ _ _ _ _ _ _
    #
    #                  ^ these two interpolation points in the tail are constant, have no real support.


def highpass(signal, fps, cf=0.5, tw=0.4):
    from gr_firdes import firdes
    cutoff_freq = cf
    transition_width = tw
    taps = firdes.high_pass_2(1.0, fps, cutoff_freq, transition_width, 60.0)
    if len(signal) == 0: return np.array([])  # workaround failing np.pad([])
    return np.convolve(np.pad(signal, (len(taps)//2,len(taps)//2), 'edge'), taps, mode='valid')

def lowpass(signal, fps, cf=3.0, tw=0.4):
    from gr_firdes import firdes
    cutoff_freq = cf
    transition_width = tw
    taps = firdes.low_pass_2(1.0, fps, cutoff_freq, transition_width, 60.0)
    if len(signal) == 0: return np.array([])  # workaround failing np.pad([])
    return np.convolve(np.pad(signal, (len(taps)//2,len(taps)//2), 'edge'), taps, mode='valid')

def highpass_fft(signal, fps, cf=0.5, tw=0.4):
    from filter import Highpass, apply_filter
    if len(signal) == 0: return np.array([])
    return apply_filter(signal, Highpass(cf, tw, fps))

def lowpass_fft(signal, fps, cf=3.0, tw=0.4):
    from filter import Lowpass, apply_filter
    if len(signal) == 0: return np.array([])
    return apply_filter(signal, Lowpass(cf, tw, fps))

def cross_corr(x, y):
    """normalized cross-correlation of the two signals of same length"""
    return np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))

def slices(arr, loc, hwin):
    """Get slices from arr +/- hwin around indexes loc."""
    ret = []
    arrp = np.pad(arr, (hwin, hwin), mode='constant')  # zero-pad
    locp = loc + hwin
    for lp in locp:
        ret.append(arrp[lp-hwin:lp+hwin+1])
    return ret

def localmax_climb(arr, loc, hwin):
    """Climb from loc to the local maxima, up to hwin to the left/right."""
    # TODO: should be called localmax_win, since it's not really climbing but looking in a window.
    new_loc = []
    arrp = np.pad(arr, (hwin, hwin), mode='constant')  # zero-pad
    locp = loc + hwin
    for lp in locp:
        im = (lp-hwin) + np.argmax(arrp[lp-hwin:lp+hwin+1]) - hwin
        im = min(max(im, 0), len(arr)-1)  # clip
        new_loc.append(im)
    return np.array(new_loc)


def cubic_resample(series, fps_old=30, fps_new=300):
    fps_old, fps_new = float(fps_old), float(fps_new)
    t = np.linspace(0.0, len(series)/fps_old, len(series), endpoint=False)
    yinterp = interpolate.UnivariateSpline(t, series, s=0)
    tnew = np.arange(0.0, len(series)/fps_old, 1.0/fps_new)
    return np.clip(yinterp(tnew), a_min=np.min(series), a_max=np.max(series))


def win_max(sig, hwin):
    out = np.zeros(len(sig))
    for i in range(len(sig)):
        s,e = max(i-hwin, 0), min(i+hwin+1, len(sig))
        out[i] = np.max(sig[s:e])
    return out


def cwt_lowpass(x, fps, cf):
    num_wavelets = int(fps / cf)
    widths = np.arange(1, num_wavelets)
    cwt_mat = signal.cwt(x, signal.ricker, widths)
    c = np.mean(cwt_mat, axis=0)
    # renormalize
    orig_e, c_e = np.sqrt(np.sum(x*x)), np.sqrt(np.sum(c*c))
    return c * (orig_e / c_e)


def even_smooth(ii, x, length, fps, cf=2.0, tw=1.0):
    """
    Turn an intermittently sampled signal into an evenly sampled one of `length`.
    Similar to `evenly_resample()` except this is index-based and smoothes afterwards.

    :param ii   indices for the samples in `x` (may be fractional)
    :param x    values sampled at uneven indices `ii`
    :param fps  sampling rate
    :param cf   cutoff frequency for subsequent low-pass filter
    :param tw   transition width for subsequent low-pass filter
    """
    # add bounds (constant value approximation)
    assert len(x) > 0
    ii = np.insert(ii, 0, -1)
    ii = np.insert(ii, len(ii), length)
    # + 1e-3 * std ... avoid interpolate division by zero?
    # x = np.insert(x, 0, x[0] + 1e-3 * np.std(x))
    # x = np.insert(x, len(x), x[-1] + 1e-3 * np.std(x))
    x = np.insert(x, 0, x[0])
    x = np.insert(x, len(x), x[-1])

    #print 'insert x[0]=', x[0], 'x[-1]=', x[-1]

    # interpolate it -> evenly sampled
    func = interp1d(ii, x, kind='linear')
    ix = func(np.arange(length))

    #print 'resulting ix[0]=', ix[0], 'ix[1]=', ix[1], 'ix[-1]=', ix[-1]

    # smooth it
    return lowpass(ix, fps=fps, cf=cf, tw=tw)


def seek_left_localmax(x, idxs, fps, win_len=0.3):
    """seek to the left of SSF-detected peaks `idxs`, to find the actual feet (localmax)"""
    ifeet = []
    imax = np.where(localmax(x))[0]
    for i in idxs:
        ii = np.where(((i - imax) >= 0) & ((i - imax) <= int(fps * win_len)))[0]
        if len(ii) == 0: continue
        # find the first localmax on the left (not necessarily the tallest)
        ifeet.append(imax[ii][-1])
    ifeet = np.array(ifeet)
    return ifeet


def localmax_interp(x, idxs, hwin_size=None):
    """
    Turn local maxima from integers to floats.
    :param x     signal
    :param idxs  integer indices of local maxima
    :param hwin_size  optional integer window size. if given, do a localmax_climb() before to find the exact maxima so the result will be correct with rough estimates.
    :returns float indices into `x`
    """
    idxs = localmax_climb(x, idxs, hwin_size) if hwin_size is not None else idxs
    assert np.all(idxs < len(x))
    assert np.all(idxs >= 0)
    der = np.pad(np.diff(x), (1, 1), 'constant')
    new_idxs = []
    for i in idxs:
        xp, fp = der[i:i+2], np.array([i,i+1])
        ii = interp1d(xp, fp, bounds_error=False, fill_value=i)([0])
        new_idxs.append(max(min(ii, i+1, len(x)-1), i))
    return np.array(new_idxs)


#
# Indexing tools
#


def dirac(length, idxs, dtype=bool):
    """:returns an array with the specified `idxs` set to 1."""
    arr = np.zeros(length, dtype=dtype)
    idxs = np.array(idxs)
    arr[idxs] = 1.0
    return arr


def cohesive_ranges(idxs):
    if sorted(list(idxs)) != list(idxs):
        raise ValueError('idxs must be sorted')
    if len(idxs) == 0:
        return []

    ranges = []
    si, pi = idxs[0], idxs[0]
    for i in idxs[1:]:
        pi += 1
        if i != pi:
            ranges.append((si, pi))
            si = i
            pi = i
    ranges.append((si, pi + 1))

    return ranges
