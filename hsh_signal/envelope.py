import numpy as np
from .signal import cross_corr


def envelopes_at_perc(slicez, perc=10):
    """:returns the lower and upper percentile envelopes of individual beat window slices."""
    # slz = np.array(slicez)
    slz = slicez
    le, ue = np.percentile(slz, perc, axis=0), np.percentile(slz, (100 - perc), axis=0)
    return le, ue


def envelopes_corr(slicez):
    """envelope similarity -- an 1d measure of general recording quality. useful for plotting."""

    # p=45 means only 10% of all beats will be used -- that would be a very bad signal indeed.
    ps, xcs = [], []
    for p in np.arange(1, 50, 1):
        le, ue = envelopes_at_perc(slicez, p)
        # non-zero indices: where zero-padding is not active. (would otherwise mess with mean removal)
        nzi = np.where((le != 0) | (ue != 0))[0]
        xc = cross_corr(le[nzi] - np.mean(le[nzi]), ue[nzi] - np.mean(ue[nzi]))
        ps.append(p)
        xcs.append(xc)
    return np.array(ps), np.array(xcs)


def envelopes_perc_threshold(slicez, corr_threshold=0.8):
    """
    :param slicez  2d np.array of slices, each zero-padded to the same length (for ECG, with centered R peak)
    :returns the smallest percentile such that envelopes_at_perc() returns envelopes which are still well correlated.
    """
    #ENVELOPE_CORR_THRESHOLD = 0.8

    slicez = np.array(slicez)
    ps, xcs = envelopes_corr(slicez)
    igood = np.where(xcs > corr_threshold)[0]
    if len(igood) == 0:
        return 50  # both le,ue == exactly the median.
    return ps[igood[0]]

"""

from hsh_signal.ecg import beatdet_ecg

ecg = beatdet_ecg(sig0.slice(slice(int(1500*fps), int(1510*fps))))

# ecg = <hsh_signal.heartseries.HeartSeries at 0x7f21b89e4a50>


from hsh_signal.quality import sqi_slices, sig_pad

slicez = sqi_slices(ecg, method='fixed', slice_front=0.5, slice_back=-0.5)
L = max([len(sl) for sl in slicez])
padded_slicez = [sig_pad(sl, L, side='center', mode='constant') for sl in slicez]

ps, xcs = envelopes_corr(padded_slicez)

plt.plot(ps, xcs)
plt.show()
"""



def beat_penalty(sl, le, ue, mb, debug=False):
    # slice, lower envelope, upper envelope, median beat
    out_of_envelope = (sl > ue) | (sl < le)
    have_envelope = (le != 0) | (ue != 0)

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(ue, c='r')
        plt.plot(sl, c='k')
        plt.plot(le, c='g')
        plt.plot(out_of_envelope * 0.2 + 0.4, c='m')
        plt.plot(have_envelope * 0.2 + 0.3, c='g')

    # frac_invalid = len(np.where(invalid)[0]) / float(len(np.where(have_envelope)[0]))
    # << would be another SQI metric on its own

    # where outside of envelopes, penalize with MSE vs. the median beat
    penalty = np.sqrt(np.mean(out_of_envelope * (sl - mb) ** 2))
    median_energy = np.sqrt(np.mean(mb ** 2))
    return penalty / median_energy


def beat_penalty_threshold(le, ue, mb, debug=False):
    # idea: set threshold for noisy beats where a beat always eps above the normal percentile envelope
    # would be penalized.
    lp, up = beat_penalty(le - 1e-6, le, ue, mb), beat_penalty(ue + 1e-6, le, ue, mb)
    # these two should be similar, if distribution is not too skewed
    if debug: print lp, up
    return np.mean([lp, up])


"""
# padded_slicez, see above

perc = envelopes_perc_threshold(padded_slicez)
le, ue = envelopes_at_perc(padded_slicez, perc)
mb = np.median(padded_slicez, axis=0)

bpt = beat_penalty_threshold(le, ue, mb)
"""

# see Untitled110.ipynb
