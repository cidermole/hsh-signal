from numpy.linalg import norm
import numpy as np
from signal import slices, cross_corr
from heartseries import HeartSeries
from dtw import dtw
from scipy.stats.mstats import spearmanr
from sklearn.linear_model import TheilSenRegressor
from iter import pairwise
import time


def kurtosis(x):
    # https://en.wikipedia.org/wiki/Kurtosis#Sample_kurtosis
    xm = np.mean(x)
    num = np.mean((x - xm)**4)
    denom = np.mean((x - xm)**2)**2
    return num/denom - 3


def skewness(x):
    # https://en.wikipedia.org/wiki/Skewness#Sample_skewness
    xm = np.mean(x)
    n = float(len(x))
    num = np.mean((x - xm)**3)
    denom = (1.0/(n-1) * np.sum((x - xm)**2)) ** (3.0 / 2.0)
    return num/denom


def sig_slice(x, s, e):
    #return x[s:e]
    idxs = np.linspace(s, e, int(e-s), False)
    iidxs = np.arange(int(s), int(e))
    return np.interp(idxs, iidxs, x[iidxs])


def sig_resample(self, sig, L = None):
    """resample to length L."""
    t = np.linspace(0, len(sig), L, endpoint=False)
    assert len(t) == L
    return np.interp(t, np.arange(len(sig)), sig)


def sig_pad(sig, L):
    """pad up with zeros to length L on the right. trims if necessary."""
    if len(sig) > L:
        # trim
        return np.array(sig[0:L])
    return np.pad(sig, (0, L - len(sig)), mode='edge')


SLICE_FRONT = 0.2


def sqi_slices(sig, method='direct'):
    if method == 'fixed':
        slicez = []
        for i in range(len(sig.ibeats) - 1):
            # need to center window on the beat, just like the template
            s, e = sig.ibeats[i], sig.ibeats[i + 1]
            l = e - s
            assert l > 0, "ibeats must be strictly ascending"
            # s,e = max(s-l*0.1, 0), max(min(e-l*0.1, len(sig.x)), 0)
            s, e = max(s - l * SLICE_FRONT, 0), max(min(e, len(sig.x)), 0)

            if s != e:
                # plt.plot(sig.x[s:e])
                # rez = sig_pad(sig_slice(sig.x,s,e), L=L)
                rez = sig_slice(sig.x, s, e)
                """plt.plot(rez)
                plt.title(cross_corr(rez, sig.template))
                plt.show()
                """
                slicez.append(rez)  # (sig.x[s:e])
        # s = sig.ibeats[-1]
        # slicez.append(sig.resample(sig.x[int(s):int(s+sig.L*sig.fps)], L=30))  # surrogate length for last beat

        # not an np.array() since the slice lengths are different!
        # use sqi_remove_ibi_outliers() to pad/trim the lengths.
        return slicez

    elif method == 'variable':
        # to do: unused. sunset this! (post-processing is written for method='fixed')

        slicez = []
        for i in range(len(sig.ibeats)-1):
            # need to center window on the beat, just like the template
            s,e = sig.ibeats[i], sig.ibeats[i+1]
            l = e-s
            s,e = max(s-l/2., 0), min(e, len(sig.x))
            if s != e:
                #plt.plot(sig.x[s:e])
                rez = sig_resample(sig_slice(sig.x,s,e), L=30)
            """plt.plot(rez)
            plt.title(cross_corr(rez, sig.template))
            plt.show()
            """
            slicez.append(rez) #(sig.x[s:e])
        s = sig.ibeats[-1]
        #slicez.append(sig_resample(sig.x[int(s):int(s+sig.L*sig.fps)], L=30))  # surrogate length for last beat

        return slicez

    else:
        raise ValueError('slices() got unknown method={}'.format(method))


def sqi_remove_ibi_outliers(slicez, debug_errors=False, keep_all=False):
    slicez = np.array(slicez)
    # pad up to maximum length (within some reasonable limits)
    # note: when does this break? check IBI distribution, and if too skewed, there is other trouble.
    # (e.g. median IBI does not fit this assumed distribution? -> exit with an error message)
    lens_ok = np.array([len(s) for s in slicez])
    ibeat_ok = np.arange(len(slicez))
    #print 'lens_ok', len(lens_ok)

    #
    # IBI length limiter.
    #
    # Filters bad interval lengths for
    # 1) removal from the beatshape average
    # 2) beatshape window size (maximum reasonable IBI length)
    #

    # model limit assumption:
    # say 300 ms SDNN on a 800 ms RR -> 0.38
    rel_dev_limit = 0.38  #: add this relative amount of tolerance to IBI limits
    ibi_limit_perc = 0.1  #: as IBI limits, use this percentile on the IBI distribution, and add `rel_dev_limit`
    #len_min, len_max = np.median(lens_ok) * (1.0 - rel_dev_limit), np.median(lens_ok) * (1.0 + rel_dev_limit)

    # boundary-percentile limits
    len_max = np.percentile(lens_ok, 100.0 * (1.0 - ibi_limit_perc)) * (1.0 + rel_dev_limit)
    len_min = np.percentile(lens_ok, 100.0 * ibi_limit_perc) * (1.0 - rel_dev_limit)

    #print 'len_min=', len_min, 'len_max=', len_max
    if np.sum(lens_ok < len_min) > ibi_limit_perc * len(lens_ok):
        if debug_errors:
            print 'lens_ok=',lens_ok
            print 'len_min=', len_min, 'len_max=', len_max
            print np.sum(lens_ok < len_min), 'ibis below len_min is too much, max.', int(ibi_limit_perc * len(lens_ok))
        raise ValueError('while slicing: ibi model len_min limit assumption violated.')
    if np.sum(lens_ok > len_max) > ibi_limit_perc * len(lens_ok):
        if debug_errors:
            print 'lens_ok=',lens_ok
            print 'len_min=', len_min, 'len_max=', len_max
            print np.sum(lens_ok > len_max), 'ibis above len_max is too much, max.', int(ibi_limit_perc * len(lens_ok))
        raise ValueError('while slicing: ibi model len_max limit assumption violated.')

    model_len_max = np.percentile(lens_ok, 100.0 * (1.0 - ibi_limit_perc)) * (1.0 + rel_dev_limit)
    model_len_min = np.percentile(lens_ok, 100.0 * ibi_limit_perc) * (1.0 - rel_dev_limit)
    model_len_bottom = np.percentile(lens_ok, 100.0 * ibi_limit_perc)
    #print 'model_len_bottom', model_len_bottom
    #print 'model_len_min', model_len_min, 'model_len_max', model_len_max
    max_filter = np.where(lens_ok < model_len_max)[0]
    lens_ok, ibeat_ok = lens_ok[max_filter], ibeat_ok[max_filter]
    #print 'lens_ok', len(lens_ok)
    min_filter = np.where(lens_ok > model_len_min)[0]
    lens_ok, ibeat_ok = lens_ok[min_filter], ibeat_ok[min_filter]
    #print 'lens_ok', len(lens_ok)
    # Lmax = max(lens_ok)
    # model_len_bottom: almost all waveshapes should still be present for the mean calculation.
    Lmax = int(model_len_bottom)
    #print 'Lmax=', Lmax
    if keep_all: ibeat_ok = np.arange(len(slicez))
    slicez = np.array([sig_pad(s, L=Lmax) for s in slicez[ibeat_ok]])

    return slicez, ibeat_ok


def sqi_slice_norm(sl):
    return np.sqrt(np.sum(np.abs(sl) ** 2)) / len(sl)


def sqi_normalize_slices(slicez):
    slicez_norm = []
    for sl in slicez:
        #e = np.sqrt(np.sum(np.abs(sl)**2)) / len(sl)  # L2 norm is too heavy on fat-reflections?
        #e = np.sqrt(np.sum(np.abs(sl))) / len(sl)  # L1 norm
        e = sqi_slice_norm(sl)
        slicez_norm.append(sl / e)
    return np.array(slicez_norm)


def gauss(x, t_mu, t_sigma):
    a = 1.0 / (t_sigma * np.sqrt(2 * np.pi))
    y = a * np.exp(-0.5 * (x - t_mu)**2 / t_sigma**2)
    return y


def sqi_remove_shape_outliers(slicez, debug_errors=False, get_envelopes=False):
    #
    # Outlier beat shape removal.
    #
    # Removes outliers that would screw the average beatshape calculation later.

    # sometimes, beat swing amplitude changes a lot during a single minute (see e.g. 1500096525_***REMOVED***_meta.b)
    # avoid large beats being detected as outliers.
    slicez_norm = sqi_normalize_slices(slicez)
    #slicez_norm = np.array(slicez)

    # limiting lower and upper beat shape envelopes
    amplitude_limit_perc = 0.1
    ampl_viol_limit_perc = 0.1
    p_min, p_max = [100.0 * amplitude_limit_perc, 100.0 * (1.0 - amplitude_limit_perc)]
    s_min, s_max = [np.percentile(slicez_norm, p, axis=0) for p in [p_min, p_max]]
    #self.s_min, self.s_max = s_min, s_max

    # a good histogram for visualization of overall beat quality:
    """
    # auto adjust waveshape percentiles
    ccs = []
    for p in range(20):
        p_min, p_max = p, 100-p
        s_min, s_max = [np.percentile(slicez, p, axis=0) for p in [p_min, p_max]]
        ccs.append(cross_corr(s_min-np.mean(s_min), s_max-np.mean(s_max)))

    plt.plot(ccs)
    plt.show()
    """
    """
    import matplotlib.pyplot as plt
    plt.plot(s_min)
    plt.plot(s_max)
    plt.show()
    """

    # most value is given to deviations at the start, while deviations towards the end are discounted
    weighting = gauss(np.arange(len(s_min)), len(s_min)*SLICE_FRONT, len(s_min)*0.4)
    weighting[:int(len(s_min)*SLICE_FRONT)] = 0.0  # blank weighting the previous beat
    weighting = weighting / np.sum(weighting)  # * len(s_min)

    num_violations = []
    for sl in slicez_norm:
        # deviations are penalized with the MSE error from boundaries
        num_violations.append(np.sqrt(np.sum((sl < s_min) * (s_min - sl)**2 * weighting) + np.sum((sl > s_max) * (sl - s_max)**2 * weighting)))
    # TODO: this is not very principled! (always kills 10% of worst beats... but what if there are more?)
    # (we should instead remove ones where a lot of strong violation happens. not 10th percentile though)
    # TODO: how to define those bounds, then?
    violation_threshold = np.percentile(num_violations, (100.0 * (1.0 - ampl_viol_limit_perc)))
    #violation_threshold = 1.0  # TODO: why does this kill template_2???
    #print 'violation_threshold=', violation_threshold
    # good = most of the beat is within the shape envelopes
    igood = np.where(num_violations < violation_threshold)[0]

    ibad = np.array(sorted(list(set(np.arange(len(slicez))) - set(igood))))
    #print 'remove_shape_outliers ibad=', ibad

    if get_envelopes:
        return slicez[igood], igood, s_min, s_max, num_violations, violation_threshold
    else:
        return slicez[igood], igood


def sqi_copy_to_idxs(shape, length, idxs, amplitudes):
    d = np.zeros(length)
    # sqi_slices() gets from -SLICE_FRONT onwards
    idxs = np.array(idxs) + int(len(shape) * (0.5 - SLICE_FRONT)) + 1
    shape[:int(len(shape) * 0.2)] = 0.0  # clear that swing
    idxs = idxs[np.where(idxs < length)[0]]
    d[idxs] = amplitudes
    return np.convolve(d, shape, mode='same')


class QsqiError(RuntimeError): pass


class QsqiPPG(HeartSeries):
    """
    qSQI signal quality indicator.

    NOTE: code assumes that beats are timed on the feet of the beats.
    For example, windows are placed and weighted carefully in `sqi_remove_shape_outliers()`

    The shape of distributions of `num_violations` and `corrs` give an indication of overall measurement quality.

    inspired by:

    Li, Q., and G. D. Clifford. "Dynamic time warping and machine learning for signal quality assessment of pulsatile signals." Physiological measurement 33.9 (2012): 1491.
    http://www.robots.ox.ac.uk/~gari/papers/Li_and_Clifford_2012_IOP_Phys_Meas.pdf
    """

    CC_THR = 0.8    #: cross-correlation threshold for including beats in template 2
    BEAT_THR = 0.3  #: more beats thrown away? fail creating template 2

    def __init__(self, *args, **kwargs):
        init_template = kwargs.pop('init_template', True)
        self.debug_errors = kwargs.pop('debug_errors', False)
        self.lock_time = kwargs.pop('lock_time', None)
        super(QsqiPPG, self).__init__(*args, **kwargs)
        if init_template:
            self.init_template()

    def init_template(self):
        self.beat_template_1()
        self.template = self.beat_template_2()
        self.template_kurtosis = kurtosis(self.template)
        self.template_skewness = skewness(self.template)

    @staticmethod
    def from_heart_series(hs, init_template=True, lock_time=None, debug_errors=False):
        """
        caution! input must be one-sided, i.e. must NOT be DC free.
        (otherwise, correlation will fail to provide high enough values for CC_THR)
        """
        return QsqiPPG(hs.x, hs.ibeats, fps=hs.fps, lpad=hs.lpad, init_template=init_template, lock_time=lock_time, debug_errors=debug_errors)

    @staticmethod
    def from_series_data(signal, idx, fps=30, lpad=0, init_template=True):
        """
        caution! input must be one-sided, i.e. must NOT be DC free.
        (otherwise, correlation will fail to provide high enough values for CC_THR)
        """
        return QsqiPPG(signal, idx, fps=fps, lpad=lpad, init_template=init_template)

    def beat_template_1(self):
        self.L = np.median(np.diff(self.tbeats))
        slicez_1, islicez_1 = self.slices(method="fixed") #, hwin=int(self.L*self.fps/2.)))
        self.ibis_good = islicez_1
        template_1 = np.mean(slicez_1, axis=0)
        #print 'template_1', template_1
        slicez, islicez = self.slices(method="fixed", scrub=False)
        corrs = np.array([cross_corr(sl, template_1) for sl in slicez])
        self.slicez_1, self.islicez_1, self.template_1 = slicez_1, islicez_1, template_1
        self.slicez, self.islicez, self.corrs = slicez, islicez, corrs
        # TODO: penalize the correlation of overlong IBIs (since correlation only goes until the beat template ends)

        # debug: put shape envelopes
        self.env_min = sqi_copy_to_idxs(self.s_min, len(self.x), self.ibeats.astype(int), [sqi_slice_norm(sl) for sl in slicez])
        self.env_max = sqi_copy_to_idxs(self.s_max, len(self.x), self.ibeats.astype(int), [sqi_slice_norm(sl) for sl in slicez])

    def beat_template_2(self):
        slicez, template_1, corrs = self.slicez, self.template_1, self.corrs
        #print 'corrs', corrs
        #good_corrs = np.where(corrs > QsqiPPG.CC_THR)[0]
        good_corrs = [i for i, c in enumerate(corrs) if (c > QsqiPPG.CC_THR and i in self.ibis_good)]
        if len(good_corrs) < QsqiPPG.BEAT_THR * len(corrs):
            raise QsqiError('template 2 would keep only {} good beats of {} detected'.format(len(good_corrs), len(corrs)))
        template_2 = np.mean(slicez[good_corrs], axis=0)
        if len(template_2) == 0:
            raise QsqiError('template 2 length == 0, cowardly refusing to do signal quality analysis')

        # unimplemented idea from Slices.ipynb:
        # auto adjust waveshape percentiles
        # (ensure we are using only a range of waveshapes that actually correlate
        # between upper 90th percentile curve, and lower 10th percentile curve)

        return template_2

    def slices(self, method='direct', scrub=True):
        si = np.where(self.tbeats > self.lock_time)[0][0] if self.lock_time is not None else 0
        slicez = sqi_slices(self, method)[si:]
        igood = np.arange(si, len(slicez)+si) # attention: len(slicez) is shorter if we use [si:] above
        ig_orig = np.array(igood)

        if scrub:
            step1, good1 = sqi_remove_ibi_outliers(slicez, debug_errors=self.debug_errors)
            self.ibad1 = np.array(sorted(list(set(ig_orig) - set(good1))))
            igood = igood[good1]
            self.igood1 = np.array(igood)
            step2, good2, self.s_min, self.s_max, self.num_violations, self.violation_threshold = sqi_remove_shape_outliers(step1, debug_errors=self.debug_errors, get_envelopes=True)
            self.ibad2 = np.array(sorted(list(set(igood) - set(good2))))
            igood = igood[good2]
            assert len(step2) == len(igood)
        else:
            step2, good1 = sqi_remove_ibi_outliers(slicez, debug_errors=self.debug_errors, keep_all=True)
            igood = igood[good1]
            #step2 = np.array(slicez)
            #igood = np.arange(len(slicez))

        # for debugging: mark which areas have been scrubbed
        sig_good = np.ones(len(self.x))
        ibad = np.array(sorted(list(set(np.arange(len(slicez))) - set(igood))))
        if len(ibad):
            for s, e in np.array(pairwise(self.ibeats))[ibad]:
                s, e = max(int(s), 0), min(int(e), len(self.x))
                sig_good[s:e] *= 0

        self.sig_good = sig_good
        #self.ibis_good = igood

        return step2, igood

    def sqi1(self):
        """direct matching (fiducial + length L template correlation)"""
        # nb. slight difference: we are centering the window on the beat, while Li et al
        slicez, islicez = self.slices(method='fixed')
        corrs = np.array([cross_corr(sl, self.template) if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    def sqi2(self):
        """linear resampling (between two fiducials up to length L, correlation)"""
        slicez, islicez = self.slices(method='variable')
        L = len(self.template)
        corrs = np.array([(cross_corr(sig_resample(sl, L), self.template) if len(sl) else 0.0) for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    def dtw_resample(self, sl):
        """TODO: slooow. and does not use the metric in the paper."""
        # downsample again, to save some CPU.
        sx = sl[::10].reshape((-1,1))
        sy = self.template[::10].reshape((-1,1))
        dist, cost, acc, path = dtw(sx, sy, dist=lambda x, y: norm(x - y, ord=1))
        return sx[path[0]], sy[path[1]]

    def sqi3(self):
        """DTW resampling (resampling to length L and correlation)"""
        slicez, islicez = self.slices(method='variable')
        corrs = np.array([cross_corr(*self.dtw_resample(sl)) if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    def kurtosis(self):
        # unclear if 'fixed' or 'variable' is any better, could not just eyeball.
        slicez, islicez = self.slices(method='fixed')
        return np.array([kurtosis(sl) if len(sl) else 0.0 for sl in slicez])

    def skewness(self):
        slicez, islicez = self.slices(method='fixed')
        return np.array([skewness(sl) if len(sl) else 0.0 for sl in slicez])

    def spearman(self):
        slicez, islicez = self.slices(method='variable')
        #corrs = np.array([spearmanr(*self.dtw_resample(sl))[0] if len(sl) else 0.0 for sl in slicez])
        corrs = np.nan_to_num([spearmanr(*self.dtw_resample(sl))[0] if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    #def sqi4(self):
    #    """SQI based on Kurtosis."""

    def plot(self, plotter=None, dt=0.0, **kwargs):
        """debug plot"""

        import matplotlib.pyplot as plt

        plotter = plt if plotter is None else plotter

        def fmt(n):
            s = '%.02f' % n
            return s[1:] if s[0] == '0' else '.99'

        for ii, i in enumerate(self.islicez):
            # for ii, i in enumerate(np.arange(len(ppgz.ibeats) - 1)):
            bl = np.percentile(self.x, 90) * 0.3
            ym = 0 if ii % 2 == 0 else bl
            is_good = (self.corrs[ii] > QsqiPPG.CC_THR) and i in self.ibis_good
            plotter.text(self.tbeats[i], ym + bl, fmt(self.corrs[ii]), color='g' if is_good else 'r')

        for ii, i in enumerate(self.igood1):
            # for ii, i in enumerate(np.arange(len(ppgz.ibeats) - 1)):
            bl = np.percentile(self.x, 90) * 0.3
            ym = 0 if ii % 2 == 0 else bl
            is_good = (self.num_violations[ii] < self.violation_threshold)
            plotter.text(self.tbeats[i], ym - 5*bl, '%.2f'%(self.num_violations[ii]), color='gray' if is_good else 'm')

        #plotter.plot(self.t, self.env_min, c='g')
        #plotter.plot(self.t, self.env_max, c='r')
        plotter.fill_between(self.t, self.env_min, self.env_max, facecolor='lightgray')

        x_good = self.x[np.where(self.t > (self.lock_time if self.lock_time is not None else 5.0))[0]]
        swing = np.max(np.abs([np.percentile(x_good, 5), np.percentile(x_good, 95)])) * 2.5
        if hasattr(plotter, 'set_ylim'):
            plotter.set_ylim([-swing, swing*0.1])
        else:
            plotter.ylim([-swing, swing*0.1])
        super(QsqiPPG, self).plot(plotter=plotter, dt=dt, **kwargs)

class BeatQuality(QsqiPPG):
    """
    Beat quality indicator.

    Quantifies downslope integrity and beat placement,
    and flags anomalies for removal or redetection
    """

    # global parameters. should not bee to sensitive. real "meat" lies in anomaly detection
    ACCEPTED_DEVIATION_PERCENTAGE = 10 # how much a downslope point may deviate from linearly regressed downslope
    MINIMUM_LINEARITY = 0.75  # minimum acceptable "linearity", i.e. fraction of downslope "close" to downslope (see BeatQuality.ACCEPTED_DEVIATION_PERCENTAGE)
    MINIMUM_R2 = 0.75 # minimum acceptable fit of downslope to linear regression

    # outlier detection param - THIS ONE SHOULD BE VALIDATED USING A LARGE DATASET
    OUTLIER_THRESHOLD = 7

    VERBOSE = True

    def __init__(self, *args, **kwargs):
        super(QsqiPPG, self).__init__(*args, **kwargs)
        tt = time.time()
        self.template = self.beat_template()
        tt = time.time() - tt
        self.template_kurtosis = kurtosis(self.template)
        self.template_skewness = skewness(self.template)

        bt = time.time()
        self.beat_quality = self.sqi2()
        self.beat_outliers = self.detect_beat_outliers()
        self.beat_outliers[self.beat_quality[:len(self.beat_outliers)] < BeatQuality.BEAT_THR] = True
        bt = time.time() - bt

        if BeatQuality.VERBOSE:
            print "beat template found in",tt
            print "outliers found in", bt
            print len(np.where(self.beat_quality < BeatQuality.BEAT_THR)[0]), "bad SQ", self.beat_quality

    @staticmethod
    def from_heart_series(hs):
        return BeatQuality(hs.x, hs.ibeats, fps=hs.fps, lpad=hs.lpad)


    @staticmethod
    def from_series_data(signal, idx, fps=30, lpad=0):
        return BeatQuality(signal, idx, fps=fps, lpad=lpad)

    @staticmethod
    def tiny_outlier_detector(values, threshold=OUTLIER_THRESHOLD):
        # idea by Iglewicz and Hoaglin (1993), summarized in NIST/SEMATECH e-Handbook of Statistical Methods
        # works for tiny sample sizes
        outlierscore = np.abs(0.6745 * (values - np.median(values)) / np.median(np.abs(values - np.median(values))))
        if np.any(outlierscore>threshold):
            print outlierscore
        return np.where(outlierscore > threshold)[0]

    def detect_beat_outliers(self):
        beat_outliers = np.array([False] * len(self.ibeats), dtype=bool)

        descriptors = []
        for i in range(len(self.ibeats)):
            ok_slope_length, ok_slope_angle, beat_downslope_orthogonal_distance, beat_downslope_peak_distance, iscrap = self.quantify_beat(i)
            if iscrap:
                beat_outliers[i] = True
            else:
                descriptors.append([ok_slope_length, ok_slope_angle, beat_downslope_orthogonal_distance, beat_downslope_peak_distance]) # track everything
                #descriptors.append([beat_downslope_orthogonal_distance, beat_downslope_peak_distance]) # do NOT track slope lengths and angles (allows for physiological changes)
        descriptors = np.array(descriptors)

        for d in range(descriptors.shape[1]):
            outlier_indices = BeatQuality.tiny_outlier_detector(descriptors[:, d])
            if len(outlier_indices) > 0:
                if BeatQuality.VERBOSE:
                    print ["ok_slope_length", "ok_slope_angle", "beat_downslope_orthogonal_distance", "beat_downslope_peak_distance"][d],\
                    "anomalies detected: ",len(outlier_indices)
                beat_outliers[outlier_indices] = True

        return beat_outliers


    def quantify_beat(self, beatnumber):
        beatindex = self.ibeats[beatnumber]
        # approx expected ibi
        meanibi = np.mean(np.diff(self.tbeats))
        # downslope is less than half of full beat. look for peaks on either side
        downslopewindow = int((meanibi / 2.5) * self.fps)
        # pick preceding maximum
        try:
            maxindex = np.where(heartbeat_localmax(self.x[(beatindex - downslopewindow):beatindex]))[0][-1]
        except:
            maxindex = np.argmax(self.x[(beatindex - downslopewindow):beatindex])
        peaki = beatindex - downslopewindow + maxindex
        # double check we didn't go beyond prev. beat
        if beatnumber > 0 and peaki <= self.ibeats[beatnumber - 1]:
            peaki = self.ibeats[beatnumber - 1] + downslopewindow + np.argmax(self.x[(self.ibeats[beatnumber - 1] + downslopewindow):beatindex])
        # pick succeeding minimum
        troughi = beatindex + np.argmin(self.x[beatindex:(beatindex + downslopewindow)])
        # double check we didn't go beyond next beat
        if beatnumber < len(self.ibeats) - 1 and troughi >= self.ibeats[beatnumber + 1]:
            troughi = beatindex + np.argmin(self.x[beatindex:(self.ibeats[beatnumber + 1] - 1)])
        # robust regression on downslope
        downslopemodel = TheilSenRegressor().fit(self.t[peaki:troughi].reshape(-1, 1), self.x[peaki:troughi])
        r2 = downslopemodel.score(self.t[peaki:troughi].reshape(-1, 1), self.x[peaki:troughi])
        # count which points are close enough to prediction
        predicted_downslope = downslopemodel.predict(self.t[peaki:troughi].reshape(-1, 1))
        amplitude = self.x[peaki] - self.x[troughi]
        m, k = downslopemodel.coef_[0], downslopemodel.intercept_
        point_to_line_distances = np.abs(k + m * self.t[peaki:troughi] - self.x[peaki:troughi]) / np.sqrt(1 + m * m)
        point_to_line_distance_percentages = 100.0 / amplitude * point_to_line_distances
        ok_points = np.where(point_to_line_distance_percentages < BeatQuality.ACCEPTED_DEVIATION_PERCENTAGE)[0]
        fraction_acceptable = 1.0 / (troughi - peaki) * len(ok_points)
        # numerically characterize non-crap portion of the slope
        ok_slope_length = fraction_acceptable * np.sqrt((troughi - peaki) ** 2 + (self.x[peaki] - self.x[troughi]) ** 2)
        ok_slope_angle = np.arctan(downslopemodel.coef_[0])
        # numerically characterize beat placement
        beat_downslope_orthogonal_distance = 0 if ok_slope_length == 0 else 1.0 / ok_slope_length * (
        np.abs(k + m * self.t[beatindex] - self.x[beatindex]) / np.sqrt(1 + m * m))
        beat_downslope_peak_distance = 0 if ok_slope_length == 0 else 1.0 / ok_slope_length * np.sqrt(
            (beatindex - peaki) ** 2 + (self.x[peaki] - self.x[beatindex]) ** 2)

        # check if certain to be bad fit
        iscrap = False
        if np.abs(r2) < BeatQuality.MINIMUM_R2 or fraction_acceptable < BeatQuality.MINIMUM_LINEARITY:
            print "crap! ",beatnumber,r2, fraction_acceptable
            iscrap = True

        return ok_slope_length, ok_slope_angle, beat_downslope_orthogonal_distance, beat_downslope_peak_distance, iscrap