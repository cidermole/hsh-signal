from numpy.linalg import norm
import numpy as np
from signal import slices, cross_corr
from heartseries import HeartSeries
from dtw import dtw
from scipy.stats.mstats import spearmanr
from sklearn.linear_model import TheilSenRegressor
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


class QsqiError(RuntimeError): pass

class QsqiPPG(HeartSeries):
    """
    qSQI signal quality indicator.

    Li, Q., and G. D. Clifford. "Dynamic time warping and machine learning for signal quality assessment of pulsatile signals." Physiological measurement 33.9 (2012): 1491.
    http://www.robots.ox.ac.uk/~gari/papers/Li_and_Clifford_2012_IOP_Phys_Meas.pdf
    """

    CC_THR = 0.8    #: cross-correlation threshold for including beats in template 2
    BEAT_THR = 0.3  #: more beats thrown away? fail creating template 2

    def __init__(self, *args, **kwargs):
        super(QsqiPPG, self).__init__(*args, **kwargs)
        self.template = self.beat_template()
        self.template_kurtosis = kurtosis(self.template)
        self.template_skewness = skewness(self.template)

    @staticmethod
    def from_heart_series(hs):
        return QsqiPPG(hs.x, hs.ibeats, fps=hs.fps, lpad=hs.lpad)

    @staticmethod
    def from_series_data(signal, idx, fps=30, lpad=0):
        return QsqiPPG(signal, idx, fps=fps, lpad=lpad)

    def beat_template(self):
        self.L = np.median(np.diff(self.tbeats))
        slicez = np.array(self.slices(method="variable")) #, hwin=int(self.L*self.fps/2.)))
        template_1 = np.mean(slicez, axis=0)
        corrs = np.array([cross_corr(sl, template_1) for sl in slicez])
        good_corrs = np.where(corrs > QsqiPPG.CC_THR)[0]
        if len(good_corrs) < QsqiPPG.BEAT_THR * len(corrs):
            raise QsqiError('template 2 would keep only {} good beats of {} detected'.format(len(good_corrs), len(corrs)))
        template_2 = np.mean(slicez[good_corrs], axis=0)
        if len(template_2) == 0:
            raise QsqiError('template 2 length == 0, cowardly refusing to do signal quality analysis')
        return template_2

    def slice(self, s, e):
        #return self.x[s:e]
        idxs = np.linspace(s, e, int(e-s), False)
        iidxs = np.arange(int(s), int(e))
        return np.interp(idxs, iidxs, self.x[iidxs])

    def slices(self, method='direct', L=30):
        if method == 'fixed':
            return np.array(slices(self.x, self.ibeats, hwin=int(self.L*self.fps/2.)))
        elif method == 'variable':
            slicez = []
            for i in range(len(self.ibeats)-1):
                # need to center window on the beat, just like the template
                s,e = self.ibeats[i], self.ibeats[i+1]
                l = e-s
                s,e = max(s-l/2., 0), min(e, len(self.x))
                if s != e:
                    #plt.plot(self.x[s:e])
                    rez = self.resample(self.slice(s,e), L=L)
                """plt.plot(rez)
                plt.title(cross_corr(rez, self.template))
                plt.show()
                """
                slicez.append(rez) #(self.x[s:e])
            s = self.ibeats[-1]
            #slicez.append(self.resample(self.x[int(s):int(s+self.L*self.fps)], L=30))  # surrogate length for last beat
            return slicez
        else:
            raise ValueError('slices() got unknown method={}'.format(method))

    def sqi1(self):
        """direct matching (fiducial + length L template correlation)"""
        # nb. slight difference: we are centering the window on the beat, while Li et al
        slicez = self.slices(method='fixed')
        corrs = np.array([cross_corr(sl, self.template) if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    def resample(self, sig, L = None):
        """resample to length L sec."""
        if L == None:
            L = len(self.template)
        #t = np.linspace(0, len(sig), int(self.L*self.fps), endpoint=False)
        t = np.linspace(0, len(sig), L, endpoint=False)
        # 2*int(self.L*self.fps/2.)+1 or, len(self.template)
        assert len(t) == L
        return np.interp(t, np.arange(len(sig)), sig)

    def sqi2(self):
        """linear resampling (between two fiducials up to length L, correlation)"""
        slicez = self.slices(method='variable')
        corrs = np.array([(cross_corr(self.resample(sl), self.template) if len(sl) else 0.0) for sl in slicez])
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
        slicez = self.slices(method='variable')
        corrs = np.array([cross_corr(*self.dtw_resample(sl)) if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    def kurtosis(self):
        # unclear if 'fixed' or 'variable' is any better, could not just eyeball.
        slicez = self.slices(method='fixed')
        return np.array([kurtosis(sl) if len(sl) else 0.0 for sl in slicez])

    def skewness(self):
        slicez = self.slices(method='fixed')
        return np.array([skewness(sl) if len(sl) else 0.0 for sl in slicez])

    def spearman(self):
        slicez = self.slices(method='variable')
        #corrs = np.array([spearmanr(*self.dtw_resample(sl))[0] if len(sl) else 0.0 for sl in slicez])
        corrs = np.nan_to_num([spearmanr(*self.dtw_resample(sl))[0] if len(sl) else 0.0 for sl in slicez])
        corrs = np.clip(corrs, a_min=0.0, a_max=1.0)
        return corrs

    #def sqi4(self):
    #    """SQI based on Kurtosis."""


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