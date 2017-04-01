from numpy.linalg import norm
import numpy as np
from signal import slices, cross_corr
from heartseries import HeartSeries
from dtw import dtw
from scipy.stats.mstats import spearmanr


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
    BEAT_THR = 0.5  #: more beats thrown away? fail creating template 2

    def __init__(self, *args, **kwargs):
        super(QsqiPPG, self).__init__(*args, **kwargs)
        self.template = self.beat_template()
        self.template_kurtosis = kurtosis(self.template)
        self.template_skewness = skewness(self.template)

    @staticmethod
    def from_heart_series(hs):
        return QsqiPPG(hs.x, hs.ibeats, fps=hs.fps, lpad=hs.lpad)

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

    def slices(self, method='direct', L=30):
        if method == 'fixed':
            return np.array(slices(self.x, self.ibeats, hwin=int(self.L*self.fps/2.)))
        elif method == 'variable':
            slicez = []
            for i in range(len(self.ibeats)-1):
                # need to center window on the beat, just like the template
                s,e = self.ibeats[i], self.ibeats[i+1]
                l = e-s
                s,e = max(int(s-l/2.), 0), min(int(e-l/2.), len(self.x))
                if s != e:
                    #plt.plot(self.x[s:e])
                    rez = self.resample(self.x[s:e], L=L)
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
