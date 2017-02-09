from __future__ import division
import numpy as np
from bisect import bisect_left
import pickle

from .signal import slices, cross_corr
from .iter import pairwise


class Series(object):
    def __init__(self, samples, fps, lpad=0):
        self.x = np.array(samples)
        self.t = (np.arange(len(self.x)) - lpad) / float(fps)
        self.fps = fps
        self.lpad = lpad

    def len_time(self):
        """for padded signal, returns the original unpadded length"""
        return float(len(self.x)-1 - 2*self.lpad) / float(self.fps)

    def pad(self, amount=None):
        """zero-pad left and right to 3x the length. Padding keeps self.t same in the signal range"""
        amount = len(self.x) if amount is None else int(amount)
        return Series(np.pad(self.x, ((amount, amount),), mode='constant'), self.fps, lpad=amount)

    def copy(self):
        return Series(self.x, self.fps, self.lpad)

    def slice(self, s):
        return HeartSeries(self.x[s], self.ibeats[s], self.fps, self.lpad)

    def add_beats(self, ibeats):
        return HeartSeries(self.x, ibeats, self.fps, self.lpad)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fi:
            return pickle.load(fi)

    def dump(self, filename):
        with open(filename, 'wb') as fo:
            pickle.dump(self, fo, protocol=2)

    def plot(self, plotter=None, dt=0.0):
        import matplotlib.pyplot as plt

        plotter = plt if plotter is None else plotter
        plotter.plot(self.t - dt, self.x)

    def stem(self, plotter=None, dt=0.0):
        import matplotlib.pyplot as plt

        plotter = plt if plotter is None else plotter
        plotter.stem(self.t - dt, self.x)


class HeartSeries(Series):
    def __init__(self, samples, ibeats, fps, lpad=0):
        # ibeats may still be float, despite the name
        super(HeartSeries, self).__init__(samples, fps, lpad=lpad)
        ibeats = np.array(ibeats).flatten()
        self.ibeats = np.array(ibeats)
        self.tbeats = (ibeats - lpad) / float(fps)

    def pad(self, amount=None):
        amount = len(self.x) if amount is None else int(amount)
        series = np.pad(self.x, ((amount, amount),), mode='constant')
        ibeats = self.ibeats + amount
        return HeartSeries(series, ibeats, self.fps, lpad=amount)

    def copy(self):
        return HeartSeries(self.x, self.ibeats, self.fps, self.lpad)

    def shift(self, dt):
        """add a time shift in seconds, moving the signal to the left."""
        self.lpad += dt * self.fps
        self.t -= dt
        self.tbeats -= dt

    def plot(self, plotter=None, dt=0.0):
        import matplotlib.pyplot as plt

        plotter = plt if plotter is None else plotter
        plotter.plot(self.t - dt, self.x)

        self.scatter(plotter, dt)

    def scatter(self, plotter=None, dt=0.0, tbeats2=None, c='r'):
        import matplotlib.pyplot as plt

        plotter = plt if plotter is None else plotter
        tbeats2 = self.tbeats if tbeats2 is None else tbeats2

        beat_y = []
        tbeats = tbeats2 + self.lpad / float(self.fps)
        for tb in tbeats:
            ib = tb * self.fps
            if ib < 1.0 or ib > len(self.x) - 2:
                beat_y.append(self.x[max(min(int(ib), len(self.x)-1), 0)])
                continue
            xs = [int(np.floor(ib)), int(np.ceil(ib))]
            if xs[0] == xs[1]:
                xs[1] += 1  # happens if ib is precise integer
            ys = self.x[xs]
            beat_y.append(np.interp([ib], xs, ys)[0])
        plotter.scatter(tbeats2 - dt, beat_y, c=c)

    def closest_beat(self, t):
        """find the ib of the beat closest to t"""
        ib = bisect_left(self.tbeats, t)
        ibl, ibr = max(ib-1, 0), min(ib+1, len(self.tbeats)-1)
        if ib == len(self.tbeats):
            ib -= 1
        idxs = [ibl,ib,ibr]
        cand_times = self.tbeats[idxs]
        #print 'closest_beat(t=',t,') cand_times=', cand_times
        return idxs[np.argmin(np.abs(cand_times - t))]

    def aligned_iibeats_repeat(self, ecg, ppg_dt=0.0):
        ppg_ibs, ecg_ibs = [], []
        for ib, t in enumerate(self.tbeats):
            ecg_ib = ecg.closest_beat(t - ppg_dt)
            ppg_ibs.append(ib)
            ecg_ibs.append(ecg_ib)
        assert len(ppg_ibs) == len(ecg_ibs)
        return ppg_ibs, ecg_ibs

    def aligned_iibeats(self, ecg, ppg_dt=0.0):
        """return all beats that align to a reference ecg. ppg is delayed by ppg_dt"""
        ppg_ibs, ecg_ibs = [], []
        ecg_used_beats = np.zeros(len(ecg.tbeats), dtype=np.bool)
        for ib, t in enumerate(self.tbeats):
            ecg_ib = ecg.closest_beat(t - ppg_dt)
            #print 'PPG beat t=', t, 'found ECG beat t=', ecg.tbeats[ecg_ib], 'ecg_used=', ecg_used_beats[ecg_ib]

            # align in reverse to check if we are using something early, that can be used better later
            tbetter = self.tbeats[self.closest_beat(ecg.tbeats[ecg_ib])]
            #if ib == 0:
            #    print 't', t - ppg_dt, 'tbetter', tbetter - ppg_dt
            better_aln_exists = np.abs(ecg.tbeats[ecg_ib] - t + ppg_dt) > np.abs(tbetter - ppg_dt - ecg.tbeats[ecg_ib])
            if ecg_used_beats[ecg_ib] or better_aln_exists and tbetter > t:
                continue
            delta_t = (t - ppg_dt) - ecg.tbeats[ecg_ib]
            if np.abs(delta_t) > 0.15:
                continue
            ecg_used_beats[ecg_ib] = True
            ppg_ibs.append(ib)
            ecg_ibs.append(ecg_ib)
        assert len(ppg_ibs) == len(ecg_ibs)
        return ppg_ibs, ecg_ibs

    def snr(self, mode='median'):
        return self.beat_snr(mode)

    def beat_snr(self, mode='median'):
        """
        signal-to-noise ratio.
        mode='neighbors': correlation with neighboring beats (returns range (-inf, inf) dB)
        mode='median': correlation with median beat (returns range (-inf, 0) dB)  -- nope. can be >0 dB

        generally, for bad signals, 'neighbors' is simply 2 dB less than 'median'.

        :return SNR in dB
        """

        #ibeats = self.ibeats
        ibeats = self.ibeats[np.where(self.tbeats > 5.0)[0]]  # cut off leading noisy stuff...

        if len(ibeats) < 2:
            return -20.0  # pretend bad SNR if not enough beats were found.

        mean_ibi = np.mean(np.diff(ibeats))
        slicez = slices(self.x, ibeats, hwin=int(mean_ibi/2))

        if mode == 'neighbors':
            # actually this is log(corr) and slightly different from SNR.
            corrs = [cross_corr(a, b) for a, b in pairwise(slicez)]
            return 10.0 * np.log10(np.mean(corrs))
        elif mode == 'median':
            # median value from each timepoint (not a single one of any of the beats)
            median_beat = np.median(np.array(slicez), axis=0)
            num = [np.sum(median_beat**2) for sl in slicez]
            denom = [np.sum((sl - median_beat)**2) for sl in slicez]
            sns = [(n/d if d != 0.0 else np.inf) for n,d in zip(num, denom)]
            return 10.0 * np.log10(np.mean(sns))
            # note: this is also affected by steep baseline wander (but so should be the zero-crossing time detection [test!]. so fine)
        else:
            raise ValueError('snr() unknown mode={}'.format(mode))
