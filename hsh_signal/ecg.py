import sys
import numpy as np
from signal import localmax_climb, slices, cross_corr, hz2bpm
import matplotlib.pyplot as plt


class NoisyECG(object):
    GOOD_BEAT_THRESHOLD = 0.5  #: normalized cross-correlation threshold for good beats, when compared vs. the median

    def __init__(self, ecg, debug=False):
        """:param ecg: heartseries.Series"""

        sys.path.append('/home/david/heartshield/ecg-beat-detector')
        from kimqrsdetector.kimqrsdetector import QRSdetection
        self.QRSdetection = QRSdetection

        self.ecg = ecg
        self.beat_idxs, self.beat_times = self.beat_detect(debug=debug)
        self.median_ibi = 0.0  #: median inter-beat interval in secs

    def is_valid(self):
        bpm = hz2bpm(1.0 / self.median_ibi)
        bpm_ok = bpm >= 30.0 and bpm <= 150.0
        return bpm_ok and len(self.beat_idxs) >= 5  # at least 5 good beats

    def beat_detect(self, debug=False):
        # Heuristics:
        # * found enough beats
        # * median ibi is in plausible range (or even: some percentile of ibis is in plausible range)
        # * histogram of beat correlations is plausible (lots of good correlation)
        #
        # Good beats have positive correlation, e.g. rho > 0.5 with the median beat.

        ecg = self.ecg

        ecg.x[:int(ecg.fps*5)] *= 0.0  # avoid the terrible swing

        #
        # Kim ECG beat detection
        #
        smoothsignal = ecg.x
        # adjust distribution to the one Kim has optimized for
        smoothsignal = (smoothsignal-np.mean(smoothsignal))/np.std(smoothsignal)*0.148213-0.191034
        loc, beattime = self.QRSdetection(smoothsignal, ecg.fps, ecg.t, ftype=0)
        loc = loc.flatten()

        #
        # check error Kim vs. localmax of R peaks
        #
        new_loc = localmax_climb(ecg.x, loc, hwin=int(0.02*ecg.fps))  # hwin = 20 ms
        peak_errs = (new_loc - loc) / float(ecg.fps)
        #print 'np.mean(peak_errs), np.std(peak_errs)', np.mean(peak_errs), np.std(peak_errs)

        ibis = np.diff(loc / float(ecg.fps))
        median_ibi = np.median(ibis)

        #
        # filter beats by cross-correlation with median beat
        #
        ecg_slices = np.array(slices(ecg.x, loc, hwin=int(np.ceil(median_ibi * ecg.fps))//2))
        # median value from each timepoint (not a single one of any of the beats)
        median_beat = np.median(ecg_slices, axis=0)
        if debug:
            plt.plot(np.arange(len(median_beat))/float(ecg.fps), median_beat)
            plt.title('median ECG beat')
        cross_corrs = [cross_corr(sl, median_beat) for sl in ecg_slices]

        good_loc_idxs = np.where(np.array(cross_corrs) > NoisyECG.GOOD_BEAT_THRESHOLD)[0]
        if debug:
            [plt.plot(np.arange(len(ecg_slices[i]))/float(ecg.fps), ecg_slices[i]) for i in range(1,len(ecg_slices)) if i in good_loc_idxs]
            plt.title('all good ECG beats with rho > {:.2f}'.format(NoisyECG.GOOD_BEAT_THRESHOLD))
            plt.show()

        beat_idxs = loc[good_loc_idxs]
        beat_times = beattime[good_loc_idxs]

        if debug:
            fig, ax = plt.subplots(2, sharex=True)

            ecg.plot(ax[0])
            ax[0].scatter(beat_times, ecg.x[beat_idxs], c='r')

            ax[1].stem(beattime, cross_corrs)

            plt.title('beat correlation with median beat')
            plt.show()

        #self.cross_corrs = np.array(cross_corrs)[good_loc_idxs]
        self.median_ibi = median_ibi
        return beat_idxs, beat_times
