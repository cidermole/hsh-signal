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
        self.median_ibi = 0.0  #: median inter-beat interval in secs
        self.good_beat_fraction = 0.0  #: fraction of Kim-detected beats which correlate well with the median beat
        self.beat_idxs, self.beat_times = self.beat_detect(debug=debug)

    def is_valid(self, debug=False):
        bpm = hz2bpm(1.0 / (self.median_ibi + 1e-6))
        bpm_ok = bpm >= 30.0 and bpm <= 150.0
        enough_beats = len(self.beat_idxs) >= 5  # at least 5 good beats
        beat_hist_ok = self.good_beat_fraction > 0.5  # meh. not good enough??

        if debug:
            print 'median_ibi={:.3f} -> bpm={:.1f}. len(beat_idxs)={} good_beat_fraction={:.2f}'.format(self.median_ibi, bpm, len(self.beat_idxs), self.good_beat_fraction)
        return bpm_ok and enough_beats and beat_hist_ok

    def slice_good(self, sl, median_beat):
        spectrum = np.abs(np.fft.fft(sl)**2)

        # around 1/8, there is a bottom in a clean signal (see plot of mean beat spectrum)
        lf_hf_db = 10.0 * np.log10(np.sum(spectrum[0:len(spectrum)//8]) / np.sum(spectrum[len(spectrum)//8:len(spectrum)//2]))

        # the slice has similar power like the median_beat
        power_ratio_db = 10.0 * np.log10(np.sum(sl**2) / np.sum(median_beat**2))

        if False:
            plt.plot(spectrum, c='r')
            plt.plot(median_beat, c='k')
            plt.plot(sl, c='b')
            plt.title('slice_good() lf_hf={:.1f} dB  slice/median power_ratio_db={:.1f} dB'.format(lf_hf_db, power_ratio_db))
            plt.show()

        # the slice has similar power like the median_beat
        power_similar = -6.0 < power_ratio_db < 6.0

        return lf_hf_db > 5.0 and power_similar

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

        spectrum_ok = np.array([self.slice_good(sl, median_beat) for sl in ecg_slices])
        ccs_ok = np.array(cross_corrs) > NoisyECG.GOOD_BEAT_THRESHOLD

        good_loc_idxs = np.where(ccs_ok & spectrum_ok)[0]
        if debug:
            [plt.plot(np.arange(len(ecg_slices[i]))/float(ecg.fps), ecg_slices[i]) for i in range(1,len(ecg_slices)) if i in good_loc_idxs]
            plt.title('all good ECG beats with rho > {:.2f}'.format(NoisyECG.GOOD_BEAT_THRESHOLD))
            plt.show()

        beat_idxs = loc[good_loc_idxs]
        beat_times = beattime[good_loc_idxs]

        self._beattime, self._cross_corrs = beattime, cross_corrs

        if debug:
            self.debug_plot()

        #self.cross_corrs = np.array(cross_corrs)[good_loc_idxs]
        self.median_ibi = median_ibi
        self.good_beat_fraction = float(len(good_loc_idxs)) / len(cross_corrs)
        return beat_idxs, beat_times

    def debug_plot(self):
        ecg = self.ecg

        beattime, cross_corrs = self._beattime, self._cross_corrs

        fig, ax = plt.subplots(2, sharex=True)

        ecg.plot(ax[0])
        ax[0].scatter(self.beat_times, ecg.x[self.beat_idxs], c='r')

        ax[1].stem(beattime, cross_corrs)

        plt.title('beat correlation with median beat')
        plt.show()