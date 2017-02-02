import sys
import numpy as np
from signal import localmax_climb, slices, cross_corr, hz2bpm
import matplotlib.pyplot as plt
from .heartseries import Series


class NoisyECG(object):
    GOOD_BEAT_THRESHOLD = 0.5  #: normalized cross-correlation threshold for good beats, when compared vs. the median

    def __init__(self, ecg, debug=False):
        """:param ecg: heartseries.Series"""

        # needs package ecg-beat-detector
        # imported here, to avoid loading the big matrices (~ 2 sec) otherwise
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
        power_similar = -6.0 < power_ratio_db < 6.0  # 10 dB is a bit lenient. 6 dB would be better, but some baseline drift is larger.

        return lf_hf_db > 5.0 and power_similar

    def beat_detect(self, debug=False, outlierthreshold=0.001):
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
        
        # kill outliers
        mn, mx = np.min(smoothsignal), np.max(smoothsignal)
        m = min(abs(mn),abs(mx))
        N = 100
        step = m/float(N)
        for i in range(N):
            n = len(np.where(smoothsignal<-m)[0]) + len(np.where(smoothsignal>m)[0])
            if n > outlierthreshold*len(smoothsignal):
                break
            m -= step
        mn, mx = -m, m

        smoothsignal[smoothsignal<mn] = mn
        smoothsignal[smoothsignal>mx] = mx
        smoothsignal[-10:] = 0 # extreme outlier in last few frames
        
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


def baseline_energy(ecg):
    """The lowest energy level in dB(1) (should be where ECG signal is)."""
    sll = int(ecg.fps*1.0)  # slice len
    idxs = np.arange(0, len(ecg.x)-sll, sll)
    slices = [ecg.x[i:i+sll] for i in idxs]
    #btt = idxs / float(ecg.fps)
    energies = [10.0*np.log10(np.mean(sl**2)) for sl in slices]
    energies_hist = list(sorted(energies))
    return np.mean(energies_hist[:5])  # at least 5 clean ECG beats should be there, hopefully


def scrub_ecg(ecg_in, THRESHOLD = 8.0):
    """return an ecg signal where noisy bits are set to zero"""
    #ecg = ecg_in.copy()
    #THRESHOLD = 8.0  # dB above baseline_energy()
    ecg = Series(ecg_in.x, ecg_in.fps, ecg_in.lpad)
    #ecg.x = highpass(ecg.x, fps=ecg.fps, cf=2.0, tw=0.4)
    baseline_db = baseline_energy(ecg)
    hwin = int(ecg.fps*0.5)
    check_centers = np.arange(hwin, len(ecg.x)-hwin+1, int(ecg.fps*0.1))  # more densely spaced than hwin
    verdict = []
    for c in check_centers:
        sl = ecg.x[c-hwin:c+hwin+1]
        energy_db = 10.0*np.log10(np.mean(sl**2))
        verdict.append(energy_db < baseline_db + THRESHOLD)

    good_locs = np.where(verdict)[0]
    #flood_fill_width = int(ecg.fps*0.8)
    flood_fill_width = 5  # cf. check_centers step size
    for i in good_locs:
        for j in range(max(i - flood_fill_width, 0), min(i + flood_fill_width + 1, len(verdict))):
            verdict[j] = True

    for c, v in zip(check_centers, verdict):
        if not v:
            ecg.x[c-hwin:c+hwin+1] *= 0.0  # zero the noisy bits

    ecg.x = np.clip(ecg.x, np.mean(ecg.x) - 5*np.std(ecg.x), np.mean(ecg.x) + 5*np.std(ecg.x))

    return ecg  #, check_centers, verdict
