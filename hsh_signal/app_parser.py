import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
import calendar
import pickle
import glob
from collections import defaultdict

from .pickling import load_zipped_pickle
from .alivecor import decode_alivecor, beatdet_alivecor, load_raw_audio
from .signal import evenly_resample, highpass
from .heartseries import Series
from .ppg import ppg_beatdetect_brueser, ppg_beatdetect_getrr, beatdet_getrr_v2
from hsh_signal.quality import QsqiPPG, QsqiError
from scipy.interpolate import interp1d
from hsh_signal.signal import localmax, lowpass
from .ecg import ecg_snr

import requests
import json
from .hsh_data import MyJSONEncoder


def classify_results(meta_data, series_data, host='https://mlapi.heartshield.net'):
    post_data = {
        'meta_data': json.dumps(meta_data, cls=MyJSONEncoder),
        'series_data': json.dumps(series_data, cls=MyJSONEncoder)
    }
    response = requests.post(host + '/v2/reclassify', post_data)
    if response.status_code != 200:
        raise RuntimeError('requests.post() status code != 200: ' + response.text)

    result_dict = json.loads(response.text)
    #prob, filtered, idx = [result_dict[k] for k in 'pred filtered idx'.split()]  # does not work for 'unclassified'

    #meta_fname = self.hs_data.put_result(meta_data, result_dict)
    return result_dict


def parse_app_series(filename):
    """TODO: deprecate"""
    series_data = np.load(filename)

    audio_data = series_data['audio_data']
    ecg_raw = decode_alivecor(series_data['audio_data'][:,1])
    #audio_fps = meta_data['audio_fps']  # no meta_data
    audio_ts = series_data['audio_data'][:,0]
    audio_fps = float(len(audio_ts) - 1) / (audio_ts[-1] - audio_ts[0])

    ecg_fps = 300
    ecg = ecg_raw[::int(audio_fps/ecg_fps)]
    ecgt = audio_data[:,0][::int(audio_fps/ecg_fps)]

    ecg_sig = ecg
    ecg_ts = ecgt

    #ecg_series = Series(ecg_sig)
    # variable time delay... grr.

    ppg_fps = 30.0
    ppg_data = evenly_resample(series_data['ppg_data'][:,0], series_data['ppg_data'][:,1], target_fps=ppg_fps)
    #ppg_data = ppg_data[int(series_data['ppg_data'][:,0][0]*ppg_fps):,:]


    """
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(ecg_ts, ecg_sig)
    ax[1].plot(series_data['ppg_data'][:,0], series_data['ppg_data'][:,1])
    #ax[2].plot(series_data['bcg_data'][:,0], series_data['bcg_data'][:,3])
    """

    return ecg_ts, highpass(ecg_sig, ecg_fps), ppg_data[:,0], highpass(highpass(ppg_data[:,1], ppg_fps), ppg_fps)


def sanitize(s, validchars):
    return re.sub('[^' + validchars + ']','_', s)


def server_series_filename(meta_data):
    unixtime = int(calendar.timegm(meta_data['start_time'].utctimetuple()))-3600  # TODO: server timezone :(
    sane_app_id = sanitize(meta_data['app_info']['id'], '0123456789ABCDEF')
    return '{}_{}_series.b'.format(unixtime, sane_app_id)


def audio1_filename(audio_base, meta_data):
    sane_app_id = sanitize(meta_data['app_info']['id'], '0123456789ABCDEF')
    start_time = int(calendar.timegm(meta_data['start_time'].utctimetuple()))-3600  # TODO: server timezone :(
    return os.path.join(audio_base, '{}_series.b_{}'.format(start_time, sane_app_id))

def audio2_filename(audio_base, meta_data):
    sane_app_id = sanitize(meta_data['app_info']['id'], '0123456789ABCDEF')
    start_time = int(calendar.timegm(meta_data['start_time'].utctimetuple()))
    return os.path.join(audio_base, '{}_{}'.format(start_time, sane_app_id))

def audio_filename(audio_base, meta_data):
    ans = audio1_filename(audio_base, meta_data), audio2_filename(audio_base, meta_data)
    for an in ans:
        if os.path.exists(an):
            return an
    return ans[0]


class BeatParseError(Exception):
    def __init__(self, *args, **kwargs):
        super(BeatParseError, self).__init__(*args, **kwargs)


class BeatShape(object):
    def __init__(self, beat_template):
        self.template = beat_template
        self.parse()

    def plot(self):
        beat_fine = self.beat_fine
        t_fine = self.t_fine
        dbeat_dt = np.diff(beat_fine)
        dbeat_dt2 = np.diff(np.diff(beat_fine))
        dbeat_dt2_smooth = lowpass(dbeat_dt2, fps=300., cf=10.0, tw=2.)

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.t, self.beat_shape)
        ax[0].plot(t_fine, beat_fine, 'r')
        # ax[1].plot(t[:-1], np.diff(beatshape), 'r')
        ax[1].plot(t_fine[:-1], dbeat_dt * 10, 'b')
        ax[1].plot(t_fine[:-2], dbeat_dt2 * 100, 'k')
        ax[1].plot(t_fine[:-2], dbeat_dt2_smooth * 100, 'y')
        ax[1].scatter([t_fine[self.imin]], [(dbeat_dt2_smooth * 100)[self.imin]], c='g')  # chosen local minimum
        ax[0].scatter([t_fine[self.imin]], [beat_fine[self.imin]], c='g')  # chosen local minimum

        #

        reflection_peak = beat_fine[self.imin]
        islope_max = np.argmax(dbeat_dt)
        zxings = np.where(localmax(-np.abs(dbeat_dt)))[0]
        iforward_peak = zxings[np.where(zxings > islope_max)[0][0]]
        forward_peak = beat_fine[iforward_peak]
        ax[0].scatter([t_fine[iforward_peak]], [forward_peak], c='k')  # forward wave maximum

        ifoot2 = np.argmin(beat_fine)
        foot2 = beat_fine[ifoot2]
        ax[0].scatter([t_fine[ifoot2]], [foot2], c='b')  # foot

        # or, it could also be the maximum of second derivative -- more stable?!
        #ifoot3 = np.argmax(dbeat_dt2[:len(dbeat_dt2) * 2 // 3])
        ifoot3 = np.argmax(dbeat_dt2_smooth[:len(dbeat_dt2) * 2 // 3])

        foot3 = beat_fine[ifoot3]
        ax[0].scatter([t_fine[ifoot3]], [foot3], c='r')  # foot
        return fig

    def parse(self):
        # shift concat template vertically so the lines fit neatly
        # beatshape = -1 * np.concatenate((sq.template[:-1], -sq.template[0] + sq.template[-1] + sq.template))[:-len(sq.template) / 2] # always inverted on mobile
        beat_shape = self.template
        t = np.linspace(0.0, len(beat_shape) / 30.0, len(beat_shape), False)
        t_fine = np.linspace(0.0, t[-1], len(beat_shape) * 10, False)
        self.t, self.t_fine, self.beat_shape = t, t_fine, beat_shape

        bsp = interp1d(t, beat_shape, kind='cubic')
        beat_fine = bsp(t_fine)
        self.beat_fine = beat_fine

        # looking for diastolic peak -- see Elgendi 2012, Fig 15. (b)

        dbeat_dt = np.diff(beat_fine)

        # look for local minimum in second derivative.
        dbeat_dt2 = np.diff(np.diff(beat_fine))
        dbeat_dt2_smooth = lowpass(dbeat_dt2, fps=300., cf=10.0, tw=2.)

        ipeak = np.argmax(beat_fine)  # TODO: what if first peak is not the highest?
        ifoot = np.argmin(beat_fine[ipeak:]) + ipeak

        ilocalmin = np.where(localmax(-dbeat_dt2_smooth))[0]
        #print 'localmax found',len(ilocalmin)
        ilocalmin = ilocalmin[np.where(ilocalmin > ipeak)[0]]
        ilocalmin = ilocalmin[np.where(ilocalmin < ifoot)[0]]
        #print 'between ipeak=', ipeak, 'and ifoot=', ifoot, 'we have', len(ilocalmin)

        if len(ilocalmin) == 0:
            raise BeatParseError('peak and foot not found')

        # TODO: catch errors

        # find lowest local minimum
        # TODO: better: find the first local minimum after the localmax plateau
        #imin = ilocalmin[np.argmin(dbeat_dt2[ilocalmin])]
        imin = ilocalmin[np.argmin(dbeat_dt2_smooth[ilocalmin])]

        # TODO: stability estimates. on the smoothed curve, how much wiggling room is there, how clear=pronounced is the minimum?

        # TODO: fit to all data points (slices), instead of just to the mean!

        # TODO: count zero crossings of dbeat_dt2 -- first local minimum after plateau (plateau = signal already been above zxing)

        self.imin = imin


        reflection_peak = beat_fine[imin]
        islope_max = np.argmax(dbeat_dt)
        zxings = np.where(localmax(-np.abs(dbeat_dt)))[0]
        if len(np.where(zxings > islope_max)[0]) == 0:
            raise BeatParseError('no good zero crossing found')
        idxing = np.where(zxings > islope_max)[0][0]
        iforward_peak = zxings[idxing]
        forward_peak = beat_fine[iforward_peak]
        #ax[0].scatter([t_fine[iforward_peak]], [forward_peak], c='k')  # forward wave maximum

        ifoot2 = np.argmin(beat_fine)
        foot2 = beat_fine[ifoot2]
        #ax[0].scatter([t_fine[ifoot2]], [foot2], c='b')  # foot

        # or, it could also be the maximum of second derivative -- more stable?!
        ifoot3 = np.argmax(dbeat_dt2[:len(dbeat_dt2) * 2 // 3])
        foot3 = beat_fine[ifoot3]
        #ax[0].scatter([t_fine[ifoot3]], [foot3], c='r')  # foot

        aix2 = (forward_peak - reflection_peak) / (forward_peak - foot2)
        aix3 = (forward_peak - reflection_peak) / (forward_peak - foot3)

        #hrs.append(hr)
        #aixs.append(aix3)
        self.aix2 = aix2
        self.aix3 = aix3


class LazyDict(dict):
    """avoids unpickling a lot of data, unless we actually need it."""
    def __init__(self, zipped, filename):
        super(LazyDict, self).__init__()
        self._loaded = False
        self.zipped, self.filename = zipped, filename
        if not os.path.exists(filename):
            raise IOError('LazyDict: file not found: {}'.format(filename))

    def load(self):
        if self.zipped:
            self.update(load_zipped_pickle(self.filename))
        else:
            self.update(np.load(self.filename))
        self._loaded = True

    def __getitem__(self, key):
        if not self._loaded:
            self.load()
        return super(LazyDict, self).__getitem__(key)


class AppData:
    """source agnostic loader, handles data from phones and server."""
    CACHE_DIR = '.cache-nosync'

    BASE_DIR = '/mnt/hsh/data/appresults.v2-nosync/appresults.v2'

    KNOWN_APP_IDS = defaultdict(str)
    KNOWN_APP_IDS.update({
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***',
        '***REMOVED***': '***REMOVED***'
    })


    def __init__(self, meta_filename):
        # , meta_filename=None, series_filename=None

        if not os.path.exists(meta_filename):
            meta_filename_new = AppData.BASE_DIR + '/' + meta_filename
            if os.path.exists(meta_filename_new):
                meta_filename = meta_filename_new
            else:
                raise IOError('AppData meta file not found: {}'.format(meta_filename))

        self._zipped = None
        try:
            # app-saved metadata: normal pickle
            meta_data = np.load(meta_filename)

            dn = os.path.dirname(meta_filename)
            series_filename = os.path.join(dn, meta_data['series_fname'])
            #series_data = np.load(series_filename)
            self._zipped = False
        except:
            # maybe it's server-saved metadata
            meta_data = load_zipped_pickle(meta_filename)

            dn = os.path.dirname(meta_filename)
            #series_filename = os.path.join(dn, server_series_filename(meta_data))
            series_filename = meta_filename.replace('_meta', '_series')  # fix hackish filenames. grrml.
            #series_data = load_zipped_pickle(series_filename)
            self._zipped = True

        self.meta_filename = meta_filename
        self.meta_data = meta_data
        #self.series_data = series_data
        self.series_data = LazyDict(self._zipped, series_filename)

    def ecg_parse_beatdetect(self):
        cache_file = os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_beatdet_ecg.b')
        if os.path.exists(cache_file):
            return np.load(cache_file)

        audio_base = os.path.join(os.path.dirname(self.meta_filename), 'audio')
        raw_sig, fps = load_raw_audio(audio_filename(audio_base, self.meta_data))
        self.series_data.load()
        ecg = beatdet_alivecor(raw_sig, fps)

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump(ecg, fo)

        # note: we do not yet honor series_data['audio_start'] here.

        return ecg

    def ecg_snr(self):
        """SNR of AliveCor ECG in raw audio. For quick (0.5 sec) checking whether audio contains ECG or not."""
        audio_base = os.path.join(os.path.dirname(self.meta_filename), 'audio')
        if not os.path.exists(audio_filename(audio_base, self.meta_data)):
            return -10.0
        # check spectrum
        try:
            raw_sig, fps = load_raw_audio(audio_filename(audio_base, self.meta_data))
            return ecg_snr(raw_sig, fps)
        except ValueError:
            # File "/mnt/hsh/hsh-signal/hsh_signal/alivecor.py", line 68, in load_raw_audio
            # arr = arr.reshape((nframes, wf.getnchannels())).T
            # ValueError: total size of new array must be unchanged
            return -10.0

    def has_ecg(self, THRESHOLD=25.0):
        """
        to do: refactor rename THRESHOLD to snr_threshold
        Returns True if an AliveCor is in the audio track. Does not mean there's a clean ECG recording.
        """
        cache_file = os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_beatdet_hasecg.b')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fi:
                pld = pickle.load(fi)
                if isinstance(pld, tuple):
                    th, rv = pld
                    if th == THRESHOLD: return rv

        audio_base = os.path.join(os.path.dirname(self.meta_filename), 'audio')
        if os.path.exists(audio_filename(audio_base, self.meta_data)):
            # check spectrum
            try:
                raw_sig, fps = load_raw_audio(audio_filename(audio_base, self.meta_data))
                retval = ecg_snr(raw_sig, fps) > THRESHOLD  # below, very few ECGs are usable...
            except (ValueError, EOFError):
                # EOFError for 0-byte transmissions (from new Android client...)

                # File "/mnt/hsh/hsh-signal/hsh_signal/alivecor.py", line 68, in load_raw_audio
                # arr = arr.reshape((nframes, wf.getnchannels())).T
                # ValueError: total size of new array must be unchanged
                retval = False
        else:
            retval = False

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump((THRESHOLD, retval), fo)

        return retval

    def ppg_fps(self):
        ppg_data = self.series_data['ppg_data']
        ts = ppg_data[:,0]
        if len(ts) < 2:
            return 0.0
        return float(len(ts) - 1) / (ts[-1] - ts[0])

    def ppg_raw(self):
        ppg_data_uneven = self.series_data['ppg_data']

        ppg_fps = 30.0
        ppg_data = evenly_resample(ppg_data_uneven[:,0], ppg_data_uneven[:,1], target_fps=ppg_fps)
        ts = ppg_data[:,0]
        return Series(ppg_data[:,1], fps=ppg_fps, lpad=-ts[0]*ppg_fps)

    def ppg_trend(self):
        ppg_data_uneven = self.series_data['ppg_data']

        ppg_fps = 30.0
        ppg_data = evenly_resample(ppg_data_uneven[:,0], ppg_data_uneven[:,1], target_fps=ppg_fps)
        ts = ppg_data[:,0]
        demean = highpass(highpass(ppg_data[:,1], ppg_fps), ppg_fps)
        trend = ppg_data[:,1] - demean

        return Series(trend, fps=ppg_fps, lpad=-ts[0]*ppg_fps)

    def bcg_vectors(self):
        fps = self.meta_data['bcg_fps']
        bcg_data_uneven = self.series_data['bcg_data']

        if len(bcg_data_uneven) == 0:
            return [Series([], fps, lpad=0) for i in range(3)]

        axes = []
        ts = []
        for i in range(1,4):
            resampled = evenly_resample(bcg_data_uneven[:,0], bcg_data_uneven[:,i], target_fps=fps)
            axes.append(resampled[:,1])
            ts = resampled[:,0]

        return [Series(ax, fps=fps, lpad=-ts[0]*fps) for ax in axes]

    def bcg_abs(self):
        vectors = self.bcg_vectors()
        accel = []
        if len(vectors[0].x) == 0: return Series([], fps=vectors[0].fps, lpad=0)
        for x,y,z in zip(*[v.x for v in vectors]):
            accel.append(np.sqrt(np.sum(np.array([x,y,z])**2)))
        return Series(accel, fps=vectors[0].fps, lpad=vectors[0].lpad)

    def bcg_abs_hp(self):
        babs = self.bcg_abs()
        return Series(highpass(highpass(babs.x, babs.fps), babs.fps), fps=babs.fps, lpad=babs.lpad)

    def ppg_data(self):
        """server-like data[] of PPG. evenly resampled and first 5sec cut off. Just like fed to getrr()"""
        ppg_data_uneven = self.series_data['ppg_data']  # (N,4) matrix with [t,r,g,b] rows
        times, series = ppg_data_uneven[:,0], ppg_data_uneven[:,1]  # red channel

        # cut off the first 5 seconds for classification, just like in the v1 /rawrfclassify API
        data = evenly_resample(times, series)
        istart = np.where(times - times[0] > 5.0)[0][0]
        data = data[istart:,:]

        return data

    def ppg_parse(self):
        ppg_data_uneven = self.series_data['ppg_data']

        ppg_fps = 30.0
        ppg_data = evenly_resample(ppg_data_uneven[:,0], ppg_data_uneven[:,1], target_fps=ppg_fps)
        ts = ppg_data[:,0]
        demean = highpass(highpass(ppg_data[:,1], ppg_fps), ppg_fps)

        return Series(demean, fps=ppg_fps, lpad=-ts[0]*ppg_fps)

    def ppg_parse_beatdetect(self, type='brueser', use_cache=True):
        cache_file = os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_' + type + '_beatdet.b')
        if os.path.exists(cache_file) and use_cache:
            return np.load(cache_file)

        if type == 'brueser':
            ppg = ppg_beatdetect_brueser(self.ppg_parse())
        elif type == 'getrr':
            # note the hidden highpass -- it is in ppg_beatdetect_getrr() -> beatdet()
            ppg = ppg_beatdetect_getrr(self.ppg_raw())
        else:
            raise ValueError('type must be one of brueser|getrr')

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump(ppg, fo)

        return ppg

    def qsqi(self):
        self.series_data.load()
        ppg_data = self.series_data['ppg_data']  # (N,4) matrix with [t,r,g,b] rows
        times, series = ppg_data[:,0], ppg_data[:,1]  # red channel
        # note: these are from second 0 of the measurement, i.e. first part is noisy!!!

        #print times
        #print series

        if 'lock_time' in self.meta_data:
            # on android client v0.5.0+
            lock_time = self.meta_data['lock_time']
        else:
            # on kivy client
            lock_time = 5.0

        # cut off the first 5 seconds for classification, just like in the v1 /rawrfclassify API
        data = evenly_resample(times, series)
        istart = np.where(times - times[0] > lock_time)[0][0] if len(np.where(times - times[0] > lock_time)[0]) > 0 else 0
        data = data[istart:,:]

        # 30 FPS hardcoded
        ibi, filtered, idxs, tbeats = beatdet_getrr_v2(data, get_tbeats=True)
        tbeats = np.array(tbeats)/1e3

        #print 'idxs', idxs
        #print 'tbeats', tbeats

        try:
            QsqiPPG.BEAT_THR = 0.25
            #sq = QsqiPPG.from_series_data(data[:,1], idx)
            sq = QsqiPPG.from_series_data(data[:,1], (tbeats - data[0,0]) * 30.0)
        except QsqiError as e:
            print e
            return None

        # to do: this is FPS quantized, like all using beatdet_getrr_v2() - including server's SQI, ppg_parse_beatdetect() etc.
        return sq

    def beat_shape(self):
        ppg = self.ppg_parse_beatdetect('getrr', use_cache=False)
        try:
            sq = self.qsqi()
        except IndexError as e:
            # trying to access ibeats[-1]
            raise BeatParseError(e)
        except Warning:
            # File "/mnt/hsh/hsh-beatdet/hsh_beatdet/ML/ppg_beatdetector_v2.py", line 66, in getrr_v2
            # raise Warning("Warning: Tiny data shape", data.shape[0])
            raise BeatParseError(e)
        return BeatShape(-1 * sq.template)

    def ppg_footed(self):
        """return upsampled and footed (detrended) PPG."""
        from hsh_signal.waveshape import ppg_wave_foot
        ppg_raw_l = self.ppg_raw()
        ppg_l = self.ppg_parse_beatdetect('getrr')
        return ppg_wave_foot(ppg_raw_l, ppg_l)

    def ecg_ppg_aligned(self):
        """>>> ecg, ppg, ecg_ibs, ppg_ibs = ad.ecg_ppg_aligned()"""
        import sys
        sys.path.append('/home/david/heartshield/noisemodel')
        from align_beats import analyze_ecg_ppg_base

        ecg = self.ecg_parse_beatdetect()
        ppg = self.ppg_parse_beatdetect('getrr', use_cache=False)

        ppg_dt, rel_fps_error = analyze_ecg_ppg_base(ecg, ppg, step=50e-3, debug=False)

        ppg.shift(ppg_dt)

        ppg_ibs, ecg_ibs = ppg.aligned_iibeats(ecg, ppg_dt=0.0)

        return ecg, ppg, ecg_ibs, ppg_ibs

    def get_result(self, reclassify=False, host='https://mlapi.heartshield.net'):
        """calls HS API /reclassify if necessary (if result not yet cached).
        :returns result dict with keys ['pred', 'filtered', 'idx']"""
        cache_files = [
            self.meta_filename.replace('_meta', '_result'),
            os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_result.b')
            ]
        if not reclassify:
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    return load_zipped_pickle(cache_file)

        self.series_data.load()
        res = classify_results(self.meta_data, self.series_data, host=host)

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump(res, fo)

        return res

    def has_diagnosis(self):
        if not 'doctor' in self.meta_data:
            # no 'doctor' key: did not save any input, or old app version
            return False

        doctor = self.meta_data['doctor']

        # directly selected "CVD" or "No CVD found"
        if doctor['status'] != '': return True

        # maybe a specific CVD was directly selected? (app UI should've selected "CVD" automatically...)
        if self.cad_or_afib(): return True

        return False

    def notes(self):
        if not 'doctor' in self.meta_data:
            # no 'doctor' key: did not save any input, or old app version
            return u''
        doctor = self.meta_data['doctor']
        return doctor['text'].replace('\n', ' ')

    def lock_time(self):
        if 'lock_time' in self.meta_data:
            # on android client v0.5.0+
            return self.meta_data['lock_time']
        else:
            # on kivy client
            return 5.0

    def model(self):
        return self.meta_data['app_info']['install_android_versions']['Build.MODEL'] if ('app_info' in self.meta_data and 'install_android_versions' in self.meta_data['app_info'] and 'Build.MODEL' in self.meta_data['app_info']['install_android_versions']) else None

    def mf(self):
        return os.path.basename(self.meta_filename)

    def start_time(self):
        return self.meta_data['start_time']

    def app_id(self):
        return self.meta_data['app_info']['id']

    def app_version(self):
        return self.meta_data['app_info']['version'] if ('app_info' in self.meta_data and 'version' in self.meta_data['app_info']) else None

    def app_codename(self):
        return self.meta_data['app_info']['codename'] if ('app_info' in self.meta_data and 'codename' in self.meta_data['app_info']) else None

    def user_name(self):
        """returns empty string if unknown."""
        app_id = self.app_id()
        return AppData.KNOWN_APP_IDS[app_id]

    def cad_or_afib(self):
        if not 'doctor' in self.meta_data: return False
        doctor = self.meta_data['doctor']
        if 'details' in doctor:
            details = doctor['details']
            if details['cad'] or ('afib' in details and details['afib']):
                return True
        return False

    def cad(self):
        return self.has_disease('cad')

    def afib(self):
        return self.has_disease('afib')

    def has_disease(self, dtype='cad'):
        if not 'doctor' in self.meta_data: return None
        doctor = self.meta_data['doctor']
        if 'details' in doctor:
            details = doctor['details']
            if dtype in details and details[dtype]:
                return True
            if not self.has_diagnosis():
                return None
            status = self.meta_data['doctor']['status']
            if status == 'healthy':
                return False
        return None

    def _age_or_gender(self, ftype='age'):
        # new app v0.3 parser (line of apps which always ask for age/gender)
        if 'user' in self.meta_data:
            if ftype == 'gender':
                return self.meta_data['user'][ftype]
            elif ftype == 'age':
                a = self.meta_data['user'][ftype]
                return int(a) if (a != '' and a is not None) else None

        # old app series v0.1.*
        if not 'doctor' in self.meta_data:
            return None
        doctor = self.meta_data['doctor']
        if not 'details' in doctor:
            return None
        details = doctor['details']
        if ftype in details and details[ftype] == '':
            return None
        if not ftype in details:
            return None
        if ftype == 'age':
            return int(details[ftype]) if details[ftype] != '' else None
        return details[ftype]

    def age(self):
        return self._age_or_gender('age')

    def gender(self):
        return self._age_or_gender('gender')

    def get_cvd_status(self):
        if not self.has_diagnosis():
            return None
        status = self.meta_data['doctor']['status']
        if status == 'cvd' or self.cad_or_afib():
            return True
        elif status == 'healthy':
            return False
        raise ValueError('unknown CVD status: {} in file {}'.format(status, self.meta_filename))

    @staticmethod
    def list_measurements():
        meta_filenames = list(sorted(glob.glob(AppData.BASE_DIR + '/*_meta.b')))

        app_data = []
        for mf in meta_filenames:
            try:
                app_data.append(AppData(mf))
            except (IOError, EOFError) as e:
                print mf, e

        return app_data
