import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
import pickle

from .pickling import load_zipped_pickle
from .alivecor import decode_alivecor
from .signal import evenly_resample, highpass
from .heartseries import Series
from .ppg import ppg_beatdetect

import requests
import json
from .hsh_data import MyJSONEncoder


def classify_results(meta_data, series_data):
    post_data = {
        'meta_data': json.dumps(meta_data, cls=MyJSONEncoder),
        'series_data': json.dumps(series_data, cls=MyJSONEncoder)
    }
    response = requests.post('https://mlapi.heartshield.net/v2/reclassify', post_data)
    if response.status_code != 200:
        raise RuntimeError('requests.post() status code != 200: ' + response.text)

    result_dict = json.loads(response.text)
    prob, filtered, idx = [result_dict[k] for k in 'pred filtered idx'.split()]

    #meta_fname = self.hs_data.put_result(meta_data, result_dict)
    return result_dict


def parse_app_series(filename):
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
    unixtime = int(time.mktime(meta_data['start_time'].timetuple()))
    sane_app_id = sanitize(meta_data['app_info']['id'], '0123456789ABCDEF')
    return '{}_{}_series.b'.format(unixtime, sane_app_id)


class AppData:
    """source agnostic loader, handles data from phones and server."""

    CACHE_DIR = '.cache-nosync'

    def __init__(self, meta_filename):
        # , meta_filename=None, series_filename=None
        try:
            # app-saved metadata: normal pickle
            meta_data = np.load(meta_filename)

            dn = os.path.dirname(meta_filename)
            series_filename = os.path.join(dn, meta_data['series_fname'])
            series_data = np.load(series_filename)
        except:
            # maybe it's server-saved metadata
            meta_data = load_zipped_pickle(meta_filename)

            dn = os.path.dirname(meta_filename)
            series_filename = os.path.join(dn, server_series_filename(meta_data))
            series_data = load_zipped_pickle(series_filename)

        self.meta_filename = meta_filename
        self.meta_data = meta_data
        self.series_data = series_data

    def ppg_fps(self):
        ppg_data = self.series_data['ppg_data']
        ts = ppg_data[:,0]
        if len(ts) < 2:
            return 0.0
        return float(len(ts) - 1) / (ts[-1] - ts[0])

    def ppg_parse(self):
        series_data = self.series_data

        ppg_fps = 30.0
        ppg_data = evenly_resample(series_data['ppg_data'][:,0], series_data['ppg_data'][:,1], target_fps=ppg_fps)
        ts = ppg_data[:,0]
        demean = highpass(highpass(ppg_data[:,1], ppg_fps), ppg_fps)

        return Series(demean, fps=ppg_fps, lpad=-ts[0]*ppg_fps)

    def ppg_parse_beatdetect(self):
        cache_file = os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_beatdet.b')
        if os.path.exists(cache_file):
            return np.load(cache_file)

        ppg = ppg_beatdetect(self.ppg_parse())

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump(ppg, fo)

        return ppg

    def get_result(self):
        """calls HS API /reclassify if necessary (if result not yet cached)."""
        cache_file = os.path.join(AppData.CACHE_DIR, os.path.basename(self.meta_filename) + '_result.b')
        if os.path.exists(cache_file):
            return np.load(cache_file)

        res = classify_results(self.meta_data, self.series_data)

        if os.path.isdir(AppData.CACHE_DIR):
            with open(cache_file, 'wb') as fo:
                pickle.dump(res, fo)

        return res

    def has_diagnosis(self):
        return 'doctor' in self.meta_data and self.meta_data['doctor']['status'] != ''

    def get_cvd_status(self):
        if not self.has_diagnosis():
            return None
        status = self.meta_data['doctor']['status']
        if status == 'cvd':
            return True
        elif status == 'healthy':
            return False
        raise ValueError('unknown CVD status: {} in file {}'.format(status, self.meta_filename))
