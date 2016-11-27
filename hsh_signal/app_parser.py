import numpy as np
import matplotlib.pyplot as plt

from hsh_signal.alivecor import decode_alivecor
from signal import evenly_resample, highpass
from heartseries import Series


def parse_app_series(filename):
    series_data = np.load(filename)

    audio_data = series_data['audio_data']
    ecg_raw = decode_alivecor(series_data['audio_data'][:,1])

    # TODO: audio FPS from series_data
    ecg = ecg_raw[::int(48000/300)]
    ecgt = audio_data[:,0][::int(48000/300)]
    ecg_fps = 300

    ecg_delay = int(0.728 * ecg_fps)  # delay from decode_alivecor()
    ecg_sig = ecg[ecg_delay:]
    ecg_ts = ecgt[:-ecg_delay]  # delay is not included in timestamps, only in signal. but length must be the same.

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
