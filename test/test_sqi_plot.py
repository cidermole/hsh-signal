
import glob
import pickle
from os.path import join, basename
from hsh_signal.app_parser import AppData
from hsh_signal.quality import QsqiPPG
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = '/mnt/hsh/src/testdata/datasets/david-getrr2-challenge-2'

data_files = sorted(glob.glob(join(DATA_DIR, '*_ecg_ppg.b')))

#df = data_files[0]
df = data_files[18]

mf = basename(df).replace('_ecg_ppg', '_meta')
print mf
ad = AppData(mf)

from hsh_beatdet.zong import ZongDetector

zd = ZongDetector()
zd.detect(ad.ppg_raw())
ppgz = zd.get_result()

with open(df, 'rb') as fi:
    (ecg, ppg, ppg_raw, ecg_ibs, ppg_ibs) = pickle.load(fi)

# not sure why necessary.
ppg_dt = (ppg_raw.t[0] - ppg.t[0])
ppgz.shift(ppg_dt)

errors = ppg.tbeats[ppg_ibs] - ecg.tbeats[ecg_ibs]

#fig, ax = plt.subplots(4, sharex=True)
#fig, ax = plt.subplots(2, sharex=True)
#ecg.plot(ax[0])
#ppg.plot(ax[0], c='y')

sq = QsqiPPG.from_heart_series(ppgz)

sq.plot()
plt.show()



def gauss(x, t_mu, t_sigma):
    a = 1.0 / (t_sigma * np.sqrt(2 * np.pi))
    y = a * np.exp(-0.5 * (x - t_mu)**2 / t_sigma**2)
    return y


s_min = sq.template
from hsh_signal.quality import SLICE_FRONT

weighting = gauss(np.arange(len(s_min)), len(s_min)*SLICE_FRONT, len(s_min)*0.4)
weighting[:int(len(s_min)*SLICE_FRONT)] = 0.0  # blank weighting the previous beat
weighting = weighting / np.sum(weighting)  # * len(s_min)


plt.plot(sq.template)
plt.plot(-weighting*10)
plt.show()