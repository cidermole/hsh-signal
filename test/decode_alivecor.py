import numpy as np
from hsh_signal.alivecor import decode_alivecor, AlivecorFilter, ChunkDataSource
import matplotlib.pyplot as plt

# test decode_alivecor

def stretch(T, f, fps=48000):
    # T in samples, NOT sec
    t = np.arange(T) / float(fps)
    sig = np.cos(2*np.pi*f*t)
    return list(sig)


def chirp(T, f1, f2, fps=48000):
    # T in samples, NOT sec
    t = np.arange(T) / float(fps)
    f = f1 + (t / t[-1]) * (f2 - f1)
    sig = np.cos(2*np.pi*f*t)
    return list(sig)

"""
plt.plot(chirp(5.0*30, 0.5, 3.0, fps=30))
plt.show()
"""


f_center = 18.8e3
f_shift = 100

sig = []
stretches = [(1.5, 0), (0.1, 1), (0.2, -1), (0.3, 1), (0.4, -1), (1.0, 0)]
fps=48000

for dur_sec, f_shift_sign in stretches:
    sig += stretch(int(fps*dur_sec), f=f_center+f_shift_sign*f_shift, fps=fps)

sig = np.array(sig) * 1e-3  # amplitude correct

chrp = chirp(0.5*fps, 100, 8000, fps=fps)
#plt.plot(chrp)
#plt.show()

sig[len(chrp):2*len(chrp)] = np.array(chrp) * 1e-3
sig[2*len(chrp):3*len(chrp)] = np.array(chrp) * 1e-3
#sig[]

#plt.plot(sig)
#plt.show()


from scipy.io import wavfile
wavfile.write('all.wav', fps, (sig * 32767 * 1e3).astype(np.int16))

wavfile.write('sync.wav', fps, (sig[len(chrp):3*len(chrp)] * 32767 * 1e3).astype(np.int16))

sig_ecg = decode_alivecor(sig)


mic = ChunkDataSource(data=sig, batch_size=179200, sampling_rate=fps)
af = AlivecorFilter(mic.sampling_rate)

# plus hilbert's 32
print 'delays', af.lowpass.delay, af.bandreject.delay, (af.lowpass.delay + af.bandreject.delay + 32)

print 'delay sec', (af.lowpass.delay + af.bandreject.delay + 32)/float(fps)

plt.plot(np.arange(len(sig_ecg)) / float(fps), sig_ecg)
plt.show()
