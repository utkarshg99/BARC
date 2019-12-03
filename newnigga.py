import numpy as np
import pylab as pl
from biosppy import storage
signal, mdata = storage.load_txt('./ningen.txt')
Fs = mdata['sampling_rate']
N = len(signal)  # number of samples
T = (N - 1) / Fs  # duration
ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
pl.plot(ts, signal, lw=2)
pl.grid()
pl.show()

from biosppy.signals import ecg
out = ecg.ecg(signal=signal, sampling_rate=Fs, show=True)