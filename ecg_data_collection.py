from scipy import signal
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as pl
from biosppy import storage

y, mdata = storage.load_txt('./values.txt')
x, mdata = storage.load_txt('./tme.txt')
f = signal.resample(x=y, num=1000, t=x)

sig = f[0]*5
Fs = mdata['sampling_rate']
out = ecg.ecg(signal=sig, sampling_rate=Fs, show=True)
N = 1000

f_dash, Pxx_den = signal.welch(sig)
pl.semilogy(f_dash, Pxx_den)
# pl.ylim([0.5e-3, 1])
pl.xlabel('frequency [Hz]')
pl.ylabel('PSD [V**2/Hz]')
pl.show()
