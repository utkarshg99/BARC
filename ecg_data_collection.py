from scipy import signal
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as pl
from biosppy import storage
import pandas

y, mdata = storage.load_txt('./values.txt')
x, mdata = storage.load_txt('./tme.txt')
f = signal.resample(x=y, num=80000, t=x)

var = pandas.read_csv('anshul_02.csv')
x = np.array(var['time(milli)'])
y = np.array(var['Volt'])
sig = f[0]*5
Fs = mdata['sampling_rate']
out = ecg.ecg(signal=sig, sampling_rate=Fs, show=True)

r_peak = out[2] #stores

print(out[2])

f_dash, Pxx_den = signal.welch(sig)
print(f_dash.shape, Pxx_den.shape)
pl.semilogy(f_dash, Pxx_den)
# pl.ylim([0.5e-3, 1])
# pl.plot(f_dash, Pxx_den)
pl.xlabel('frequency [Hz]')
pl.ylabel('PSD [V**2/Hz]')
pl.show()
