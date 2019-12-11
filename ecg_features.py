from scipy import signal
import pandas as pd
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as pl
from biosppy import storage

data = pd.read_csv("night1.csv") 

# print(data[1])
time1 = np.array(data['time'])
raw1 = np.array(data['ecg'])
# print((time))
time = time1.astype(np.float)
raw = raw1.astype(np.float)
# num_data_points = 1.1 * time.shape[0]
max_time = time[time.shape[0]-1]
sampling_freq = 100
num_data_points = sampling_freq* max_time/1000 * 1.1
num = int(num_data_points)
# print(num_data_points)
# print(num)
# print(1)
f = signal.resample(x=raw, num=num, t=time)
sig = f[0]*5
print(raw.shape)
out = ecg.ecg(signal=sig, sampling_rate=sampling_freq, show=True)
# print(type(ecg))
# pl.plot(time,ecg)
# pl.show()
# print(time(1))