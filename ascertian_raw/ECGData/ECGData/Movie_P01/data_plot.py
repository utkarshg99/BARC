from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from biosppy import storage
import numpy as np


a = loadmat('ECG_Clip6.mat')
raw = a['Data_ECG']
# print(raw.shape)
# plt.plot(raw[0], raw[2])
# plt.show()
#print(raw[][1])
channel1=raw[0:10000,1:2]/5
timestamp=raw[0:10000,0:1]
# print(channel1)
# plt.plot(timestamp, channel1)
# plt.show()
max_time = timestamp[timestamp.shape[0]-1]
# print(max_time)
sampling_freq = 250 
num_data_points = sampling_freq* max_time/1000 * 1.1
num = int(num_data_points)

c1 = np.reshape(channel1, -1)
t1 = np.reshape(timestamp,-1)
print(t1.shape)
# f = signal.resample(x=c1, num=num, t=t1)
# sig = f[0]*5

# c1.flatten('F')

# c1 = np.transpose(c1)
# print(c1.shape)
out = ecg.ecg(signal=c1, sampling_rate=sampling_freq, show=True)
# print(type(ecg))
