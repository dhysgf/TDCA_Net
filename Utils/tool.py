import numpy as np
from scipy import signal
from scipy.signal import resample, butter, lfilter
def resample_eeg_data(x, resample_fs):
    resample_list = []
    for channel in range(x.shape[0]):
        resample_list.append(resample(x[channel], resample_fs))
    return np.array(resample_list)
def butter_bandpass_filter(data,Fs, lowcut, highcut, order=6):
    low_lst = lowcut
    high_lst = highcut
    b, a = signal.butter(order, Wn=[2 * low_lst / Fs, 2 * high_lst / Fs], btype='bandpass')
    filter_X = signal.filtfilt(b, a, data, axis=-1)
    return filter_X
def notch_filter(x,fs=1000, freq=50, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    # 应用陷波滤波器
    y_filtered = signal.lfilter(b, a, x)
    return y_filtered
