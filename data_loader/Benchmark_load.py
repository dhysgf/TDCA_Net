import numpy as np
from scipy.io import loadmat
from  scipy import signal
def Data_load(path,train_balock=4,test_block=2,ws=0.2):#加载Benchmark_load数据集
    chanel=[47,53,54,55,56,57,60,61,62]#选择的通道
    data = loadmat(path)
    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for i in range(data["data"].shape[2]):#类别
        for j in range(train_balock):#block
            train_dataset.append(data["data"][:, :, i, j])
            train_label.append(i)
    for i in range(data["data"].shape[2]):
        for j in range(test_block):
            test_dataset.append(data["data"][:, :, i, data["data"].shape[3] - (j + 1)])#取最后的test_block个blocks的数据作为测试集
            test_label.append(i)
    if test_block!=0:
        train_dataset = np.array(train_dataset)
        train_dataset = train_dataset[:, chanel, :]
        train_label = np.array(train_label)
        test_dataset = np.array(test_dataset)
        test_dataset = test_dataset[:, chanel, :]
        test_label = np.array(test_label)
        train_dataset = filter_bank(train_dataset)
        test_dataset = filter_bank(test_dataset)
    else:
        train_dataset = np.array(train_dataset)
        train_dataset = train_dataset[:, chanel, :]
        train_label = np.array(train_label)
        test_dataset =np.zeros(train_dataset.shape)
        test_label =np.zeros(train_label.shape)
    return train_dataset,train_label,test_dataset,test_label

def filter_bank(X, Fs=250):
    """
    带50Hz陷波的零相移带通滤波器（减少失真）
    参数:
        X: 输入信号
        Fs: 采样频率 (Hz)
    """
    # 2. 6-80Hz带通滤波
    low_freq = 6  # 低频截止
    high_freq = 80  # 高频截止
    wn = [2 * low_freq / Fs, 2 * high_freq / Fs]
    sos_band = signal.butter(4, wn, btype='bandpass', output='sos')
    filtered_X = signal.sosfiltfilt(sos_band,X, axis=-1)
    return filtered_X
