from torch.utils.data import Dataset
import torch
from  scipy import signal
import scipy.io
import numpy as np
class getSSVEP12Intra(Dataset):
   def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train",data_root=""):
       super(getSSVEP12Intra, self).__init__()
       self.Nh = 180  # number of trials
       self.Nc = 8    # number of channels
       self.Nt = 1024  # number of time points
       self.Nf = 12    # number of target frequency
       self.Fs = 256   # Sample Frequency
       self.subject = subject  # current subject
       self.data_root=data_root
       self.eeg_data = self.get_DataSub()
       self.label_data = self.get_DataLabel()
       self.num_trial = self.Nh // self.Nf   # number of trials of each frequency
       self.train_idx = []
       self.test_idx = []
       if KFold is not None:
           fold_trial = self.num_trial // n_splits   # number of trials in each fold
           self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

       for i in range(0, self.Nh, self.Nh // self.Nf):
           for j in range(self.Nh // self.Nf):
               if n_splits == 2 and j == self.num_trial - 1:
                   continue    # if K = 2, discard the last trial of each category
               if KFold is not None:  # K-Fold Cross Validation
                   if j not in self.valid_trial_idx:
                       self.train_idx.append(i + j)
                   else:
                       self.test_idx.append(i + j)
               else:                 # Split Ratio Validation
                   if j < int(self.num_trial * train_ratio):
                      self.train_idx.append(i + j)
                   else:
                      self.test_idx.append(i + j)
       self.eeg_data_train = self.eeg_data[self.train_idx]
       self.label_data_train = self.label_data[self.train_idx]
       self.eeg_data_test = self.eeg_data[self.test_idx]
       self.label_data_test = self.label_data[self.test_idx]
       if mode == 'train':
          self.eeg_data = self.eeg_data_train
          self.label_data = self.label_data_train
       elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test
       print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
       print(f'label_data for subject {subject}:', self.label_data.shape)

   def __getitem__(self, index):
       return self.eeg_data[index], self.label_data[index]
   def __len__(self):
       return len(self.label_data)
   # get the single subject data
   def get_DataSub(self):
      subjectfile = scipy.io.loadmat(f'{self.data_root}/Dial/DataSub_{self.subject}.mat')
      samples = subjectfile['Data']
      eeg_data = samples.swapaxes(1, 2)
      eeg_data=self.filter_bank(eeg_data,256)
      eeg_data = np.array(eeg_data)
      eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))
      return eeg_data
   # get the single label data
   def get_DataLabel(self):
      labelfile = scipy.io.loadmat(f'{self.data_root}/Dial/LabSub_{self.subject}.mat')
      labels = labelfile['Label']
      label_data = torch.from_numpy(labels)
      return label_data - 1

   def filter_bank(self, X, Fs=256):
       """
       带50Hz陷波的零相移带通滤波器（减少失真）
       参数:
           X: 输入信号
           Fs: 采样频率 (Hz)
       """
       # 1. 50Hz陷波滤波（去除工频干扰）
       notch_freq = 50  # 陷波中心频率
       Q = 30  # 品质因数，值越大陷波越窄（通常30-50）
       # 计算归一化频率
       w0 = notch_freq / (Fs / 2)
       # 设计陷波滤波器并转换为SOS格式
       b, a = signal.iirnotch(w0, Q)
       sos_notch = signal.tf2sos(b, a)
       # 应用零相移陷波滤波
       X_notched = signal.sosfiltfilt(sos_notch, X, axis=-1)
       # 2. 8-80Hz带通滤波
       low_freq = 6  # 低频截止
       high_freq = 80  # 高频截止
       wn = [2 * low_freq / Fs, 2 * high_freq / Fs]
       sos_band = signal.butter(6, wn, btype='bandpass', output='sos')
       filtered_X = signal.sosfiltfilt(sos_band, X_notched, axis=-1)
       return filtered_X