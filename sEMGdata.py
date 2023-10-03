import torch
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset

# if using multi-processing
from multiprocessing import cpu_count

class BaseData(object):
  def set_num_processes(self, n_proc):
    if (n_proc is None) or (n_proc <= 0):
      self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
    else:
      self.n_proc = min(n_proc, cpu_count())

class sEMGData():
  # Load the matlab file and save the necessary info
  def __init__(self, file_dir="data/subjects_40_vowels_v6.mat"):
    data_all = sio.loadmat(file_dir)
    self.sig_all   = data_all['DATA']    # raw sEMG signals
    self.label_all = data_all['LABEL']
    
    VFI_1      = data_all['SUBJECT_VFI']        
    self.VFI_1 = [VFI_1[i][0][0][0] for i in range(40)]
    
    SUBJECT_ID      = data_all['SUBJECT_ID']
    self.subject_id = [SUBJECT_ID[i][0][0][0] for i in range(40)]
    
    print(f"Total number of subjects: {self.sig_all.shape[0]:>8}")

  # leave-one-subject-out data/label partition 
  def load_data(self, sub_test, sub_normalize=False):
    print(f"Testing Subject ID: {self.subject_id[sub_test]:>14}")
    print(f"Testing Subject VFI-1: {self.VFI_1[sub_test]:>10}")        
    num_signal = np.shape(self.sig_all[sub_test,0])[0]
    x_test = np.zeros((num_signal, 4000, 4))
    for ch in range(4):
      x_test[:,:,ch] = self.sig_all[sub_test,ch]
    if sub_normalize: x_test = (x_test - x_test.mean(axis=(0,1)))/x_test.std(axis=(0,1))
    y_test = self.label_all[sub_test,0].flatten()
    if np.sum(y_test == -1) > 0:
      print(f"# of Testing Samples: {np.sum(y_test == -1):>13}") 
    else:
      print(f"# of Testing Samples: {np.sum(y_test ==  1):>13}")

    x_train = np.zeros((0,4000,4))
    y_train = np.zeros(0)          
    for sub_train in range(40):
      if sub_train != sub_test:
        num_signal = np.shape(self.sig_all[sub_train,0])[0]
        x = np.zeros((num_signal, 4000, 4))
        for ch in range(4):
            x[:,:,ch] = self.sig_all[sub_train,ch]
        if sub_normalize: x = (x - x.mean(axis=(0,1)))/x.std(axis=(0,1))
        y = self.label_all[sub_train,0].flatten()
        x_train = np.concatenate((x_train, x), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)

    print(f"# of Healthy Training Samples: {np.sum(y_train == -1):>5}")
    print(f"# of Fatigued Training Samples: {np.sum(y_train ==  1):>4}")
    
    return x_train, y_train, x_test, y_test

class sEMGDataset(Dataset):
  def __init__(self, signal, label, transform=None, verbose=False):
    self.signal = torch.from_numpy(signal.astype(np.float32))
    self.label  = torch.from_numpy(label.astype(np.float32))

  def __len__(self):
    return np.shape(self.label)[0]
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    signal = self.signal[idx,...]
    label  = self.label[idx,...]
    
    return signal, label