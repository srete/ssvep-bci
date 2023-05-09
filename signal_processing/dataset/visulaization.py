import pickle
import numpy as np
import matplotlib.pyplot as plt
from signal_processing.preprocessing_functions import *

EEG_CHN = {
    'PO7': 0,
    'PO3': 1,    
    'O1': 2,
    'POZ': 3,
    'OZ': 4,  # BEST
    'PO4': 5,
    'O2': 6,
    'PO8': 7
}

# 1st participant (S02), frequency conf D

data_folder = r'signal_processing/dataset/raw/s02/'
data_path = data_folder + r's02_typeD.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 5 seconds of data per frequency
tf = 5  #s
fs = 250  #Hz
t = np.arange(0, tf, 1/fs)  #s

freqs = data['freqs']
eeg = data['data'] # original eeg data shape = (n_channels, n_samples, n_frequencies, n_blocks)
print(data.keys())

# transformed data_shape = (n_frequencies, n_blocks, n_channels, n_samples)
eeg = np.transpose(eeg, (2, 3, 0, 1))

# first 2 seconds of the first block of the first channel (OZ electrode)
oz_eeg = remove_dc_offset(eeg[0, 0, EEG_CHN['OZ'], :])
po7_eeg = remove_dc_offset(eeg[0, 0, EEG_CHN['PO7'], :])
# plt.figure(figsize=(10, 5))
# plt.plot(t[:2*250], oz_eeg[:2*250], color='red')
# plt.plot(t[:2*250], po7_eeg[:2*250], color='blue')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')  # uV?
# plt.show()

recorded_data_path = r'data_collection\recording\first_recording\test_eeg_channels_01.csv'
recorded_data = np.loadtxt(recorded_data_path)
rec_tf = recorded_data.shape[0] / fs
print(rec_tf)
rec_t = np.arange(0, rec_tf, 1/fs)
print(recorded_data.shape)
ch0 = recorded_data[:, 0]
ch1 = recorded_data[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(rec_t, ch0, color='red')
plt.plot(rec_t, ch1, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')  # uV?
plt.show()