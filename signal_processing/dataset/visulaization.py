import pickle
import numpy as np
import matplotlib.pyplot as plt

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
plt.figure(figsize=(10, 5))
plt.plot(t[:2*250], eeg[0, 0, EEG_CHN['OZ'], :2*250])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')  # uV?
plt.show()

