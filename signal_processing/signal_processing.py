from preprocessing_functions import *
from cca import *
import matplotlib.pyplot as plt
import numpy as np


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Ovo promeniti da se ne unosi rucno

EEG_CHN = {
    'O2': 0,
    'Oz': 1,
    'O1': 2,
    'POz': 3
}

CHN_TO_POS = {v: k for k, v in EEG_CHN.items()}


# ---main---

data_folder = r'data_collection\recorded_data\online_app\2023-05-30\vukasin_sp'
refresh_rate = 60   # Hz
fs = 200    # Hz

freqs, data = format_data(data_folder)
print(f'Frequencies: {freqs}')
time = data[0, :, -1, 1*fs:]
t = np.arange(0, len(time[0])/fs, 1/fs)
data = data[:, :, :-1, 1*fs:]
print('SHAPE (TIME):', time.shape)
print('SHAPE (ORIGINAL DATA):', data.shape, 'shape = (n_freqs, n_trials, n_channels, n_samples)')

data_filtered = filter_data(data, t, fs)
data_filtered = np.squeeze(data_filtered)  # shape = (n_freqs, n_channels, n_samples) if only one trial exists
print('SHAPE (FILTERED DATA): ', data_filtered.shape)

### Plotting spectrogram ###
# for freq_idx in range(data_filtered.shape[0]):
#     plot_spectrogram(data_filtered[freq_idx, EEG_CHN['Oz'], :], fs, freq_idx, freqs)

############### Spliting data ###########################
window_length = input('Enter window length in seconds: ')
window_size = int(window_length)*fs
splited_data = split_data(np.squeeze(data), window_size, 0)
splited_data = filter_data(splited_data, t, fs, plotting=False)
print('SHAPE (SPLITED): ', splited_data.shape)  # shape: (n_freqs, n_channels, n_windows, n_samples)
data_filtered = splited_data.transpose((0, 2, 1, 3))
print('SHAPE (TRANSPOSED): ', data_filtered.shape)  # shape: (n_freqs, n_windows, n_channels, n_samples)
#################################################

data_fft = np.apply_along_axis(signal_fft, -1, data_filtered, fs)
fft_fs = data_fft[0, 0, 0, 1, :]
data_fft = data_fft[:, :, :, 0, :]
print('SHAPE (DATA FFT): ', data_fft.shape) # shape: (n_freqs, n_windows, n_channels, n_samples)

trial = input('Enter the index of window for spectra plotting: ')
plot_trial(int(trial), data_fft, fft_fs,  freqs, CHN_TO_POS)
# plot_one_freq(0, data_fft[0, :, :, :], fft_fs, freqs[0])

print('\n' + 'CCA:' + '\n')
target_freqs = np.array([refresh_rate/i for i in range(3, 11)])  # 6-20 Hz
target_freqs = freqs
ref = np.array([get_cca_reference_signals(data_filtered.shape[-1], f, fs) for f in target_freqs])

predictions, labels = cca_classify(data_filtered, ref)

print('\n'+'-'*20)
print('Labels\t\tPredictions')
print('-'*20)
for i in range(len(predictions)):
    print(f'{freqs[labels[i]] : .2f}\t\t{target_freqs[predictions[i]] : .2f}')
print('-'*20+'\n')

