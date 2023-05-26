from preprocessing_functions import *
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import pandas as pd
import pickle

from cca import *

# Set fontsizes for plotting
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

EEG_CHN = {
    'O2': 0,
    'Oz': 1,
    'O1': 2,
    'POz': 3
}

def filter_data(noisy_data):
    fs = 200
    # Notch filter
    quality_factor = 20
    notch_freq = 50.0  # Hz
    noisy_data = remove_dc_offset(noisy_data)
    data_notch= notch_filter(noisy_data, fs, notch_freq, quality_factor)
    # Chebyshev bandpass filter
    low=4.5  # Hz
    high=20  # Hz
    order=3
    rp=0.1
    data_filtered  = cheby_filter(data_notch, fs, low, high, order, rp, ampl_response=False)
    return data_filtered

# function to split data into windows
def split_data(data, window_size, overlap):
    n_samples = data.shape[-1]
    n_windows = int((n_samples - window_size)/(window_size - overlap) + 1)
    windows = []
    for i in range(n_windows):
        windows.append(data[:, :, i*(window_size - overlap):i*(window_size - overlap) + window_size])
    return np.stack(windows, axis=-2)

f = '12'
#data_folder = r'data_collection\recorded_data\dioda\2023-05-16\teodora\freq_{}HZ'.format(f)
#data_folder = r'data_collection\recorded_data\online_app\freq_12HZ'
data_folder = r'data_collection\recorded_data\online_app\2023-05-24\caric\all'
freqs, data = format_data(data_folder)
fs = 200
FREQ = {i: freqs[i] for i in range(len(freqs))}
time = data[0, :, -1, 1*fs:]
t = np.arange(0, len(time[0])/fs, 1/fs)
data = data[:, :, :-1, 1*fs:]
print(time.shape, data.shape)


'''
data_folder = r'signal_processing/dataset/raw/s02/'
data_path = data_folder + r's02_typeD.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 5 seconds of data per frequency
tf = 5  #s
fs = 250  #Hz
t = np.arange(0, tf, 1/fs)  #s

freqs = data['freqs']
FREQ = {i: freqs[i] for i in range(len(freqs))}
print('Frequencies: ', freqs)
data = data['data'] 
data = np.transpose(data, (2, 3, 0, 1))  # shape = (n_frequencies, n_blocks, n_channels, n_samples)
'''

#plt.figure()
# Notch filter
quality_factor = 20
notch_freq = 50.0  # Hz
noisy_data = np.apply_along_axis(remove_dc_offset, -1, data)
#plt.subplot(311)
#plt.plot(t, noisy_data[0, 0, 0, :], label='noisy')
data_notch = np.apply_along_axis(notch_filter, -1, noisy_data, fs, notch_freq, quality_factor)
#plt.subplot(312)
#plt.plot(t, data_notch[0, 0, 0, :], label='notch')
# Chebyshev bandpass filter
low = 6
#low=np.min(freqs) - 1  # Hz
high = 32
#high=np.max(freqs) + 1  # Hz
print('Filtering from {} to {} Hz'.format(low, high))
order=4
rp=0.1
data_filtered = np.apply_along_axis(cheby_filter, -1, data_notch, fs, low, high, order, rp)
#plt.subplot(313)
#plt.plot(t, data_filtered[0, 0, 0, :], label='filtered')
#plt.show()

# data_filtered = np.squeeze(data_filtered)
# print('SHAPE: ', data_filtered.shape)

splited_data = split_data(np.squeeze(data_filtered), 4*fs, 0)
print('SPLITED: ', splited_data.shape)
data_filtered = splited_data.transpose((0, 2, 1, 3))
#print('SHAPE: ', data_filtered.shape)

# average data acros windows
avr_data = np.mean(splited_data, axis=-2)
print(avr_data.shape)

# avr_data = np.mean(data_filtered, axis=1)
# print(f'Average data shape: {avr_data.shape}')

data_fft = np.apply_along_axis(signal_fft, -1, data_filtered, fs)
fft_fs = data_fft[0, 0, 0, 1, :]
data_fft = data_fft[:, :, :, 0, :]
#fft_avg = np.apply_along_axis(signal_fft, -1, avr_data, fs)
#fft_avg_fs = fft_avg[0, 0, 0, 1, :]
#fft_avg = fft_avg[:, :, :, 0, :]

refresh_rate = 60  # Hz
target_freqs =np.array([refresh_rate/i for i in range(3, 11)])  # 6-20 Hz
target_freqs = freqs
ref = np.array([get_cca_reference_signals(data_filtered.shape[-1], f, fs) for f in target_freqs])


# function to show spectrogram of one freguency, one trial, one channel. Use scipy.signal.spectrogram
def plot_spectrogram(data, fs):
    # shape of data is (n_trials, n_channels, n_samples)
    f, t, Sxx = spectrogram(data[0, 0, 0, :], fs)
    #spectrogram_data.append(Sxx)
    #spectrogram_data = np.stack(spectrogram_data, axis=0)
    spectrogram_data = np.array(Sxx)
    print(spectrogram_data.shape)
    plt.figure()
    plt.pcolormesh(t, f, spectrogram_data)
    #show colorbar
    plt.colorbar()

    plt.ylim([0, 10])
    #plt.title(f'Spectrogram of {freq} Hz, trial {n}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


'''
test_data = np.array([np.sin(2*np.pi*f*t) for f in target_freqs])
test_data = np.expand_dims(test_data, axis=0)
test_data = np.expand_dims(test_data, axis=0)
test_data = np.transpose(test_data, (2, 0, 1, 3))

print('Test data shape: ', test_data.shape)
'''
predictions, labels = cca_classify(data_filtered, ref)

print('\n'+'-'*20)
print('Labels\t\tPredictions')
print('-'*20)
for i in range(len(predictions)):
    print(f'{freqs[labels[i]] : .2f}\t\t{target_freqs[predictions[i]] : .2f}')
print('-'*20+'\n')


#print(data_fft.shape)
#plt.plot(fft_fs, data_fft[1, 0, 2, :])
#plt.show()

# function to plot nth trial, all frequencies, all channels
def plot_trial(n, data, x_axis):
    print(data.shape)
    if data.ndim == 4:
        n_fs, n_trial, n_chn, n_samples = data.shape
        fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
        fig.tight_layout(pad=3.0)
        for fs_cnt in range(n_fs):
            for chn_cnt in range(n_chn):
                data[fs_cnt, n, chn_cnt, :]
                axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, n, chn_cnt, :])
                axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(round(FREQ[fs_cnt], 2)) + ' Hz, Chn: ' + str(chn_cnt))
                axs[fs_cnt, chn_cnt].set_xlim([0, 35])
    elif data.ndim == 3:
        n_fs, n_chn, n_samples = data.shape
        fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
        fig.tight_layout(pad=3.0)
        for fs_cnt in range(n_fs):
            for chn_cnt in range(n_chn):
                axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, chn_cnt, :])
                axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(round(FREQ[fs_cnt], 2)) + ' Hz, Chn: ' + str(chn_cnt))
                axs[fs_cnt, chn_cnt].set_xlim([0, 35])

    plt.show()


def plot_one_freq(n, data, x_axis, freq):
    print(data.shape)
    n_trial, n_chn, n_samples = data.shape
    fig, axs = plt.subplots(1, n_chn, figsize=(15, 5))
    fig.tight_layout(pad=3.0)
    for chn_cnt in range(n_chn):
        axs[chn_cnt].plot(x_axis, data[n, chn_cnt, :])
        axs[chn_cnt].set_title(f'{round(freq, 2)} Hz, Chn: ' + str(chn_cnt))
        axs[chn_cnt].set_xlim([0, 35])
    plt.show()

#plot_trial(0, fft_avg, fft_avg_fs)
plot_trial(0, data_fft, fft_fs)
#plot_one_freq(0, data_fft[0, :, :, :], fft_fs, freqs[0])

#plot_spectrogram(data_filtered, fs)