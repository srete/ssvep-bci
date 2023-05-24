from preprocessing_functions import *
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

import matplotlib.pyplot as plt
import pandas as pd
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
    'Pz': 0,
    'Oz': 1,
    'O1': 2,
    'O2': 3
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
        windows.append(data[:, :, :, i*(window_size - overlap):i*(window_size - overlap) + window_size])
    return np.stack(windows, axis=-2)

f = '12'
#data_folder = r'data_collection\recorded_data\dioda\2023-05-16\teodora\freq_{}HZ'.format(f)
#data_folder = r'data_collection\recorded_data\online_app\freq_12HZ'
data_folder = r'data_collection\recorded_data\dioda\2023-05-16\teodora'
freqs, data = format_data(data_folder)
fs = 200
FREQ = {i: freqs[i] for i in range(len(freqs))}

# First frequency, first trial, all channels except last one (timestamp), all samples
time = data[0, :, -1, 200:]
t = np.arange(0, len(time[0])/fs, 1/fs)
data = data[:, :, :-1, 200:]
print(time.shape, data.shape)


plt.figure()
fs = 200
# Notch filter
quality_factor = 20
notch_freq = 50.0  # Hz
noisy_data = np.apply_along_axis(remove_dc_offset, -1, data)
plt.subplot(311)
plt.plot(t, noisy_data[0, 0, 0, :], label='noisy')
data_notch = np.apply_along_axis(notch_filter, -1, noisy_data, fs, notch_freq, quality_factor)
plt.subplot(312)
plt.plot(t, data_notch[0, 0, 0, :], label='notch')
# Chebyshev bandpass filter
low = 5
#low=np.min(freqs) - 1  # Hz
high = 13
#high=np.max(freqs) + 1  # Hz
print('Filtering from {} to {} Hz'.format(low, high))
order=3
rp=0.1
data_filtered = np.apply_along_axis(cheby_filter, -1, data_notch, fs, low, high, order, rp)
plt.subplot(313)
plt.plot(t, data_filtered[0, 0, 0, :], label='filtered')
plt.show()

splited_data = split_data(data_filtered, 400, 0)
print(data_filtered.shape)
# average data acros windows
avr_data = np.mean(splited_data, axis=-2)
print(avr_data.shape)

# avr_data = np.mean(data_filtered, axis=1)
# print(f'Average data shape: {avr_data.shape}')

data_fft = np.apply_along_axis(signal_fft, -1, data_filtered, fs)
fft_fs = data_fft[0, 0, 0, 1, :]
data_fft = data_fft[:, :, :, 0, :]
fft_avg = np.apply_along_axis(signal_fft, -1, avr_data, fs)
fft_avg_fs = fft_avg[0, 0, 0, 1, :]
fft_avg = fft_avg[:, :, :, 0, :]

refresh_rate = 60  # Hz
#target_freqs =np.array([refresh_rate/i for i in range(3, 11)])  # 6-20 Hz
target_freqs = freqs
ref = np.array([get_cca_reference_signals(data_filtered.shape[-1], f, 200) for f in target_freqs])
predictions, labels = cca_classify(data_filtered, ref)
print(target_freqs[predictions])
print(freqs[labels])

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
                axs[fs_cnt, chn_cnt].set_xlim([0, 25])
    elif data.ndim == 3:
        n_fs, n_chn, n_samples = data.shape
        fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
        fig.tight_layout(pad=3.0)
        for fs_cnt in range(n_fs):
            for chn_cnt in range(n_chn):
                axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, chn_cnt, :])
                axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(round(FREQ[fs_cnt], 2)) + ' Hz, Chn: ' + str(chn_cnt))
                axs[fs_cnt, chn_cnt].set_xlim([0, 25])

    plt.show()


def plot_one_freq(n, data, x_axis, freq):
    print(data.shape)
    n_trial, n_chn, n_samples = data.shape
    fig, axs = plt.subplots(1, n_chn, figsize=(15, 5))
    fig.tight_layout(pad=3.0)
    for chn_cnt in range(n_chn):
        axs[chn_cnt].plot(x_axis, data[n, chn_cnt, :])
        axs[chn_cnt].set_title(f'{round(freq, 2)} Hz, Chn: ' + str(chn_cnt))
        axs[chn_cnt].set_xlim([0, 25])
    plt.show()

#plot_trial(0, fft_avg, fft_avg_fs)
#plot_trial(0, data_fft, fft_fs)

#plot_one_freq(0, data_fft[0, :, :, :], fft_fs, freqs[0])

