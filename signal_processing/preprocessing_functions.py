import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.signal import spectrogram


def format_data(data_folder):

    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    data = []
    min_shapes_per_trial = []
    for f in csv_files:
        data_path = os.path.join(data_folder, f)
        df = pd.read_csv(data_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.groupby('blink_freq').apply(lambda x: x.to_numpy()[:, 1:].transpose())
        freqs = df.index.values
        shapes = [x.shape[-1] for x in df]
        min_shapes_per_trial.append(min(shapes))
        data.append(df.tolist())

    data_list = []
    for trail in data:
        trial_data = []
        for freq in trail:
            freq = freq[:, :min(min_shapes_per_trial)]
            trial_data.append(freq)
        trial_data = np.stack( trial_data, axis=0 )
        data_list.append(trial_data)

    data_arr = np.stack( data_list, axis=0 )
    data_arr = data_arr.transpose(1, 0, 2, 3)

    return freqs, data_arr


def remove_dc_offset(data):
    return data - data.mean()


def notch_filter(x, fs, notch_freq=50, quality_factor=20, ampl_response=False):

    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    filtered_x = signal.filtfilt(b, a, x)
    if ampl_response:
        freq, h = signal.freqz(b, a, fs=fs)

        return filtered_x, freq, h
    
    return filtered_x
    

def cheby_filter(x, fs, low_f, high_f, order=1, rp=1, ampl_response=False):

    b, a = signal.cheby1(order, rp, [low_f, high_f], btype='bandpass', fs=fs)
    filtered_x = signal.filtfilt(b, a, x)
    if ampl_response:
        freq, h = signal.freqz(b, a, fs=fs)

        return filtered_x, freq, h
    
    return filtered_x


def filter_data(data, t, fs, plotting = True):

    # removing DC offset
    noisy_data = np.apply_along_axis(remove_dc_offset, -1, data)
    # Notch filter
    quality_factor = 20
    notch_freq = 50.0  # Hz
    data_notch = np.apply_along_axis(notch_filter, -1, noisy_data, fs, notch_freq, quality_factor)
    # Chebyshev bandpass filter
    low = 6  # Hz
    high = 32  # Hz
    print('Bandpass filter from {} to {} Hz'.format(low, high))
    order = 4
    rp = 0.1
    data_filtered = np.apply_along_axis(cheby_filter, -1, data_notch, fs, low, high, order, rp)

    t = t[0:data_filtered.shape[-1]]

    if plotting:
        plt.figure()
        plt.subplot(311)
        plt.plot(t, noisy_data[0, 0, 0, :], label='original EEG data')
        plt.legend()
        plt.subplot(312)
        plt.plot(t, data_notch[0, 0, 0, :], label='notch filter')
        plt.legend()
        plt.subplot(313)
        plt.plot(t, data_filtered[0, 0, 0, :], label='notch + bandpass filter')
        plt.legend()
        plt.show()

    return data_filtered


def signal_fft(x, fs):
    Nfft = int(2**np.ceil(np.log2(len(x))))
    x_fft = fft.fft(x, n=Nfft)[0:Nfft//2]
    fft_freqs = fft.fftfreq(Nfft, 1/fs)[:Nfft//2]
    return 2.0/Nfft * np.abs(x_fft), fft_freqs


def plot_spectrogram(data, fs, freq_idx, freqs):
    
    window_size = int(fs*0.1)  # 0.1 seconds
    overlap = int(window_size * 0.5)  # 50% overlap
    f, t, Sxx = spectrogram(data, nperseg = window_size, fs = fs, noverlap = overlap, nfft=1024)
    spectrogram_data = np.array(Sxx)
    plt.figure()
    plt.pcolormesh(t, f, spectrogram_data)
    plt.colorbar()
    plt.ylim([6, 32])
    plt.title(f'Spectrogram of {freqs[freq_idx]} Hz')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def split_data(data, window_size, overlap):

    n_samples = data.shape[-1]
    n_windows = int((n_samples - window_size)/(window_size - overlap) + 1)
    windows = []
    for i in range(n_windows):
        windows.append(data[:, :, i*(window_size - overlap):i*(window_size - overlap) + window_size])

    return np.stack(windows, axis=-2)


def plot_trial(n, data, x_axis, FREQ, CHN_TO_POS):

    if data.ndim == 4:
        n_fs, n_trial, n_chn, n_samples = data.shape
        fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
        fig.tight_layout(pad=3.0)
        for fs_cnt in range(n_fs):
            for chn_cnt in range(n_chn):
                data[fs_cnt, n, chn_cnt, :]
                axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, n, chn_cnt, :])
                axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(round(FREQ[fs_cnt], 2)) + ' Hz, Chn: ' + CHN_TO_POS[chn_cnt])
                axs[fs_cnt, chn_cnt].set_xlim([0, 35])

    elif data.ndim == 3:
        n_fs, n_chn, n_samples = data.shape
        fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
        fig.tight_layout(pad=3.0)
        for fs_cnt in range(n_fs):
            for chn_cnt in range(n_chn):
                axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, chn_cnt, :])
                axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(round(FREQ[fs_cnt], 2)) + ' Hz, Chn: ' + CHN_TO_POS[chn_cnt])
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

