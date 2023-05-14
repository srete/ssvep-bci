# Pre-processing functions

from scipy import signal, fft
import numpy as np
import os
import pandas as pd

def format_data(data_folder):
    # get all csv files in data_folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    data = []
    min_shapes_per_trial = []
    for f in csv_files:
        data_path = os.path.join(data_folder, f)
        df = pd.read_csv(data_path)
        #print('DF dimenstions: ', df.shape)
        # remove unnamed column
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]        
        # group by blink_freq, and convert each group to numpy array witout blink_freq column
        df = df.groupby('blink_freq').apply(lambda x: x.to_numpy()[:, 1:].transpose())
        # get blink_freqs values
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
    data_arr = data_arr.transpose(1, 0, 2, 3)  # ()
    return freqs, data_arr

def remove_dc_offset(data):
    return data - data.mean()

def notch_filter(x, fs, notch_freq=50, quality_factor=20, ampl_response=False):
    # Design a notch filter 
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    # Apply notch filter to the noisy signal
    filtered_x = signal.filtfilt(b, a, x)
    if ampl_response:
        # Compute magnitude response of the designed filter
        freq, h = signal.freqz(b, a, fs=fs)
        return filtered_x, freq, h
    return filtered_x
    

def cheby_filter(x, fs, low_f, high_f, order=1, rp=1, ampl_response=False):
    # Design Chebyshev bandpass filter
    b, a = signal.cheby1(order, rp, [low_f, high_f], btype='bandpass', fs=fs)
    # Apply Chebyshev bandpass filter
    filtered_x = signal.filtfilt(b, a, x)
    if ampl_response:
        # Compute magnitude response of the designed filter
        freq, h = signal.freqz(b, a, fs=fs)
        return filtered_x, freq, h
    return filtered_x

def signal_fft(x, fs):
    Nfft = int(2**np.ceil(np.log2(len(x))))
    x_fft = fft.fft(x, n=Nfft)[0:Nfft//2]
    # Compute the corresponding frequencies
    fft_freqs = fft.fftfreq(Nfft, 1/fs)[:Nfft//2]
    return 2.0/Nfft * np.abs(x_fft), fft_freqs