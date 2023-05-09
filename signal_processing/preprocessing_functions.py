# Pre-processing functions

from scipy import signal, fft
import numpy as np

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