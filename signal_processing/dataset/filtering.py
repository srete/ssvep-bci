import pickle
import numpy as np
import matplotlib.pyplot as plt
from signal_processing.preprocessing_functions import *
from scipy import signal, fft


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
    'PO7': 0,
    'PO3': 1,    
    'O1': 2,
    'POZ': 3,
    'OZ': 4,  # BEST
    'PO4': 5,
    'O2': 6,
    'PO8': 7
}

EEG_CHN_INV = {v: k for k, v in EEG_CHN.items()}

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
FREQ = {i: freqs[i] for i in range(len(freqs))}
print('Frequencies: ', freqs)
data = data['data'] 
data = np.transpose(data, (2, 3, 0, 1))  # shape = (n_frequencies, n_blocks, n_channels, n_samples)

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

#plt.figure()
# Notch filter
quality_factor = 20
notch_freq = 60.0  # Hz
noisy_data = np.apply_along_axis(remove_dc_offset, -1, data)
# plt.subplot(311)
# plt.plot(t, noisy_data[0, 0, 0, :], label='noisy')
data_notch = np.apply_along_axis(notch_filter, -1, noisy_data, fs, notch_freq, quality_factor)
# plt.subplot(312)
# plt.plot(t, data_notch[0, 0, 0, :], label='notch')
# Chebyshev bandpass filter
low=5.75  # Hz
high=13.15  # Hz
order=2
rp=0.3
data_filtered = np.apply_along_axis(cheby_filter, -1, data_notch, fs, low, high, order, rp)
#plt.subplot(313)
#plt.plot(t, data_filtered[0, 0, 0, :], label='filtered')
#plt.show()

data_fft = np.apply_along_axis(signal_fft, -1, data_filtered, fs)
fft_fs = data_fft[0, 0, 0, 1, :]
data_fft = data_fft[:, :, :, 0, :]

print(data_fft.shape)
#plt.plot(fft_fs, data_fft[-1, 0, data_fft.shape[2]-4, :])
#plt.show()

# function to plot nth trial, all frequencies, all channels
def plot_trial(n, data, x_axis, FREQ, eeg_idx):
    print(data.shape)
    n_fs, n_trial, n_chn, n_samples = data.shape
    fig, axs = plt.subplots(n_fs, n_chn, figsize=(15, 15))
    fig.tight_layout(pad=3.0)

    for fs_cnt in range(n_fs):
        for chn_cnt in range(n_chn):
            axs[fs_cnt, chn_cnt].plot(x_axis, data[fs_cnt, n, chn_cnt, :])
            axs[fs_cnt, chn_cnt].set_title('Freq: ' + str(FREQ[fs_cnt]) + ' Hz, Chn: ' + str(EEG_CHN_INV[eeg_idx[chn_cnt]]))
            axs[fs_cnt, chn_cnt].grid()
            # set xlim to 0-30 Hz
            axs[fs_cnt, chn_cnt].set_xlim([4, 15])

    plt.show()


chn_idx = [i for i in range(data_fft.shape[2]-4, data_fft.shape[2])]
plot_trial(0, data_fft[data_fft.shape[0]-4:, :, data_fft.shape[2]-4:, :], fft_fs, freqs[data_fft.shape[0]-4:], chn_idx)


'''
# first 2 seconds of the first block of the first channel (OZ electrode)
oz_eeg = remove_dc_offset(eeg[-1, 0, EEG_CHN['OZ'], :])
#po7_eeg = remove_dc_offset(eeg[0, 0, EEG_CHN['PO7'], :])

###
To denoise our signal, we first apply a 60 Hz notch filter to remove the EMG noise. 
Then, to isolate frequencies of interest, we apply a 2nd order Chebyshev bandpass filter with a ripple 
of 0.3 dB to the range roughly corresponding to flashing frequencies (5.75-13.15Hz). 
We tuned hyperparameters to optimize performance. We experimented with signal smoothing and a procedure to reject 
channels based on an RMS threshold but found no significant performance benefits. In the end, we found only 1 
channel is sufficient and best for our purposes.
###
 
# Notch filter
quality_factor = 20
notch_freq = 60.0  # Hz
oz_eeg_notch = notch_filter(oz_eeg, fs, notch_freq, quality_factor)
# Chebyshev bandpass filter
low=5.75  # Hz
high=13.15  # Hz
order=2
rp=0.3
oz_eeg_filtered = cheby_filter(oz_eeg_notch, fs, low, high, order, rp)

# Plotting noisy signal and filtered signal
fig = plt.figure(figsize=(8, 6))
plt.subplot(311)
plt.plot(t, oz_eeg, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Noisy Signal')
 
# Plot notch-filtered version of signal
plt.subplot(312)
# Plot output signal of notch filter
plt.plot(t, oz_eeg_notch)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Notch Filter')

plt.subplot(313)
plt.plot(t, oz_eeg_filtered)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Notch + Chebyshev Filter')
plt.subplots_adjust(hspace=0.33)
fig.tight_layout()
plt.show()

# FFT of filtered signal
# Compute FFT of the filtered signal with scipy library
oz_eeg_fft, fft_freqs = signal_fft(oz_eeg_filtered, fs)
plt.plot(fft_freqs, oz_eeg_fft)
plt.grid()
plt.show()

# Plot magnitude response of  filters
# fig = plt.figure()
# plt.subplot(211)
# plt.plot(freq_notch, 20 * np.log10(abs(h_notch)), 'r', label='Bandpass filter', linewidth='2')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Notch Filter')

# plt.subplot(212)
# plt.plot(freq_cheby, 20 * np.log10(abs(h_cheby)), 'r', label='Bandpass filter', linewidth='2')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.title('Notch Filter')
# plt.grid()
# plt.subplots_adjust(hspace=0.5)
# fig.tight_layout()
# plt.show()
'''