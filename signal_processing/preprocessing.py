from preprocessing_functions import *
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

import matplotlib.pyplot as plt
import pandas as pd

'''
datapath = r'data_collection\recording\first_recording\gui_data\OpenBCISession_2023-05-03_19-37-36\BrainFlow-RAW_2023-05-03_19-37-36_1.csv'

data = DataFilter.read_file(datapath)
data = pd.DataFrame(np.transpose(data))
fs = 200 #Hz
# Use first 5 second of data
data = data.iloc[0:fs*5, :]

board_id = BoardIds.GANGLION_BOARD.value
eeg_chn = BoardShim.get_eeg_channels(board_id)
timestamps_chn = BoardShim.get_timestamp_channel(board_id)

time = data[timestamps_chn].to_numpy()
t = np.arange(0, len(time)/fs, 1/fs)
eeg = data[eeg_chn].to_numpy().transpose()

# Apply function remove_DC to eeg data by columns
eeg = np.apply_along_axis(remove_dc_offset, 1, eeg)
eeg = eeg[[0, 1], :]

print(time.shape, eeg.shape)

plt.plot(time[0:fs*2], eeg[0][0:fs*2], label='ch1')
plt.plot(time[0:fs*2], eeg[1][0:fs*2], label='ch2')
plt.legend()
plt.show()
'''
data_folder = r'data_collection\recorded_data\2023-05-10\test_session\01-25-00'
data = format_data(data_folder)
fs = 200
# First frequency, first trial, all channels except last one (timestamp), all samples
eeg = data[0, 0, :-1, :]
time = data[0, 0, -1, :]
t = np.arange(0, len(time)/fs, 1/fs)

for cnt, ch in enumerate(eeg):
    # Notch filter
    quality_factor = 20
    notch_freq = 50.0  # Hz
    ch_notch = notch_filter(ch, fs, notch_freq, quality_factor)
    # Chebyshev bandpass filter
    low=5.75  # Hz
    high=13.15  # Hz
    order=2
    rp=0.3
    ch_filtered = cheby_filter(ch_notch, fs, low, high, order, rp)

    # Plotting noisy signal and filtered signal
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(311)
    plt.plot(t, ch, color='r', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Noisy Signal')
    
    # Plot notch-filtered version of signal
    plt.subplot(312)
    # Plot output signal of notch filter
    plt.plot(t, ch_notch)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Notch Filter')

    plt.subplot(313)
    plt.plot(t, ch_filtered)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Notch + Chebyshev Filter')
    plt.subplots_adjust(hspace=0.33)
    fig.tight_layout()
    plt.show()

    # FFT of filtered signal
    # Compute FFT of the filtered signal with scipy library
    oz_eeg_fft, fft_freqs = signal_fft(ch_filtered, fs)
    plt.plot(fft_freqs, oz_eeg_fft)
    plt.grid()
    plt.show()

    break