from pprint import pprint
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from brainflow.data_filter import DataFilter
import time
import pandas as pd
import numpy as np
import os

# SESSION DETAILS:

serial_port = 'COM5'

session_name = 'ime_prezime'
target_freq = 12
freq_on_screen = [7.5, 8.57, 10, 12]
recording_time = 21
fs = 200

EEG_CHN = {
    'O1': 0,
    'Oz': 1,
    'O2': 2,
    'POz': 3
}

additional_information = '\nzuta - 0\nnarandzasta - 1\nbraon - 2\ncrvena - 3\ncrna - REF\nbela - D_G'


# ---main---

date = time.strftime('%Y-%m-%d')
data_path = r'data_collection\recorded_data\online_app' + '\{}\{}\data'.format(date, session_name)
if not os.path.exists(os.path.dirname(data_path)):
    os.makedirs(os.path.dirname(data_path))

board_id = BoardIds.GANGLION_BOARD.value
#board_id = BoardIds.SYNTHETIC_BOARD.value

# Data acquisition
params = BrainFlowInputParams()
params.serial_port = serial_port
board = BoardShim(board_id, params)
board.prepare_session()
recording_start = time.strftime("%H:%M:%S", time.localtime())
n_samples = int(recording_time*fs)
board.start_stream(n_samples)
time.sleep(recording_time)
data = board.get_board_data()
board.stop_stream()
recording_end = time.strftime("%H:%M:%S", time.localtime())
board.release_session()
eeg_chn = BoardShim.get_eeg_channels(board_id)
timestamps_chn = BoardShim.get_timestamp_channel(board_id)
data = data[eeg_chn+[timestamps_chn]]

# Convert data to dataframe
df = pd.DataFrame(columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4', 'timestamp'])
data = data.transpose()
data = np.concatenate((np.full(shape=(data.shape[0], 1), fill_value=target_freq), data), axis=1)
df = pd.DataFrame(data, columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4', 'timestamp'])
csv_path = data_path + '\{}.csv'.format(target_freq)
df.to_csv(csv_path, index=False)

# Save session information
file_path = data_path + '\{}_recording_info.txt'.format(target_freq)
file = open(file_path,'w')
file.write('Session name: ' + session_name + '\n')
file.write('Date: ' + date+ '\n')
file.write('Positions of EEG electrodes: '+ '\n')
CHN_TO_POS = {v: k for k, v in EEG_CHN.items()}
file.write('\t'+ '0: ' + CHN_TO_POS[0]+ '\n')
file.write('\t'+ '1: ' + CHN_TO_POS[1]+ '\n')
file.write('\t'+ '2: ' + CHN_TO_POS[2]+ '\n')
file.write('\t'+ '3: ' + CHN_TO_POS[3]+ '\n')
file.write('Frequencies shown on screen: ' + str(freq_on_screen) + '\n')
file.write('Recording time: ' + str(recording_time) + '\n')
file.write('\t'+'Recording started at: ' + recording_start + '\n')
file.write('\t'+'Recording ended at: ' + recording_end + '\n')
if additional_information != '':
    file.write('\n')
    file.write('Additional information: '+ '\n')
    file.write(additional_information)
file.close()

# Methods
# insert_marker, get_board_data, get_current_board_data, add_streamer, get_sampling_rate
# get_timestamp_channel, get_eeg_channels