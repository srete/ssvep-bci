from pprint import pprint
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from brainflow.data_filter import DataFilter

import time
import pandas as pd
import numpy as np
import os

freq = 12
recording_time = 11

session_name = 'teodora'
date = time.strftime('%Y-%m-%d')
#parent_folder = r'data_collection\recorded_data\dioda'
data_path = r'data_collection\recorded_data\dioda\{}\{}'.format(date, session_name)
# create folder with info_path if it doesn't exist
if not os.path.exists(os.path.dirname(data_path)):
    os.makedirs(os.path.dirname(data_path))

# Getting board descriptions
board_id = BoardIds.GANGLION_BOARD.value
#board_id = BoardIds.SYNTHETIC_BOARD.value
#print(board_id)
#print(BoardShim.get_board_descr(board_id))

# Methods
# insert_marker, get_board_data, get_current_board_data, add_streamer, get_sampling_rate
# get_timestamp_channel, get_eeg_channels

# Initizalizing board
#BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = 'COM7'
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream ()
time.sleep(recording_time)
# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and remove it from internal buffer
board.stop_stream()
board.release_session()
eeg_chn = BoardShim.get_eeg_channels(board_id)
timestamps_chn = BoardShim.get_timestamp_channel(board_id)
data = data[eeg_chn+[timestamps_chn]]
print('Board data shape: ', data.shape)


# Demo how to convert it to pandas DF and plot data
df = pd.DataFrame(columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4', 'timestamp'])
print(f'Data shape: {data.shape}')
data = data.transpose()
data = np.concatenate((np.full(shape=(data.shape[0], 1), fill_value=freq), data), axis=1)
print(f'Data shape: {data.shape}')
df = pd.DataFrame(data, columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4', 'timestamp'])
# save data to csv file
# current time
curr_time = time.strftime('%H-%M-%S')
df.to_csv(data_path + '\\' + curr_time + '.csv', index=False)

#data = np.concatenate((np.full(shape=(recorded_data.shape[0], 1), fill_value=freqs[data]), recorded_data), axis=1)
# create dataframe with recorded data
#curr_df = pd.DataFrame(recorded_data, columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4', 'timestamp'])
# add recorded data to dataframe with pandas.concat
#df = pd.concat([df, curr_df], ignore_index=True)
#print('Data From the Board')
#print(df.head(10))

#Demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
#DataFilter.write_file(data, r'data_collection\recording\test.csv', 'w')  # use 'a' for append mode
#restored_data = DataFilter.read_file(r'data_collection\recording\test.csv')
#restored_df = pd.DataFrame(np.transpose(restored_data))
#print('Data From the File')
#print(restored_df.head(10))

# Plot first eeg channel
#import matplotlib.pyplot as plt
#plt.plot(data[eeg_channels[0]])
#plt.show()