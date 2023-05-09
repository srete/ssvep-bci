from pprint import pprint
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from brainflow.data_filter import DataFilter

import time
import pandas as pd
import numpy as np

# Getting board descriptions
board_id = BoardIds.GANGLION_BOARD.value
print(board_id)
pprint(BoardShim.get_board_descr(board_id))

# Methods
# insert_marker, get_board_data, get_current_board_data, add_streamer, get_sampling_rate
# get_timestamp_channel, get_eeg_channels

# Initizalizing board
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = 'COM4'
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream ()
time.sleep(10)
# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and remove it from internal buffer
board.stop_stream()
board.release_session()

# Demo how to convert it to pandas DF and plot data
eeg_channels = BoardShim.get_eeg_channels(board_id)
df = pd.DataFrame(np.transpose(data))
print('Data From the Board')
print(df.head(10))

# Demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
restored_data = DataFilter.read_file('test.csv')
restored_df = pd.DataFrame(np.transpose(restored_data))
print('Data From the File')
print(restored_df.head(10))