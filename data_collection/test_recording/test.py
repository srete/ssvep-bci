import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import mne
# from mne.channels import read_layout

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM7')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    # parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #                     required=True)
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=0xb317)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board
    params.board_id = args.board_id

    print(args)
    board = BoardShim(BoardIds.GANGLION_BOARD, params)
    board.prepare_session()
    board.start_stream ()  # fs=200Hz, buffer_time=1min, 24b
    time.sleep(30)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    #print(data.shape)
    #np.save('test_data_00', data)

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value) # BrainFlow returns uV
    eeg_data = data[eeg_channels, :]

    plt.plot(eeg_data[0], label='ch1')
    plt.plot(eeg_data[1], label='ch2')
    # plt.plot(eeg_data[2], label='ch3')
    # plt.plot(eeg_data[3], label='ch4')
    plt.legend()
    plt.show()

    df = pd.DataFrame(np.transpose(eeg_data))
    print('Data From the Board')
    #print(df.head(10))

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    DataFilter.write_file(eeg_data, 'test-eeg-2.csv', 'w')  # use 'a' for append mode
    restored_data = DataFilter.read_file('test-eeg-2.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))
    #print('Data From the File')
    #print(restored_df.head(10))


if __name__ == "__main__":
    main()