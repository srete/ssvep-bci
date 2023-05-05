
import tkinter as tk
from threading import Thread
from queue import Queue

import pandas as pd
import math as m
import time
import numpy as np


# function to make squere blink
def blink(frame, blink_freq, blink_time):
    if blink_time > 0:
        bg_color = frame.cget('bg')
        frame.config(bg='gray')
        time.sleep(1/(2*blink_freq))
        frame.config(bg=bg_color)
        time.sleep(1/(2*blink_freq))
        blink(frame, blink_freq, blink_time-1/blink_freq)

def blink_all(frames, freqs, blink_time, pause_time, q1, q2, text_label):
    try:
        for i, frame in enumerate(frames):
            q1.put(i)
            data = q2.get()
            if data:
                text_label.config(text=f'Recording data from squere {i+1}')
                frame.config(highlightbackground='red', highlightthickness=3)
                blink(frame, freqs[i], blink_time)
                frame.config(highlightbackground='black', highlightthickness=0)
            text_label.config(text=f'Pause for {pause_time} seconds')
            time.sleep(pause_time)
        return
    except:
        return

def record_data(q1, q2, text_label):
    try:
        df = pd.DataFrame(columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4'])
        while True:
            # pandas dataframe with columns: blink_freq, ch1, ch2, ch3, ch4
            data = q1.get()
            text_label.config(text=f'Prepare to record data from squere {data+1}')
            # prepare recording
            time.sleep(2)
            # recoring is ready, send signal to blink and start recording
            q2.put(1)
            recorded_data = np.random.rand(4, 256)
            recorded_data = recorded_data.transpose()
            recorded_data = np.concatenate((np.full(shape=(recorded_data.shape[0], 1), fill_value=data), recorded_data), axis=1)
            # create dataframe with recorded data
            curr_df = pd.DataFrame(recorded_data, columns=['blink_freq', 'ch1', 'ch2', 'ch3', 'ch4'])
            # add recorded data to dataframe with pandas.concat
            df = pd.concat([df, curr_df], ignore_index=True)
            if data == 3:
                # save data to csv
                df.to_csv(f'data_collection\\recorded_data\\data_{time.time()}.csv')
                return
    except:
        return

def start_recording(frames, freqs, blink_time, pause_time, text_label):
    q1 = Queue()
    q2 = Queue()
    # create thread
    blink_thread = Thread(target=blink_all, args=(frames, freqs, blink_time, pause_time, q1, q2, text_label))
    # second thread to record data
    record_thread = Thread(target=record_data, args=(q1, q2, text_label))
    # start both threads
    record_thread.start()
    blink_thread.start()