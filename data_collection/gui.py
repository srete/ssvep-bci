# This app is simple python gui for steady state visual evoked potential (SSVEP) data collection.
# It has 4 squeres with different blinking frequencies.
# User should look at one of them and collect data
# After that data will be saved to csv file.

import tkinter as tk
from threading import Thread
from queue import Queue

import pandas as pd
import math as m
import time

from gui_functions import *

blink_fs = [5, 6, 8, 10]  #Hz
blink_time = 2
pause_time = 0.5
n_trials = 2
pause_between_trials = 2
session_name = 'test_session'

# initialize gui
root = tk.Tk()
root.title('SSVEP Data Collection')
root.geometry('1920x1080')

# blinking frames dimenstions
w = 150
h = 150

stimulus_frame = tk.Frame(root, width=0.25*root.winfo_width(), height=0.25*root.winfo_height(), relief=tk.RAISED, bd=2)
stimulus_frame.grid(row=0, column=0, padx=10, pady=10)

btn_frame = tk.Frame(root, width=0.25*root.winfo_width(), height=0.25*root.winfo_height(), relief=tk.RAISED, bd=2)
btn_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

# 1st squere
frame1 = tk.Frame(stimulus_frame, bg='black', width=w, height=h)
frame1.grid(row=0, column=0, padx=0.5*w, pady=0.5*h)
# 2nd squere
frame2 = tk.Frame(stimulus_frame, bg='black', width=w, height=h)
frame2.grid(row=0, column=1, padx=0.5*w, pady=0.5*h)
# 3rd squere
frame3 = tk.Frame(stimulus_frame, bg='black', width=w, height=h)
frame3.grid(row=1, column=0, padx=0.5*w, pady=0.5*h)
# 4th squere
frame4 = tk.Frame(stimulus_frame, bg='black', width=w, height=h)
frame4.grid(row=1, column=1, padx=0.5*w, pady=0.5*h)

def main():
    frames = [frame1, frame2, frame3, frame4]
    start_recording(frames, blink_fs, blink_time, pause_time, n_trials, pause_between_trials, session_name, text_label)

# text label to show notifications
text_label = tk.Label(btn_frame, text='Text Label')
text_label.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

# add button to start data collection
button = tk.Button(btn_frame, text='Start Data Collection', command=main)
button.grid(row=1, column=0, columnspan=1,  padx=10, pady=10)

# add button to exit gui
button = tk.Button(btn_frame, text='Stop Data Collection', command=root.destroy)
button.grid(row=1, column=1, columnspan=1, padx=10, pady=10)

# run gui
root.mainloop()