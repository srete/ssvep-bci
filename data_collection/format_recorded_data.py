import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = r'data_collection\recorded_data\2023-05-10\test_session\01-25-00'

# get all csv files in data_folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# empty numpy array to store data
data = []

for f in csv_files:
    data_path = os.path.join(data_folder, f)
    df = pd.read_csv(data_path)
    # remove unnamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.head())
    print(df.shape)
    # group by blink_freq, and convert each group to numpy array witout blink_freq column
    df = df.groupby('blink_freq').apply(lambda x: x.to_numpy()[:, 1:].transpose())
    data_curr_trial = np.array(df.tolist())
    print(data_curr_trial.shape)  # (n_freqs, n_channels+1, n_samples)
    #print(data_curr_trial.tolist())
    data.append(data_curr_trial)

# covert array of arrays to numpy array
print(data[1].shape)
data = np.stack( data, axis=0 )
data = data.transpose(1, 0, 2, 3)  # ()
print(data.shape)

# plot first channel of first trial of first frequency
plt.plot(data[0, 0, 0, :])
plt.show()