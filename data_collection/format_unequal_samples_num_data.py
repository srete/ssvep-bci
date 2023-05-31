import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = r'data_collection\recorded_data\2023-05-10\test_session\17-05-48'

# get all csv files in data_folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# empty numpy array to store data
data = []

min_shapes_per_trial = []
for f in csv_files:
    data_path = os.path.join(data_folder, f)
    df = pd.read_csv(data_path)
    print('DF dimenstions: ', df.shape)
    # remove unnamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    #print(df.head())
    
    # group by blink_freq, and convert each group to numpy array witout blink_freq column
    df = df.groupby('blink_freq').apply(lambda x: x.to_numpy()[:, 1:].transpose())

    shapes = [x.shape[-1] for x in df]
    min_shape = min(shapes)
    min_shapes_per_trial.append(min_shape)
    print(len(df.tolist()))
    data.append(df.tolist())

data_list = []
for trail in data:
    trial_data = []
    for freq in trail:
        print(freq.shape)
        freq = freq[:, :min(min_shapes_per_trial)]
        trial_data.append(freq)
    trial_data = np.stack( trial_data, axis=0 )
    print(trial_data.shape)
    data_list.append(trial_data)

# covert array of arrays to numpy array
print(len(data_list))
data_arr = np.stack( data_list, axis=0 )
print(data_arr.shape)
data_arr = data_arr.transpose(1, 0, 2, 3)  # ()
print(data_arr.shape)

# # plot first channel of first trial of first frequency
plt.plot(data_arr[0, 0, 0, :])
plt.show()