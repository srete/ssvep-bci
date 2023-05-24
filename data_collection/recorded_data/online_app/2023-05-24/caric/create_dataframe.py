# load csv from all subfolder

import pandas as pd
import os
import pickle

# get all csv in subfolders with glob.glob
import glob
# get all csv files in subfolders
csv_files = glob.glob(r'data_collection\recorded_data\online_app\2023-05-24\caric\**\*.csv')
print(len(csv_files))

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
# save dataframe to csv file
df.to_csv(r'data_collection\recorded_data\online_app\2023-05-24\caric\data.csv', index=False)
# print dataframe
print(df)