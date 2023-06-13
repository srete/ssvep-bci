# Code to merge all dataframes in csv_files folder

import pandas as pd
import glob

session_name = 'ispitanik'
date = '2023-05-30'

csv_files = glob.glob(r'data_collection\recorded_data\{}\{}\data\*.csv'.format(date, session_name))
print('Number of .csv files: ', len(csv_files))

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df.to_csv(r'data_collection\recorded_data\{}\{}\data.csv'.format(date, session_name), index=False)
print(df)

