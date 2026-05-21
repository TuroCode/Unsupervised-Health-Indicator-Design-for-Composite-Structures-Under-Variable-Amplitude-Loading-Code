# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from Data_Standarization import Filter_Scale_File_Def
from PMT_Evaluation import PMT_report
from tqdm import tqdm

"""
# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Select the file to read
parquet_file = os.path.join(BASE_DIR, "clean_data/Training", "filtered_spec_01_45_old_clean.parquet")
df = pd.read_parquet(parquet_file)

# Get the time and feature column
time = df['t'].values
column_chosen = 'CCNT'
features = Filter_Scale_File_Def(df, [column_chosen])[0][column_chosen].values

# Parameters
window_seconds = 125  # seconds per window"""

# FUNCTION: compute features per time window
def feature_extraction(feature, time, window_seconds=125):
    window_centers = []
    start = 0
    end_time = time[-1]

    sums = []
    averages = []
    standard_deviations = []
    skewnesses = []
    maxima = []
    kurtosi = []

    while start + window_seconds <= end_time:
        idx = (time >= start) & (time < start + window_seconds)
        window_data = feature[idx]
        window_data = np.array(window_data)
        
        n = len(window_data)
        sum_window_data = np.sum(window_data)
        sums.append(sum_window_data)
        
        if n > 1:
            average = sum_window_data / n
            averages.append(average)

            std_dev_help = window_data - average
            std_dev = np.sqrt(1 / (n - 1) * np.sum(std_dev_help**2))
            standard_deviations.append(std_dev)

            max_val = np.max(window_data)
            maxima.append(max_val)

            if std_dev == 0:
                skewnesses.append(0)
                kurtosi.append(0)
            else:
                skewness = 1/n * np.sum((window_data - average)**3) / std_dev**3
                skewnesses.append(skewness)
                kurtosis = 1/n * np.sum((window_data - average)**4) / std_dev**4 - 3
                kurtosi.append(kurtosis)
        else:
            averages.append(np.nan)
            standard_deviations.append(np.nan)
            skewnesses.append(np.nan)
            maxima.append(np.nan)
            kurtosi.append(np.nan)

        window_centers.append(start + window_seconds / 2)
        start += window_seconds

    return sums, window_centers, averages, standard_deviations, skewnesses, maxima, kurtosi



# sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(
#     features, time, window_seconds=window_seconds
# )

"""
# OPTIONAL: plot the sums feature over time
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
axes[0, 0].plot(window_times, sums, color='blue')
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Cummulative Sum")
axes[0, 0].set_title("Cummulative Sum")

axes[0, 1].plot(window_times, averages, color='orange')
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Averages of windows")
axes[0, 1].set_title("Averages")

axes[1, 0].plot(window_times, standard_deviations, color='green')
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Standard Deviations")
axes[1, 0].set_title("Standard deviations of windows")

axes[1, 1].plot(window_times, skewnesses, color='red')
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Skewnesses")
axes[1, 1].set_title("Skewnesses of windows")

axes[2, 0].plot(window_times, maxima, color='purple')
axes[2, 0].set_xlabel("Time (s)")
axes[2, 0].set_ylabel("Maximum")
axes[2, 0].set_title("Maxima of windows")

axes[2, 1].plot(window_times, kurtosi, color='brown')
axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].set_ylabel("Kurtosis")
axes[2, 1].set_title("Kurtosi of windows")



plt.tight_layout()
plt.show()



#=================ADDED SECTION FOR PMT ANALYSIS==========================

data_folder = Path(os.path.join(BASE_DIR, "clean_data/Training"))

matrix = []

for file in data_folder.iterdir():
    df = pd.read_parquet(file)
    time = df['t'].values #Returns array from the t table
    column_chosen = 'CSS' #For analysis of the feature developed
    features = Filter_Scale_File_Def(df, [column_chosen])[0][column_chosen].values #Scales the data

    sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(
        features, time, window_seconds=window_seconds
    )
    matrix.append(sums)  #Sums will be an array of (n_files x n_windows in that file) note that the matrix is not square!




# Pad all rows to the same length to handle files with different durations
max_len = max(len(row) for row in matrix)
matrix_padded = np.array([row + [0] * (max_len - len(row)) for row in matrix])

#matrix_padded=np.array([np.arange(0,100,1),np.arange(0,100,1),np.arange(0,100,1),np.arange(0,-100,-1),np.arange(0,-100,-1)])
#window_seconds=1
#print(matrix_padded)

value = PMT_report(matrix_padded, window_seconds)

print(value)
"""