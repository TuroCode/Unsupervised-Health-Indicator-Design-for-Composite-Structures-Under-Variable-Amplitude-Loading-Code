import numpy as np
from scipy import stats
import duckdb
import matplotlib.pyplot as plt
import os
 
def Filter_Scale_File_Def(X,columns):
    '''Takes a pandas table and returns a table with scaled columns and returns an array of standard deviations and means of the respective columns'''
    std=np.zeros(len(columns))
    mean=np.zeros(len(columns))
    
    for i,column in enumerate(columns):#Normalises the data and returns the scaling coefficients
        X[column]=(X[column]-np.mean(X[column]))/np.std(X[column]) #standarise data for the training set
        std[i]=np.std(X[column])
        mean[i]=np.mean(X[column])
        X[column]=(X[column]-mean[i])/std[i] #Scales the columns based on the values of std and mean attained
    return X,std,mean #note that std and mean are arrays containing scaling factors based on this file given

window_size = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "filtered_data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

window_size = 2  # example

# Loop through files
for file in os.listdir(CLEAN_DIR):
    if file.endswith(".parquet"):
        input_path = os.path.join(CLEAN_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, f"filtered_{file}")

        duckdb.sql(f"""
        COPY (
            WITH base AS (
                SELECT *,
                    floor("t" / {window_size}) AS step
                FROM read_parquet('{input_path}')
            ),
            percentiles AS (
                SELECT *,
                    quantile_disc("E", 0.10) OVER(PARTITION BY step) AS p10,
                    quantile_disc("E", 0.95) OVER(PARTITION BY step) AS p95
                FROM base
            )
            SELECT "t", "E", step
            FROM percentiles
            WHERE "E" >= GREATEST(p10, 1000)
              AND "E" <= p95
              AND "E" < 500000
        )
        TO '{output_path}'
        (FORMAT PARQUET);
        """)

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".parquet"):
        input_path = os.path.join(CLEAN_DIR, file)
        filtered_path = os.path.join(OUTPUT_DIR, f"filtered_{file}")

        # Load data
        df_original = duckdb.sql(f"SELECT * FROM '{input_path}'").df()
        df_filtered = duckdb.sql(f"SELECT * FROM '{filtered_path}'").df()

        # Create plot
        plt.figure(figsize=(12, 6))

        plt.scatter(df_original['t'], df_original['E'],
                    color='orange', alpha=0.4, s=15,
                    label='Original Data (All Hits)')

        plt.scatter(df_filtered["t"], df_filtered["E"],
                    s=1, alpha=0.5, color='blue',
                    label='Filtered Hits')

        plt.xlabel("Time (s)")
        plt.ylabel("Energy (E)")
        plt.ylim(0, 100000)
        plt.title(f"{file} — Energy vs Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Save instead of show
        plot_filename = file.replace(".parquet", ".png")
        plot_path = os.path.join(PLOT_DIR, plot_filename)

        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # VERY important to avoid memory issues


