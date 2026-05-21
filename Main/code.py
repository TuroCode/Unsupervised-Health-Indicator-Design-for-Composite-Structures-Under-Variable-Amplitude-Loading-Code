import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(BASE_DIR, "data_2224", "spec_01_45_old.csv")

df = pd.read_csv(input_file, sep="\t", dtype=str)

# Fix header spaces
df.columns = df.columns.str.strip()

print(df.columns.tolist())

print('hellow world')