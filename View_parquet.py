#USE WHICHEVER ONE U NEED 

#PACKAGES
import pyarrow.parquet as pq
import os
import duckdb
import pandas as pd

#SELECT THE FILE TO BE READ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_2224")
file = os.path.join(BASE_DIR, "clean_data", "spec_01_45_old_clean.parquet") #CHANGE HERE MAKE SURE TO ONLY USE PARQUET
csv = os.path.join(BASE_DIR, "data_2224", "spec_01_45_old.csv")
df = pd.read_csv(csv, sep='\t')
print(df.columns)
print(len(df.columns))
#1) to inspect the schema only comment if not needed
schema = pq.read_schema(file)
print(schema) #Comment out if not needed

#2) READ FIRST N ROWS 
# N = 5 #ONLY CHANGE THIS
# table = pq.read_table(file)
# sample = table.slice(0,N)
# print(sample) #Comment out if not needed


#3) raed single row group
#pf = pq.ParquetFile(file)

#print("Row groups:", pf.num_row_groups)
#df = pf.read_row_group(0).to_pandas()
#print(df.head())#comment out here

#4) Load into pandas THIS PROBS WONT WORK
#df = pd.read_parquet(file)
#print(df.head()) #comment this out

#5) duckdb this should work idk what it is exactly though
# result = duckdb.sql(f"SELECT * FROM '{file}' LIMIT 20")
# print(result) #If you see (double) it means its a 64-bit float which is the case for stuff like time and E.

# #6 Query specic colum using the duckdb
# N = 20 #Adjust this for how many rows u wanna read
# result = duckdb.sql(f"SELECT t, E FROM '{file}' LIMIT {N}") #CHANGE THE COLUMNS NAMES DEPENDING ON WHAT U WANT here's all the names: 
# t
# CHAN
# A
# R
# THR
# E
# PA0
# NoTRAI
# PA1
# TRAI
# CSS
# CENY
# CCNT
# RMS
# PCTA
# PCTD
# D
# CHIT
# SS
# CNTS
# ALIN
# print(result) #If you see (double) it means its a 64-bit float which is the case for stuff like time and E.


#WORK IN PROGRESS:
#THIS STUFF SHOULD HOPEFULLY MAKE LIFE EASIER ONCE I FIGURE IT OUT
# result = duckdb.sql(f"""
# SELECT
#     COUNT(*) AS rows,
#     AVG(E) AS avg_E,
#     MAX(E) AS max_E,
#     MIN(E) AS min_E,
#     STDDEV_SAMP(E) AS stddev_E,
#     AVG(t) AS avg_t,
#     MAX(t) AS max_t,
#     MIN(t) AS min_t,
#     STDDEV_SAMP(t) AS stddev_t
# FROM '{file}'
# """)

# print(result)

# df = duckdb.sql(f"""
# SELECT t, E
# FROM '{file}'
# WHERE E > 10000
# """).df()

# print(df.head())

# duckdb.sql(f"""
# SELECT t, E, A
# FROM '{file}'
# WHERE E > 20000
# ORDER BY E DESC
# LIMIT 20
# """).show()

# duckdb.sql(f"""
# SELECT CHAN, COUNT(*) AS hits
# FROM '{file}'
# GROUP BY CHAN
# """).show()