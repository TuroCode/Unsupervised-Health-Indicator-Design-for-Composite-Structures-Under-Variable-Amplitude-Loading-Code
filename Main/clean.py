import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#NEW EDITION TO CLEAN ALL EXCEPT THE _NEW ONES IN ONE RUN
DATA_DIR = os.path.join(BASE_DIR, "data_2224")
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
#MAKING SURE OUTPUT EXISTS ON UR LAPTOP
os.makedirs(CLEAN_DIR, exist_ok=True)

#UNCOMMENT TO GENERATE CSV
# output_file = os.path.join(BASE_DIR, "clean_data", "clean_01_45_old.csv")
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# parquet_file = os.path.join(BASE_DIR, "clean_data", "parquet_01_45_old.parquet")
# os.makedirs(os.path.dirname(parquet_file), exist_ok=True)

files_to_clean = [
    f for f in os.listdir(DATA_DIR)
    #remove after and not to clean new files as well
    if f.endswith(".csv") 
]
for file_name in files_to_clean:
    #SO U KNOW ITS FUCKING WORKING N NOT DYING
    print(f"CLEANING: {file_name}")
    #input stuff
    input_file = os.path.join(DATA_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]
    #output stuff
    parquet_file = os.path.join(CLEAN_DIR, f"{base_name}_clean.parquet")
    #UNCOMMENT TO MAKE CSVS
    # csv_file = os.path.join(CLEAN_DIR, f"{base_name}_clean.csv")
    #MAKING SURE IT EXISTS ON UR LAPTOP
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
# -----------------------------
# READ FILE
# -----------------------------
    df = pd.read_csv(
        input_file,
        sep="\t",
        skiprows=[1,2,3,4,5],   # remove rows 2-5 IF 5 IS INCLUDED THEN WE R REMOVING FIRST row of data as well
        low_memory=False
    )

    # Clean column spacing
    df.columns = df.columns.str.strip()

    # Clean whitespace in data
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # -----------------------------
    # TIME LINEARISATION
    # -----------------------------
    time = pd.to_datetime(df["HHMMSS"], format="%H:%M:%S")

    fraction = (
        df["MSEC"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float) / 1e6
    )

    time_seconds = (
        time.dt.hour * 3600 +
        time.dt.minute * 60 +
        time.dt.second +
        fraction
    )

    t_cumulative = time_seconds.diff().fillna(0)
    t_cumulative = t_cumulative.where(t_cumulative >= 0, t_cumulative + 86400)

    t_final = t_cumulative.cumsum()
    t_final = t_final - t_final.iloc[0]

    # -----------------------------
    # KEEP DATA FROM CHAN ONWARD
    # -----------------------------
    data_df = df.loc[:, "CHAN":].copy()
    #DROPPING THE ONES BENJAMIN SAID TO.
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
    cols_to_drop = ["CHAN", "CENY", "ALIN", "PA0", "PA1"] 
    data_df = data_df.drop(columns=cols_to_drop, errors="coerce")
    # Fix decimal commas
    for col in data_df.columns:
        data_df[col] = (
            data_df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

    #REMOVE ANY ROW WITH A NAN IN IT
    data_df = data_df.dropna()
    # Insert time column
    data_df.insert(0, "t", t_final)

    # -----------------------------
    # SAVE
    # -----------------------------
    #UNCOMMENT TO GENERATE TSV/CSV FILE
    # data_df.to_csv(output_file, sep="\t", index=False)
    # print("Done: Clean AE dataset TSV file created.")
    # #SIZING TEST for TSV
    # print("CSV size:", os.path.getsize(output_file)/1e6, "MB")
    data_df.to_parquet(parquet_file, compression="snappy", index = False)
    #CAN ADJUST TO NOT SEE FILE SIZE ALL THE TIME
    print(f"Done: {file_name} -> {parquet_file} ({os.path.getsize(parquet_file)/1e6:.2f} MB.)")
    #SIZING TEST for Parquet
    # print("Parquet size:", os.path.getsize(parquet_file)/1e6, "MB")

#TO KNOW U DONE:
print("All files are spick n span")


#PARQUET TEST DONT WANNA DO FOR EVERY FILE SO DO NOT UNCOMMENT ITS RISKY 
# test = pd.read_parquet(parquet_file)
# print(test.head())
