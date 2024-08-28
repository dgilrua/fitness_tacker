import pandas as pd
import re
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

f = files[1]

participant = f.split("-")[0][-1]
label = f.split("-")[1]
category = re.match("^(\w+?)(?=[\d_])", f.split("-")[2]).group(1)

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:

    participant = f.split("-")[0][-1]
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


def from_files_to_dataset(files):

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:

        participant = f.split("-")[0][-1]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
    gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)

    return acc_df, gyr_df


df_acc, df_gyr = from_files_to_dataset(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

df_merged = pd.concat([df_acc.iloc[:, :3], df_gyr], axis=1)

df_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

days = [g for n, g in df_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample("200ms").apply(sampling).dropna() for df in days]
)

data_resampled["set"] = data_resampled["set"].astype(int)


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/data_resampled.pkl")
