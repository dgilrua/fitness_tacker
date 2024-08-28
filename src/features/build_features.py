import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/outliers_removed_chauvenet.pkl")

predictor_cols = df.columns[:6].to_list()

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 100
plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_cols:
    df[col] = df[col].interpolate()

df.isna().sum()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]

    duration = end - start
    df.loc[df["set"] == s, "duration"] = duration.seconds

duration_df = df.groupby("category")["duration"].mean()
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 100
cutoff = 1.4

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_x", fs, cutoff)

subset = df_lowpass[df_lowpass["set"] == 30]
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(subset["acc_x"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_x_lowpass"].reset_index(drop=True), label="lowpass")
ax[1].set_xlabel("samples")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), shadow=True, fancybox=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), shadow=True, fancybox=True)
fig.suptitle("Lowpass filter")


# --------------------------------------------------------------
# Loop over all columns and apply lowpass filter
# --------------------------------------------------------------

for col in predictor_cols:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass.drop(col + "_lowpass", axis=1, inplace=True)

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

pca_df = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(pca_df, predictor_cols)

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(range(1, len(predictor_cols) + 1), pca_values, marker="o")
plt.xlabel("Principal component")
plt.ylabel("Explained variance")
plt.title("PCA explained variance")
plt.show()

pca_df = PCA.apply_pca(pca_df, predictor_cols, 3)

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = pca_df.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]

subset[["acc_r", "gyr_r"]].plot(subplots=True, figsize=(20, 10))

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temp = df_squared.copy()

ws = int(1000 / 100)
NumAbs = NumericalAbstraction()

predictor_cols = predictor_cols + ["acc_r", "gyr_r"]

df_temporal_list = []

for s in df_temp["set"].unique():
    subset = df_temp[df_temp["set"] == s].copy()
    for col in predictor_cols:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temp = pd.concat(df_temporal_list)

fig, ax = plt.subplots(figsize=(20, 10))
df_temp[df_temp["set"] == 14][
    ["acc_r", "acc_r_temp_mean_ws_10", "acc_r_temp_std_ws_10"]
].plot(ax=ax)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 0.5), shadow=True, fancybox=True, ncol=3
)

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temp.copy().reset_index()
FourTrans = FourierTransformation()

fs = int(1000 / 100)
ws = int(2800 / 100)

df_freq_list = []

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
