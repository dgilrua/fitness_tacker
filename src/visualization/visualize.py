import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/data_resampled.pkl")


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

mpl.rcParams["figure.figsize"] = (20, 7)
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 100

for label in df.label.unique():
    fig, ax = plt.subplots()
    label_df = df[df["label"] == label]
    plt.plot(label_df[:200]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = (
    df.query("label == 'squat' & participant == 'A'")
    .sort_values("category")
    .reset_index(drop=True)
)

fig, ax = plt.subplots()
category_df.groupby("category")["acc_y"].plot(legend=True)
plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = (
    df.query("label == 'bench'").sort_values("participant").reset_index(drop=True)
)

import seaborn as sns

fig, ax = plt.subplots()
participant_df.groupby("participant")["acc_y"].plot(legend=True)
plt.show()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
