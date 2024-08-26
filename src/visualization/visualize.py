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

fig, ax = plt.subplots()
participant_df.groupby("participant")["acc_y"].plot(legend=True)
plt.show()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"

all_axis_df = df.query(
    f"label == '{label}' & participant == '{participant}'"
).reset_index(drop=True)
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax, legend=True)


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df.label.unique()
participants = df.participant.unique()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(
            f"label == '{label}' & participant == '{participant}'"
        ).reset_index(drop=True)

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax, legend=True)
            plt.show()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(
            f"label == '{label}' & participant == '{participant}'"
        ).reset_index(drop=True)

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax, legend=True)
            plt.show()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

for label in labels:
    for participant in participants:

        all_axis_df = df.query(
            f"label == '{label}' & participant == '{participant}'"
        ).reset_index(drop=True)

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(2, sharex=True, figsize=(20, 14))
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0], legend=True)
            fig.suptitle(f"{label} - {participant}".title(), fontsize=16)
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1], legend=True)
            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("Samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
