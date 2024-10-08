import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from LearningAlgorithms import ClassificationAlgorithms, ForwardSelection
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/data_clustered.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set", "duration"], axis=1)

X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="skyblue", label="Total")
y_train.value_counts().plot(kind="bar", ax=ax, color="salmon", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="lightgreen", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
pca_features = ["pca_1", "pca_2", "pca_3"]
square_features = ["acc_r", "gyr_r"]
time_features = [i for i in df_train.columns if "_temp_" in i]
freq_features = [i for i in df_train.columns if ("_freq" in i) or ("_pse" in i)]
cluster_features = ["cluster"]

features_1 = basic_features
features_2 = list(set(basic_features + pca_features + square_features))
features_3 = list(set(features_2 + time_features))
features_4 = list(set(features_3 + freq_features + cluster_features))

selected_features = [
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_x_freq_0.0_Hz_ws_14",
    "gyr_r_freq_0.0_Hz_ws_14",
    "acc_z",
    "acc_y_freq_0.0_Hz_ws_14",
    "gyr_r_freq_1.429_Hz_ws_14",
    "gyr_z_max_freq",
    "acc_z_freq_2.143_Hz_ws_14",
    "acc_z_freq_2.5_Hz_ws_14",
    "pca_1",
]

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()
selection = ForwardSelection()

max_features = 10
"""selected_features, ordered_features, ordered_scores = selection.forward_selection(
    max_features, X_train, y_train
)"""

ordered_scores = [
    0.885556704584626,
    0.9886246122026887,
    0.9955187866253016,
    0.9975870389520854,
    0.9982764563943468,
    0.9993105825577387,
    0.9993105825577387,
    0.9993105825577387,
    0.9993105825577387,
    0.9993105825577387,
]

fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(range(1, max_features + 1), ordered_scores, marker="o")
plt.xticks(range(1, max_features + 1))
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.title("Forward feature selection")
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    features_1,
    features_2,
    features_3,
    features_4,
    selected_features,
]

feature_names = [
    "Features set 1",
    "Features set 2",
    "Features set 3",
    "Features set 4",
    "Selected features",
]

iterations = 1

score_df = pd.read_pickle("../../data/interim/accuracy_df.pkl")
"""
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    performance_test_lstm = 0

    print("\tTraining LSTM,")

    class_test_y_lstm = learner.LSTM_nn(selected_train_X, y_train, selected_test_X)
    performance_test_lstm = accuracy_score(y_test, class_test_y_lstm)

    print("\tTraining NN,")

    class_test_y_nn = learner.NN(selected_train_X, y_train, selected_test_X)
    performance_test_nn = accuracy_score(y_test, class_test_y_nn)

    print("\tTraining CNN,")

    class_test_y_cnn = learner.CNN(selected_train_X, y_train, selected_test_X)
    performance_test_cnn = accuracy_score(y_test, class_test_y_cnn)

    models = ["LSTM", "NN", "CNN"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_lstm,
                performance_test_nn,
                performance_test_cnn,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

score_df.sort_values("accuracy", ascending=False, inplace=True)
score_df.to_pickle("../../data/interim/accuracy_df.pkl")
"""
# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

fig, ax = plt.subplots(figsize=(20, 7))
sns.barplot(data=score_df, x="feature_set", y="accuracy", hue="model", ax=ax)

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

class_test_y_lstm = learner.LSTM_nn(X_train[features_4], y_train, X_test[features_4])
performance_test_lstm = accuracy_score(y_test, class_test_y_lstm)
classes = y_test.unique()
cm = confusion_matrix(y_test, class_test_y_lstm, labels=classes)

plt.figure(figsize=(13, 13))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    square=True,
)
plt.title("Confusion matrix", pad=20, fontsize=30)
plt.ylabel("True label", labelpad=10)
plt.xlabel("Predicted label", labelpad=10)
plt.show()

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["category", "set", "duration"], axis=1)
X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop("participant", axis=1)
X_test = X_test.drop("participant", axis=1)

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

class_test_y_nn_participant, model = learner.feedforward_neural_network(
    X_train[features_4], y_train, X_test[features_4]
)
performance_test_nn_participant = accuracy_score(y_test, class_test_y_nn_participant)

cm = confusion_matrix(y_test, class_test_y_nn_participant, labels=classes)

plt.figure(figsize=(13, 13))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    square=True,
)
plt.title("Confusion matrix", pad=20, fontsize=30)
plt.ylabel("True label", labelpad=10)
plt.xlabel("Predicted label", labelpad=10)
plt.show()

joblib.dump(model, "../../models/model.pkl")
