##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 12-01-2023

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Reshape,
    Dropout,
    MaxPooling1D,
    Conv1D,
    Flatten,
    GlobalMaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import copy


class ForwardSelection:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ForwardSelection()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            print(i)

            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ""

            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                (
                    pred_y_train,
                    pred_y_test,
                    prob_training_y,
                    prob_test_y,
                ) = ca.decision_tree(
                    X_train[temp_selected_features],
                    y_train,
                    X_train[temp_selected_features],
                )
                perf = accuracy_score(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f
            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)
        return selected_features, ordered_features, ordered_scores

    def decision_tree(
        self,
        train_X,
        train_y,
        test_X,
        min_samples_leaf=50,
        criterion="gini",
        print_model_details=False,
        export_tree_path="Example_graphs/Chapter7/",
        export_tree_name="tree.dot",
        gridsearch=True,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "criterion": ["gini", "entropy"],
                }
            ]
            dtree = GridSearchCV(
                DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, criterion=criterion
            )

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=dtree.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(dtree.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            print("Feature importance decision tree:")
            for i in range(0, len(dtree.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(dtree.feature_importances_[ordered_indices[i]])
            tree.export_graphviz(
                dtree,
                out_file=export_tree_path + export_tree_name,
                feature_names=train_X.columns,
                class_names=dtree.classes_,
            )

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y


class ClassificationAlgorithms:
    def LSTM_nn(self, X_train, y_train, X_test):

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_train = to_categorical(y_train_encoded)
        n_classes = len(label_encoder.classes_)
        n_features = len(X_train.columns)

        model = Sequential(
            [
                Input(shape=(n_features,)),
                Reshape((1, n_features)),
                LSTM(128, return_sequences=True),
                Dropout(0.5),
                LSTM(64),
                Dense(32, activation="relu"),
                Dense(n_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train, y_train, validation_split=0.2, epochs=30, batch_size=32
        )

        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        predictions = label_encoder.inverse_transform(predictions)

        return predictions

    def CNN(self, X_train, y_train, X_test):

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_train = to_categorical(y_train_encoded)
        n_classes = len(label_encoder.classes_)
        n_features = len(X_train.columns)

        X_train_reshaped = X_train.values.reshape(
            (X_train.shape[0], X_train.shape[1], 1)
        )
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential(
            [
                Input(shape=(n_features, 1)),
                Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
                Conv1D(filters=32, kernel_size=5, activation="relu", padding="same"),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(n_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train_reshaped, y_train, validation_split=0.2, epochs=30, batch_size=32
        )

        predictions = model.predict(X_test_reshaped)
        predictions = np.argmax(predictions, axis=1)
        predictions = label_encoder.inverse_transform(predictions)

        return predictions

    def feedforward_neural_network(
        self,
        train_X,
        train_y,
        test_X,
        hidden_layer_sizes=(100,),
        max_iter=2000,
        activation="logistic",
        alpha=0.0001,
        learning_rate="adaptive",
        gridsearch=False,
        print_model_details=False,
    ):

        if gridsearch:
            tuned_parameters = [
                {
                    "hidden_layer_sizes": [
                        (5,),
                        (10,),
                        (25,),
                        (100,),
                        (
                            100,
                            5,
                        ),
                        (
                            100,
                            10,
                        ),
                    ],
                    "activation": [activation],
                    "learning_rate": [learning_rate],
                    "max_iter": [1000, 2000],
                    "alpha": [alpha],
                }
            ]
            nn = GridSearchCV(
                MLPClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            nn = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                max_iter=max_iter,
                learning_rate=learning_rate,
                alpha=alpha,
            )

        nn.fit(
            train_X,
            train_y.values.ravel(),
        )

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return pred_test_y, nn
