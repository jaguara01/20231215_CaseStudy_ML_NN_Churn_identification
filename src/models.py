import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Neural Network Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


def get_report_dict(y_test, y_pred, model_name):
    """Helper function to format the classification report."""
    # Added zero_division=0 to prevent errors if a class is not predicted
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Use .get() for safety, in case class "1" is not in the report
    class_1_report = report.get("1", {})

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Churn_Precision": class_1_report.get("precision", 0),
        "Churn_Recall": class_1_report.get("recall", 0),
        "Churn_F1-Score": class_1_report.get("f1-score", 0),
    }


# --- THIS FUNCTION IS UNCHANGED ---
# It is the ONLY one that needs pre-oversampled data
def train_neural_network(X_train, y_train, X_test, y_test):
    """
    Trains the Keras Neural Network.
    EXPECTS PRE-OVERSAMPLED X_train, y_train.
    """
    n_features = X_train.shape[1]
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stopper = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[early_stopper],
        verbose=0,
    )
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    return get_report_dict(y_test, y_pred, "Neural Network")


# --- ALL OTHER FUNCTIONS ARE UPDATED ---
# They now expect ORIGINAL IMBALANCED data and handle imbalance internally.


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Tunes and trains a Random Forest model.
    EXPECTS IMBALANCED X_train, y_train.
    """
    print("  Tuning Random Forest...")
    param_grid = {
        "n_estimators": randint(100, 400),
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],  # RE-ADDED
    }

    rf = RandomForestClassifier(random_state=42)

    rs_model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring="recall",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    # Fit on the original (imbalanced) X_train, y_train
    rs_model.fit(X_train, y_train)
    best_rf_model = rs_model.best_estimator_

    y_pred = best_rf_model.predict(X_test)
    return get_report_dict(y_test, y_pred, "Random Forest")


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Tunes and trains a Logistic Regression model.
    EXPECTS IMBALANCED X_train, y_train.
    """
    print("  Tuning Logistic Regression...")
    param_grid = {"C": loguniform(1e-3, 1e2), "solver": ["liblinear"]}

    # RE-ADDED class_weight='balanced'
    log_reg = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    )

    rs_model = RandomizedSearchCV(
        estimator=log_reg,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring="recall",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    # Fit on the original (imbalanced) X_train, y_train
    rs_model.fit(X_train, y_train)
    best_log_reg = rs_model.best_estimator_

    y_pred = best_log_reg.predict(X_test)
    return get_report_dict(y_test, y_pred, "Logistic Regression")


def train_svm(X_train, y_train, X_test, y_test):
    """
    Trains a Support Vector Machine.
    REMOVED tuning to fix convergence issues.
    EXPECTS IMBALANCED X_train, y_train.
    """
    print("  Training SVM (default params)...")

    # We will use a single, default RBF kernel SVM
    # This bypasses the search that was failing.
    svm_model = SVC(class_weight="balanced", random_state=42)

    # Fit on the original (imbalanced) X_train, y_train
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    return get_report_dict(y_test, y_pred, "SVM")


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Tunes and trains an XGBoost model.
    EXPECTS IMBALANCED X_train, y_train.
    """
    print("  Tuning XGBoost...")
    # RE-ADDED scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_grid = {
        "n_estimators": randint(100, 400),
        "max_depth": [3, 5, 7, 9],
        "learning_rate": loguniform(1e-2, 5e-1),
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # RE-ADDED
        random_state=42,
        eval_metric="logloss",
    )

    rs_model = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring="recall",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    # Fit on the original (imbalanced) X_train, y_train
    rs_model.fit(X_train, y_train)
    best_xgb_model = rs_model.best_estimator_

    y_pred = best_xgb_model.predict(X_test)
    return get_report_dict(y_test, y_pred, "XGBoost")
