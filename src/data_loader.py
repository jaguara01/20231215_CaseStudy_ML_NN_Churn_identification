import pandas as pd
import numpy as np


def _engineer_features(df):
    """Helper function to create new features."""
    df_out = df.copy()

    # 1. Create new features
    df_out["hours_per_game"] = df_out["hours"] / (df_out["games"] + 1)
    df_out["age_group"] = pd.cut(
        df_out["age"],
        bins=[17, 21, 25, 30, 35, 40, 70],
        labels=["18-21", "22-25", "26-30", "31-35", "36-40", "41+"],
    )

    # 2. Drop original columns we've 'replaced' or don't need
    df_out = df_out.drop(["age"], axis=1)

    # 3. Create dummies
    df_out = pd.get_dummies(
        df_out,
        columns=["gender", "genre1", "genre2", "plan", "currency", "age_group", "type"],
    )

    return df_out


def _get_user_groups(sales_path="sales.csv"):
    """
    Internal helper to load sales data and identify user groups.
    Returns:
        df_users (pd.DataFrame): All users with seq, last_date, currency, is_churn
        user_reject_list (list): All account_IDs to exclude from training
        user_pred_list (list): All account_IDs for the final prediction
    """
    df_sales_orig = pd.read_csv(sales_path)

    # Re-create user dataframe
    df_sales_seq = (
        df_sales_orig.groupby(["account_id", "plan", "currency"])["order_id"]
        .count()
        .reset_index(name="seq")
    )
    df_sales_last = (
        df_sales_orig.groupby(["account_id"])["start_date"]
        .max()
        .reset_index(name="last_date")
    )
    df_users = pd.merge(df_sales_last, df_sales_seq, on="account_id")
    df_users["is_churn"] = np.where(df_users["seq"] < 7, 1, 0)

    # Find ALL users to exclude from training (uncertain OR seq<=2)
    user_reject_df = df_users[
        (
            (df_users["last_date"] > "2020-11-30")
            & (df_users["is_churn"] == 1)
            & (df_users["currency"] == "USD")
        )
        | (
            (df_users["last_date"] > "2020-09-31")
            & (df_users["is_churn"] == 1)
            & (df_users["currency"] == "EUR")
        )
        | (df_users["seq"] <= 2)
    ]
    user_reject_list = user_reject_df["account_id"].unique()

    # Find the "uncertain" users we need to predict on (seq > 2)
    user_pred_df = user_reject_df[user_reject_df["seq"] > 2]
    user_pred_list = user_pred_df["account_id"].unique()

    return df_users, user_reject_list, user_pred_list


def get_training_data(sales_path="sales.csv", activity_path="user_activity.csv"):
    """
    Loads all original data and returns a clean, feature-engineered
    dataframe for training and testing (all "known" users).
    """

    try:
        df_users, user_reject_list, _ = _get_user_groups(sales_path)
        df_activity_orig = pd.read_csv(activity_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please make sure 'sales.csv' and 'user_activity.csv' are in the root project folder."
        )
        return None

    # Get all users NOT in the reject list
    df_train_full = df_users[~df_users["account_id"].isin(user_reject_list)]
    df_train_full = pd.merge(
        df_train_full, df_activity_orig, on="account_id", how="inner"
    )

    # Drop columns we don't need for modeling
    cols_to_drop = ["last_date", "seq", "account_id"]
    df_train_full = df_train_full.drop(columns=cols_to_drop)

    # Apply Feature Engineering
    df_train_processed = _engineer_features(df_train_full)

    print(f"Data loading complete.")
    print(f"Loaded {len(df_train_processed)} known users for training/testing.")

    return df_train_processed


def get_prediction_data(sales_path="sales.csv", activity_path="user_activity.csv"):
    """
    Loads and processes the data for the "uncertain" users
    who require a final prediction.

    Returns:
        df_predict_processed (pd.DataFrame): The feature-engineered data to predict on.
        df_predict_info (pd.DataFrame): The original user info for the final report.
    """

    try:
        df_users, _, user_pred_list = _get_user_groups(sales_path)
        df_activity_orig = pd.read_csv(activity_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please make sure 'sales.csv' and 'user_activity.csv' are in the root project folder."
        )
        return None, None

    # Get all users in the "predict" list
    df_to_predict = df_users[df_users["account_id"].isin(user_pred_list)]
    df_to_predict = pd.merge(
        df_to_predict, df_activity_orig, on="account_id", how="inner"
    )

    # --- Create the info dataframe for the final report ---
    df_predict_info = df_to_predict[
        ["account_id", "age", "gender", "hours", "games", "plan", "currency"]
    ].copy()

    # --- Prepare the data for the model ---
    cols_to_drop = ["last_date", "seq", "account_id"]
    df_to_predict_model = df_to_predict.drop(columns=cols_to_drop)

    # Apply Feature Engineering
    df_predict_processed = _engineer_features(df_to_predict_model)

    print(f"Loaded {len(df_predict_processed)} uncertain users for prediction.")

    # --- Align Columns ---
    # We must load the training data *columns* to ensure alignment
    print("Aligning prediction columns to training data...")
    df_train_cols = get_training_data(sales_path, activity_path)

    if df_train_cols is None:
        print("Error: Could not load training data for column alignment.")
        return None, None

    train_cols = df_train_cols.columns

    missing_in_predict = set(train_cols) - set(df_predict_processed.columns)
    for c in missing_in_predict:
        if c != "is_churn":  # Don't add the target column
            df_predict_processed[c] = 0

    extra_in_predict = set(df_predict_processed.columns) - set(train_cols)
    for c in extra_in_predict:
        df_predict_processed = df_predict_processed.drop(c, axis=1)

    # Match column order
    df_predict_processed = df_predict_processed[train_cols.drop("is_churn")]

    return df_predict_processed, df_predict_info
