import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def get_model_data(csv_path="df_input.csv", scaled=True, random_state=42):
    """
    Loads, feature-engineers, splits, and preprocesses the data.

    Args:
        csv_path (str): Path to the cleaned 'df_input.csv'.
        scaled (bool): If True, applies StandardScaler to features.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # --- 1. Load and Feature Engineer ---
    df = pd.read_csv(csv_path)

    # Create new features
    df["hours_per_game"] = df["hours"] / (df["games"] + 1)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 21, 25, 30, 35, 40, 70],
        labels=["18-21", "22-25", "26-30", "31-35", "36-40", "41+"],
    )

    # Drop original columns we've 'replaced' or don't need
    df_model_ready = df.drop(["account_id", "age"], axis=1)

    # Create dummies for ALL categorical features
    df_model_ready = pd.get_dummies(
        df_model_ready,
        columns=["gender", "genre1", "genre2", "plan", "currency", "age_group", "type"],
    )

    # --- 2. Split Data (before any preprocessing) ---
    X = df_model_ready.drop("is_churn", axis=1)
    y = df_model_ready["is_churn"]

    # Align columns in case some dummies didn't appear in one set (rare but possible)
    X = X.loc[:, ~X.columns.duplicated()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Get column names before scaling (for scaler)
    X_train_cols = X_train.columns
    X_test_cols = X_test.columns

    # --- 3. Scale Data (if required) ---
    if scaled:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train, columns=X_train_cols)
        X_test = pd.DataFrame(X_test, columns=X_test_cols)

    # --- 4. Handle Imbalance (on training data only) ---
    ros = RandomOverSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test
