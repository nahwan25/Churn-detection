import pandas as pd
import joblib

# Load artifacts
scaler = joblib.load('artifacts/scaler.pkl')
binary_maps = joblib.load('artifacts/binary_maps.pkl')
train_columns = joblib.load('artifacts/train_columns.pkl')

def preprocess_input(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Binary encoding
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # One-hot encoding
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cat_cols)

    # Lengkapi kolom agar match
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[train_columns]

    # Scaling
    num_cols = ['TotalCharges', 'MonthlyCharges', 'tenure']
    df[num_cols] = scaler.transform(df[num_cols])

    return df
