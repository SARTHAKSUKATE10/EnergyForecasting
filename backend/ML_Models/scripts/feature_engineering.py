import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(file_path):
    df = pd.read_csv(file_path)

    # Identify categorical columns and encode them
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders if needed later

    # Create the target column but DO NOT include it in features
    df['Total Energy (kWh)'] = df['Urban Usage (kWh)'] + df['Rural Usage (kWh)']
    df.drop(columns=['Total Energy (kWh)'], inplace=True)  # Remove to prevent leakage

    # Save feature-engineered dataset
    df.to_csv("../data/features.csv", index=False)

    print("Feature engineering completed! Data saved.")

    return df

if __name__ == "__main__":
    file_path = "../data/processed_data.csv"
    feature_engineering(file_path)
