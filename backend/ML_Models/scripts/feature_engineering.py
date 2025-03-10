import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(file_path):
    # Load the processed data
    df = pd.read_csv(file_path)
    
    # Convert 'Date' to datetime; infer format to handle inconsistencies and coerce errors
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, dayfirst=True, errors='coerce')
    df.sort_values('Date', inplace=True)  # Ensure data is in chronological order
    
    # Encode categorical columns (exclude 'Date')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Date' in categorical_cols:
        categorical_cols.remove('Date')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create the target column: Total Energy = Urban Usage + Rural Usage
    df['Total Energy (kWh)'] = df['Urban Usage (kWh)'] + df['Rural Usage (kWh)']
    
    # Create lag features for Total Energy usage (e.g., lag1, lag2, lag3)
    df['Lag1_TotalEnergy'] = df['Total Energy (kWh)'].shift(1)
    df['Lag2_TotalEnergy'] = df['Total Energy (kWh)'].shift(2)
    df['Lag3_TotalEnergy'] = df['Total Energy (kWh)'].shift(3)
    
    # Drop rows with NaN values in lag features (first 3 rows will be NaN)
    df = df.dropna(subset=['Lag1_TotalEnergy', 'Lag2_TotalEnergy', 'Lag3_TotalEnergy'])
    
    # Separate features and target
    # Features: All columns except the ones directly providing the target information
    features_df = df.drop(columns=['Urban Usage (kWh)', 'Rural Usage (kWh)', 'Total Energy (kWh)'])
    # Target: The actual total energy usage
    target_df = df[['Total Energy (kWh)']]
    
    # Save the engineered features and target to separate CSV files
    features_df.to_csv("../data/features.csv", index=False)
    target_df.to_csv("../data/target.csv", index=False)
    
    print("Feature engineering completed with lag features! Features and target files saved.")
    return features_df, target_df

if __name__ == "__main__":
    file_path = "../data/processed_data.csv"
    feature_engineering(file_path)
