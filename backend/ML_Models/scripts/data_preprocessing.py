import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    #to convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # to select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    
    expected_cols = ['Urban Usage (kWh)', 'Rural Usage (kWh)']
    for col in expected_cols:
        if col not in df.columns:
            print(f" Warning: Column '{col}' not found!")


    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    
    df.to_csv("../data/processed_data.csv", index=False)

    return df

if __name__ == "__main__":
    file_path = "../data/sectorwise_energy_updated.csv"
    load_and_preprocess_data(file_path)
