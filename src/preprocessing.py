import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Return DataFrame
def load_data(path):
    df = pd.read_csv(path)
    return df

# Preprocessing
def preprocess():
    
    path = '../data/archive/Housing.csv'
    df = load_data(path)
    
    # Collect all categorical columns except 'furnishingstatus'
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'furnishingstatus']
    
    # Ordinal Encoding for binary categorical columns
    oe = OrdinalEncoder(categories=[['no','yes']])
    for col in categorical_cols:
        df[col] = oe.fit_transform(df[[col]])
    
    # Ordinal Encoding for 'furnishingstatus'
    oe = OrdinalEncoder(categories=[['unfurnished', 'semi-furnished', 'furnished']])
    df['furnishingstatus'] = oe.fit_transform(df[['furnishingstatus']])
    
    # Collect numerical columns for normalization
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("Preprocessing Complete")
    
    df.to_csv('processed_housing_data.csv', index=False)

    return df
