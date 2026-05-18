import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


### Load dataset
def load_dataset(path):
    return pd.read_csv(path)


### Preview dataset
def preview_dataset(df, n=5):
    return df.head(n)


### Dataset information
def dataset_info(df):
    print(df.info())


### Descriptive statistics
def descriptive_statistics(df):
    return df.describe()


### Check missing values
def check_missing_values(df):
    return df.isnull().sum()


### Visualize distribution
def visualize_distribution(df):
    df.hist(figsize=(10, 8))
    plt.show()


### Remove missing values
def remove_missing_values(df):
    return df.dropna()


### Remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()


### Encode categorical data
def encode_categorical(df):
    return pd.get_dummies(df, drop_first=True)


### Normalize data
def normalize_data(df):
    scaler = MinMaxScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    return df_scaled


### Save dataset
def save_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Dataset berhasil disimpan di: {path}")


### Full preprocessing pipeline
def preprocessing_pipeline(df):

    ### Remove missing values
    df = remove_missing_values(df)

    ### Remove duplicates
    df = remove_duplicates(df)

    ### Encoding categorical data
    df = encode_categorical(df)

    ### Normalize data
    df = normalize_data(df)

    return df
