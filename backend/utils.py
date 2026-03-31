import pandas as pd

def preprocess_uploaded_file(file_path):
    df = pd.read_csv(file_path)
    return df
