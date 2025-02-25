import os
import pandas as pd
import numpy as np

def load_data(file_path):
    is_possible = os.path.exists(file_path)
    df = pd.read_csv(file_path)
    is_possible = is_possible or df.empty
    return df, is_possible

def preprocess_data(df, drop_columns=None, fillna_method=None):
    df = df.copy()
    df = drop_unnecessary_columns(df, drop_columns)
    df = fill_missing_values(df, fillna_method)
    df = standardize_numeric_columns(df)
    return df

def drop_unnecessary_columns(df, drop_columns):
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
    return df

def fill_missing_values(df, fillna_method):
    if fillna_method:
        if fillna_method == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif fillna_method == 'median':
            df.fillna(df.median(), inplace=True)
        elif fillna_method == 'mode':
            df.fillna(df.mode().iloc[0], inplace=True)
        else:
            raise ValueError(f"Método de relleno no reconocido: {fillna_method}")
    return df

def standardize_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if not numeric_columns.any():
        raise ValueError("No se encontraron columnas numéricas para estandarizar.")
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    return df
