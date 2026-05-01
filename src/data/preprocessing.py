import pandas as pd
import numpy as np

def clean_data(df):
    """Limpa strings, remove nulos e trata interrogações.
    
    Comando utilizado para remover espaços em branco extras das colunas de texto"""
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    
    # Substitui '?' por NaN e remove as linhas
    df = df.replace('?', np.nan) # remover casos de valores nulos
    df = df.dropna()
    
    return df

def split_features_target(df, target_col='income'):
    """Separa X (features) e y (alvo)."""
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda x: 1 if '>50K' in x else 0)
    return X, y