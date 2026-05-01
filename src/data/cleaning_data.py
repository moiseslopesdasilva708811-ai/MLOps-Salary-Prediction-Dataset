import pandas as pd

def limpar_dados(df):
    """
    Comando usado para executar o Pipeline ETL descrito no Relatório.
    """
    df_clean = df.copy()
    
    # Mapeando Caracteres Especiais "?"
    df_clean = df_clean.replace("?", None)
    
    # Imputando Moda em Variáveis Categóricas
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
    # Tratamento de Valores Nulos Numéricos
    df_clean = df_clean.dropna()
    
    return df_clean