import pandas as pd

def selecionar_features(df):
    """
    Seleciona as colunas conforme análise de Spearman e Mutual Information.
    """
    # Lista de colunas baseada no seu relatório técnico
    colunas_vencedoras = [
        'age', 'workclass', 'education_num', 'marital_status', 
        'occupation', 'relationship', 'race', 'sex', 
        'capital_gain', 'capital_loss', 'hours_per_week'
    ]
    return df[colunas_vencedoras]