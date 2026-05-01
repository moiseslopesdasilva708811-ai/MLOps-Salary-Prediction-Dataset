import pandas as pd
import pytest
from src.data.cleaning_data import limpar_dados 

def test_limpeza_valores_ausentes():
    # Criamos um mini-dataframe com o erro que seu relatório cita ("?")
    df_sujo = pd.DataFrame({'age': [25, "?"], 'workclass': ["Private", None]})
    df_limpo = limpar_dados(df_sujo)
    
    # Valida que não existem mais "?" nem valores nulos
    assert "?" not in df_limpo.values
    assert df_limpo.isnull().sum().sum() == 0

def test_colunas_obrigatorias():
    # Garante que as 14 variáveis selecionadas no seu relatório estão presentes
    df_exemplo = pd.read_csv("data/raw/adult.csv").head(1)
    df_limpo = limpar_dados(df_exemplo)
    assert df_limpo.shape[1] >= 14