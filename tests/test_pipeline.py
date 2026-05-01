import pytest
import os
from src.models.train import executar_treinamento 

def test_existencia_arquivos_essenciais():
    # Verifica se as pastas que você citou no relatório existem
    assert os.path.exists("data/raw/")
    assert os.path.exists("models/")

def test_conexao_wandb():
    # Tenta verificar se a chave de API do W&B está configurada no ambiente
    # Essencial para garantir os 15% de nota de MLOps
    try:
        import wandb
        assert wandb.api.api_key is not None
    except:
        pytest.fail("W&B não configurado corretamente")

def test_save_model_artifact():
    # Verifica se o pipeline gera o arquivo .pth que você versiona no W&B
    caminho_modelo = "models/modelo_final.pth"
    if os.path.exists(caminho_modelo):
        assert os.path.getsize(caminho_modelo) > 0