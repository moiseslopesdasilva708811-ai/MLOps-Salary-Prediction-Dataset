import torch
import pytest
from src.models.model import RedeneuralMLP

def test_arquitetura_camadas():
    input_dim = 14
    model = RedeneuralMLP(input_dim)
    
    # Testa se o modelo aceita um input e retorna um output (Forward pass)
    input_dummy = torch.randn(1, input_dim)
    output = model(input_dummy)
    
    # O output deve ser um único valor (probabilidade) entre 0 e 1
    assert output.shape == (1, 1)
    assert not torch.isnan(output).any()

def test_dropout_ativo():
    model = RedeneuralMLP(14)
    model.train() # Ativa o modo de treino (com Dropout)
    
    input_dummy = torch.randn(1, 14)
    out1 = model(input_dummy)
    out2 = model(input_dummy)
    
    # No modo treino com Dropout, as saídas devem ser levemente diferentes
    assert not torch.equal(out1, out2)