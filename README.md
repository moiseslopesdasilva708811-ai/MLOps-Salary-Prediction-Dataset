# 💰 Pipeline MLOps: Adult Census Income Prediction

Este repositório apresenta uma solução completa de **Machine Learning com práticas de MLOps** para classificação de renda utilizando o dataset **Adult Census Income**.

O objetivo do modelo é prever se a renda anual de um indivíduo excede **US$ 50.000**, utilizando uma arquitetura baseada em **Redes Neurais Multilayer Perceptron (MLP)**.

Além do treinamento do modelo, o projeto também enfatiza:

- Engenharia de Dados
- Limpeza Estatística
- Seleção Inteligente de Atributos
- Rastreabilidade de Experimentos
- Versionamento de Modelos
- Monitoramento em Nuvem com Weights & Biases

---

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)

<br>

### 🚀 Dashboard de Experimentos (W&B)

[![W&B Dashboard](https://img.shields.io/badge/W&B-Acesse_o_Dashboard-orange?style=for-the-badge&logo=WeightsAndBiases)](https://wandb.ai/moiseslopesdasilva708811-ufrn/MLOps_Salary_Prediction)

*Visualize curvas de loss, accuracy, importance de atributos e métricas em tempo real.*

</div>

---

# 📌 Objetivo do Projeto

Este projeto foi desenvolvido com o propósito de aplicar conceitos modernos de:

- Ciência de Dados
- Machine Learning
- Deep Learning
- Engenharia de Machine Learning (MLOps)

A proposta é construir um pipeline reproduzível, escalável e monitorável, aproximando o fluxo de desenvolvimento de um ambiente de produção real.

---

# 🧠 Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Linguagem principal |
| Pandas | Manipulação de dados |
| NumPy | Operações numéricas |
| Scikit-Learn | Pré-processamento e métricas |
| PyTorch | Construção da rede neural |
| Weights & Biases | Rastreamento de experimentos |
| Matplotlib | Visualização gráfica |
| Jupyter Notebook | Desenvolvimento e prototipagem |

---

# 🏗️ Arquitetura do Pipeline

```text
Coleta de Dados
        ↓
Limpeza e Pré-processamento
        ↓
Análise Estatística
        ↓
Feature Selection
        ↓
Normalização
        ↓
Treinamento da Rede Neural
        ↓
Avaliação
        ↓
Monitoramento com W&B
        ↓
Versionamento do Modelo
```

---

# 📂 Estrutura do Projeto

```text
MLOps-Salary-Prediction-Dataset/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── Multilayer_Perceptron_Salary_Prediction_Classification.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   └── train.py
│   │
│   └── utils/
│       └── helpers.py
│
├── artifacts/
│   └── mlp_salary_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔬 Etapas do Pipeline

## 📥 1. Coleta e Carregamento dos Dados

O projeto utiliza o dataset Adult Census Income contendo informações como:

- Idade
- Escolaridade
- Estado civil
- Ocupação
- Relação familiar
- Horas trabalhadas
- Nacionalidade
- Capital Gain/Loss

---

# 🧹 2. Limpeza e Tratamento dos Dados

Foram aplicadas diversas técnicas de pré-processamento:

## ✔ Remoção de Duplicatas

Evita viés estatístico causado por amostras repetidas.

## ✔ Tratamento de Valores Nulos

Utilização de:

- Mediana para variáveis numéricas
- Estratégias robustas para dados categóricos

## ✔ Tratamento de Outliers

Aplicação do método IQR (Interquartile Range).

---

# 📊 3. Feature Selection

O projeto aplica múltiplas estratégias de seleção de atributos:

- Variance Inflation Factor (VIF)
- Correlação de Spearman
- Mutual Information

Objetivos:

- Redução do custo computacional
- Minimização de ruído
- Redução de multicolinearidade
- Melhoria da generalização do modelo

---

# 🧠 4. Arquitetura da Rede Neural

A rede neural foi construída utilizando PyTorch.

## Estrutura da MLP

```text
Entrada
   ↓
Linear (64)
   ↓
ReLU
   ↓
Linear (32)
   ↓
ReLU
   ↓
Linear (1)
   ↓
Sigmoid
```

## Características

- Arquitetura em funil
- Early Stopping
- Regularização implícita
- Pipeline rastreável
- Monitoramento de métricas em tempo real

---

# ⚙️ 5. Treinamento

Durante o treinamento são registrados:

- Loss
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Hiperparâmetros
- Tempo de execução

Tudo integrado automaticamente ao Weights & Biases.

---

# 📈 Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# ☁️ Integração com Weights & Biases

O projeto utiliza o W&B para:

- Rastreamento de experimentos
- Comparação entre execuções
- Monitoramento em tempo real
- Armazenamento de métricas
- Versionamento de modelos
- Registro de hiperparâmetros

---

# 🚀 Guia de Execução

## 1️⃣ Clonar o Repositório

```bash
git clone https://github.com/moiseslopesdasilva708811-ai/MLOps-Salary-Prediction-Dataset.git

cd MLOps-Salary-Prediction-Dataset
```

---

# 🐍 2️⃣ Criar Ambiente Virtual

## Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

## Linux/Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

# 📦 3️⃣ Instalar Dependências

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

# 🔐 4️⃣ Autenticar no W&B

```bash
wandb login
```

Cole sua API Key quando solicitado.

---

# ▶️ 5️⃣ Executar o Projeto

Abra o notebook:

```text
notebooks/
```

e execute:

```text
Multilayer_Perceptron_Salary_Prediction_Classification.ipynb
```

O pipeline irá automaticamente:

1. Baixar/carregar os dados
2. Realizar limpeza estatística
3. Codificar atributos categóricos
4. Aplicar Feature Selection
5. Normalizar os dados
6. Treinar a MLP
7. Avaliar o modelo
8. Registrar métricas no W&B
9. Salvar o modelo treinado

---

# 📊 Resultados

Ao final do treinamento são gerados:

- Curvas de Loss
- Curvas de Accuracy
- Confusion Matrix
- Classification Report
- Comparação entre runs
- Importância dos atributos

Todos os resultados ficam disponíveis no dashboard do W&B.

---

# 🎯 Possíveis Melhorias Futuras

- Deploy com FastAPI
- Dockerização do pipeline
- Integração CI/CD
- MLflow
- Kubernetes
- DVC
- Hyperparameter Tuning
- API REST para inferência

---

# 🎓 Autor

## Moisés Lopes da Silva

Desenvolvedor focado em:

- Machine Learning
- Deep Learning
- MLOps
- Engenharia de Software
- Inteligência Artificial
- Engenharia Elétrica

---

# 📜 Licença

Este projeto foi desenvolvido para fins acadêmicos, científicos e educacionais.

---

<div align="center">

## ⭐ Se este projeto te ajudou, considere dar uma estrela no repositório!

</div>
