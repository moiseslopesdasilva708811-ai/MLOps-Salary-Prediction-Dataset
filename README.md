# Pipeline MLOps: Classificação de Renda (Adult Census)

[cite_start]Este repositório contém o desenvolvimento de uma rede neural **Multilayer Perceptron (MLP)** para classificar se um indivíduo recebe mais de 50k anuais[cite: 13, 62]. [cite_start]O foco do projeto é a implementação de um ciclo de vida sustentável e rastreável utilizando práticas de **MLOps**[cite: 8, 46].

---

## 🛠 Metodologia Aplicada

[cite_start]Para atender às exigências da disciplina de **Tópicos Avançados em IA (UFRN)**, o projeto seguiu estas etapas fundamentais[cite: 4, 14, 21]:

* [cite_start]**Saneamento de Dados:** Limpeza de ruídos (remoção de valores `?`) e tratamento de outliers utilizando o método **IQR**[cite: 25, 26].
* [cite_start]**Seleção de Variáveis:** Cruzamento de três métodos (**Correlação**, **Random Forest** e **Mutual Information**) para gerar um ranking robusto de importância[cite: 29, 31, 32].
* [cite_start]**Arquitetura da MLP:** Implementação de rede profunda com justificativa técnica de hiperparâmetros como camadas ocultas, neurónios, funções de ativação e taxa de aprendizagem[cite: 35, 37, 40].
* [cite_start]**Monitoramento e MLOps:** Versionamento de artefatos, logs de métricas por época e armazenamento de hiperparâmetros via **Weights & Biases (W&B)**[cite: 47, 49, 50, 51].

---

## 🗂 Estrutura do Projeto

[cite_start]A organização das pastas segue as diretrizes de reprodutibilidade e organização de código[cite: 81]:

```text
MLOps-Salary-Prediction-Dataset/
├── data/          # Datasets (raw e clean)
├── notebooks/     # Análise exploratória e prototipagem
├── src/           # Scripts Python e pipeline de treino
├── models/        # Artefatos do modelo (.pt)
├── reports/       # Relatório técnico (PDF) e gráficos
├── requirements.txt
└── README.md
