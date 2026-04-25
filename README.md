# Pipeline MLOps: Classificação de Renda (Adult Census)

Este repositório contém o desenvolvimento de uma rede neural **Multilayer Perceptron (MLP)** para classificar se um indivíduo recebe mais de 50k anuais. O foco do projeto é a implementação de um ciclo de vida sustentável e rastreável utilizando práticas de **MLOps**.

---

## 🛠 Metodologia Aplicada

Para atender às exigências da disciplina de **Tópicos Avançados em IA (UFRN)**, o projeto seguiu estas etapas fundamentais:

* **Saneamento de Dados:** Limpeza de ruídos (remoção de valores `?`) e tratamento de outliers utilizando o método **IQR**.
* **Seleção de Variáveis:** Cruzamento de três métodos (**Correlação**, **Random Forest** e **Mutual Information**) para gerar um ranking robusto de importância.
* **Arquitetura da MLP:** Implementação de rede profunda com justificativa técnica de hiperparâmetros como camadas ocultas, neurônios, funções de ativação e taxa de aprendizagem.
* **Monitoramento e MLOps:** Versionamento de artefatos, logs de métricas por época e armazenamento de hiperparâmetros via **Weights & Biases (W&B)**.

---

## 🗂 Estrutura do Projeto

A organização das pastas segue boas práticas de reprodutibilidade e organização de código:

```text
MLOps-Salary-Prediction-Dataset/
├── data/          # Datasets (raw e clean)
├── notebooks/     # Análise exploratória e prototipagem
├── src/           # Scripts Python e pipeline de treino
├── models/        # Artefatos do modelo (.pt)
├── reports/       # Relatório técnico (PDF) e gráficos
├── requirements.txt
└── README.md
