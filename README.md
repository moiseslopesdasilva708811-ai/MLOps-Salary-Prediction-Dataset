##Pipeline MLOps: Classificação de Renda (Adult Census)

- Este repositório contém o desenvolvimento de uma rede neural Multilayer Perceptron (MLP) para classificar se um indivíduo recebe mais de 50k anuais. O foco aqui não é apenas o modelo, mas a implementação de um ciclo de vida sustentável via MLOps.

#🛠 O que foi feito (Metodologia)

Para atender às exigências da disciplina de Tópicos Avançados em IA, o projeto seguiu estas etapas:

-Saneamento de Dados: Limpeza de ruídos (como os valores "?" do dataset original) e tratamento de outliers via IQR.
-Seleção de Variáveis: Não confiei em apenas um método. Cruzei Correlação, Random Forest e Mutual Information para definir o que realmente importa para a rede neural.Arquitetura da MLP: Implementação de uma rede profunda com justificativa técnica de cada hiperparâmetro (camadas, neurônios e funções de ativação)
-Monitoramento: Todo o treinamento foi logado no Weights & Biases (W&B), permitindo versionar cada "run" e os pesos do modelo

##Conjunto de pastas
🗂 Estrutura do Projeto
MLOps-Salary-Prediction-Dataset/
│
├── data/
│   └── salary.csv
│
├── notebooks/
│   └── Tratamento_e_limpeza_dados.ipynb
│
├── src/
│   └── download_dataset.py
│
├── requirements.txt
└── README.md

🚀 Como rodarInstale as dependências:Bashpip install -r requirements.txt
Para rodar o pipeline completo e logar no W&B:Bashpython src/mlops_pipeline.py
📊 Resultados PrincipaisAs métricas finais (Acurácia, Precisão, Recall e F1-Score) e a matriz de confusão detalhada podem ser consultadas no Relatório Técnico dentro da pasta /reports.
