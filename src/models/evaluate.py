import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb

def evaluate_and_plot(model, X_test, y_test):
    """Gera matriz de confusão e envia para o WandB."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plt.title("Matriz de Confusão - Predição Salarial")
    # Salva o gráfico para o WandB
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()