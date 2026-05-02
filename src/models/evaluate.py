import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb

def evaluate_and_plot(model, X_test, y_test):
    """Generates a confusion matrix and uploads it to WandB."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plt.title("Matriz de Confusão - Predição Salarial")
    # Save confusion matrix in W&B
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()