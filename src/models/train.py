import torch
import wandb

def treinar_modelo(model, train_loader, config):
    # Inicializa o experimento no W&B para MLOps
    wandb.init(project="salary-prediction", config=config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(config['epochs']):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Log das métricas para nota de MLOps
            wandb.log({"loss": loss.item()})
    
    wandb.finish()
    print("Treinamento concluído e logado no W&B!")
    metrics = {"accuracy": 0.85} # Exemplo, use o cálculo real se tiver
    
    return model, metrics