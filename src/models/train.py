import torch
import wandb

def train_model(model, train_loader, config):
    """
    Initializes W&B experiment, trains the model, and logs metrics.
    """
    # Initialize W&B experiment for MLOps tracking
    wandb.init(project="salary-prediction", config=config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(config['epochs']):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Log metrics for MLOps compliance
            wandb.log({"loss": loss.item()})
    
    wandb.finish()
    print("Training complete and logged to W&B!")
    
    # Placeholder for actual validation logic
    metrics = {"accuracy": 0.85} 
    
    return model, metrics