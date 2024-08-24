import pandas as pd
import torch
from data_pipeline import get_datasets
from model import MLP, PyTorchRegressor
from utils import load_config


def train(config_path="config.yaml"):
    config = load_config(config_path)
    
    # Load data and preprocess the data
    X_train, X_test, y_train, y_test = get_datasets(config)

    # Training
    model_params = config['model']
    training_params = config['training']
    input_size = X_train.shape[1]
    
    mlp = MLP(**model_params)
    mlp_regressor = PyTorchRegressor(model=mlp, **training_params)
    mlp_regressor.fit(X_train, y_train)
    
    # Save model
    torch.save(mlp.state_dict(), config['output']['model_save_path'])

if __name__ == "__main__":
    train()
