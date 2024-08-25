import os
import torch
import numpy as np
from model import PyTorchRegressor, MLP
from utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config):
    # Initialize the model
    model_params = config['model']
    training_params = config['training']
    
    if not os.path.exists(config['output']['model_save_path']):
        raise FileNotFoundError("Model file not found. Please train the model first.")

    # Load the saved model state
    mlp = MLP(**model_params)
    mlp.load_state_dict(torch.load(config['output']['model_save_path']))
    
    # Initialize the PyTorchRegressor
    mlp_regressor = PyTorchRegressor(model=mlp, **training_params)
     
    return mlp_regressor

def predict(model, X):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model.predict(X)
    return predictions.flatten()

def inference(config, input_data: np.ndarray=None, model=None):    
    if not model:
        model = load_model(config)
    
    # Perform inference
    predictions = predict(model, input_data)
    return predictions

if __name__ == "__main__":
    # Example input data
    X = np.array([[ 0.69244414, -0.78913691,  1.06185505, -0.14785831,  0.36025724,  1.04140897,
                0.46058942, -0.60517324,  1.41441897, -0.03905156,  0.1218106,  -0.81435728,
               -0.08680354, -0.08078841, -0.0912004,  -0.21545621, -0.10440862,  0.45791971,
                1.,          0.        ]])
    
    # Paths to the configuration and model files
    config = load_config("config.yaml")

    # Perform inference
    predictions = inference(config, X)
    print("Predictions:", predictions)