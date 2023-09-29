import numpy as np
import mlflow
import torch


def run_model(weather_points_list):
    weather_points = np.array(weather_points_list)

    model = mlflow.pytorch.load_model('models:/WR_pretrained_model/1')

    tensor_input = torch.tensor(weather_points).float()
    tensor_input = tensor_input.reshape(1, -1)
    model.eval()
    probs = torch.sigmoid(model.forward(tensor_input))

    return bool((probs >= 0.5).int())