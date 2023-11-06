import numpy as np
import torch


def run_model(weather_points_list):
    weather_points = np.array(weather_points_list)

    from BinaryClassification import BinaryClassification
    model_states = torch.load('./pretrained_wr_v2.pt')['state_dict']
    model = BinaryClassification()
    model.load_state_dict(model_states)

    tensor_input = torch.tensor(weather_points).float()
    tensor_input = tensor_input.reshape(1, -1)
    model.eval()
    probs = torch.sigmoid(model.forward(tensor_input))

    return bool((probs >= 0.5).int())