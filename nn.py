import torch
import torch.nn as nn

class LinearDelayPredictor(nn.Module):
    def __init__(self, input_dim=11):
        super(LinearDelayPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x_normalized, sta):
        pred_delay = self.linear(x_normalized).squeeze(-1)
        pred_delay = torch.clamp(pred_delay, min=-45.0, max=150.0) 
        pred_ata = sta + pred_delay
        return pred_ata, pred_delay