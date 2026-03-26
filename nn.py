import torch
import torch.nn as nn

class LinearDelayPredictor(nn.Module):
    def __init__(self, input_dim=3):
        super(LinearDelayPredictor, self).__init__()
        # 输入: [feat_weather, buffer, interval_next], 输出: 预测的延误时间
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_normalized, sta):
        pred_delay = self.linear(x_normalized).squeeze(-1)
        pred_ata = sta + pred_delay
        return pred_ata, pred_delay

