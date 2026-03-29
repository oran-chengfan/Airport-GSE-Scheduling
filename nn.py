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

class PredictiveMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # 确保输出的维度与原有 Linear 模型一致
        return self.net(x)

