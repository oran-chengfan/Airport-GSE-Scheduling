import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from nn import LinearDelayPredictor

def load_and_norm_data(csv_path, global_mean=None, global_std=None):
    df = pd.read_csv(csv_path)
    features = df[['feat_weather', 'buffer', 'interval_next']].values
    
    if global_mean is None:
        global_mean = np.mean(features, axis=0)
        global_std = np.std(features, axis=0) + 1e-6
        
    norm_features = (features - global_mean) / global_std
    
    x_tensor = torch.tensor(norm_features, dtype=torch.float32)
    sta_tensor = torch.tensor(df['sta_min'].values, dtype=torch.float32)
    true_ata_tensor = torch.tensor(df['ata_min'].values, dtype=torch.float32)
    return x_tensor, sta_tensor, true_ata_tensor, global_mean, global_std

def train_po_baseline(prefix):
    
    x_train, sta_train, y_train, g_mean, g_std = load_and_norm_data(f"{prefix}-Train.csv")
    x_val, sta_val, y_val, _, _ = load_and_norm_data(f"{prefix}-Val.csv", g_mean, g_std)

    model = LinearDelayPredictor(input_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss_fn = nn.MSELoss()

    num_epochs = 10000000
    patience = 20
    best_val_mse = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    print("=== 开始 PO Baseline 训练 ===")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 训练集纯 MSE 前向传播
        pred_ata_train, _ = model(x_train, sta_train)
        loss = mse_loss_fn(pred_ata_train, y_train)
        loss.backward()
        optimizer.step()

        # 验证集无梯度评估
        model.eval()
        with torch.no_grad():
            pred_ata_val, _ = model(x_val, sta_val)
            val_mse = mse_loss_fn(pred_ata_val, y_val).item()

        # Early Stopping 逻辑
        if val_mse < best_val_mse - 1e-4:
            best_val_mse = val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:03d} | Train MSE: {loss.item():.4f} | Val MSE: {val_mse:.4f} | Best Val MSE: {best_val_mse:.4f}")

        if epochs_no_improve >= patience:
            print(f"--> [早停触发] Epoch {epoch+1}: 验证集 MSE 连续 {patience} 轮未降低。")
            break

    # 保存最佳模型
    torch.save({'state_dict': best_model_state, 'g_mean': g_mean, 'g_std': g_std}, f"{prefix}-PO_Best.pth")
if __name__ == "__main__":
    train_po_baseline(prefix = "toy_data/D30-F30-S42")

    