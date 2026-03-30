import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from nn import LinearDelayPredictor
from utils import rebuild_instance_from_group,augment_features

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


def train_po_baseline(prefix, num_flights=20):
    print("=== 开始 PO 基准训练===")


    df_train = pd.read_csv(f"{prefix}-Train.csv")
    df_val = pd.read_csv(f"{prefix}-Val.csv")
    
    # 构建全局 11 维归一化，使用有效区间 K=8~15
    all_aug_features = []
    for k in range(8, 16):
        raw = df_train[['feat_weather', 'buffer', 'interval_next']].values
        all_aug_features.append(augment_features(raw, k, num_flights))
    all_aug_features = np.vstack(all_aug_features)
    g_mean_11d = np.mean(all_aug_features, axis=0)
    g_std_11d = np.std(all_aug_features, axis=0) + 1e-6

    # 只用 3 维特征进行绝对无偏的拟合
    model_3d = LinearDelayPredictor(input_dim=3)
    optimizer = optim.Adam(model_3d.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    mse_loss_fn = nn.MSELoss()

    best_val_mse = float('inf')
    best_weights_3d = None
    best_bias_3d = None
    epochs_no_improve = 0

    for epoch in range(500):
        model_3d.train()
        for day in df_train['day_id'].unique():
            optimizer.zero_grad()
            day_df = df_train[df_train['day_id'] == day].sort_values('flight_id')
            raw = day_df[['feat_weather', 'buffer', 'interval_next']].values
            
            norm_3d = (raw - g_mean_11d[:3]) / g_std_11d[:3]
            x_tensor = torch.tensor(norm_3d, dtype=torch.float32)
            sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
            true_ata = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
            
            pred_ata, _ = model_3d(x_tensor, sta_tensor)
            loss = mse_loss_fn(pred_ata, true_ata)
            loss.backward()
            optimizer.step()

        model_3d.eval()
        val_mse = 0.0
        with torch.no_grad():
            for day in df_val['day_id'].unique():
                day_df = df_val[df_val['day_id'] == day].sort_values('flight_id')
                raw = day_df[['feat_weather', 'buffer', 'interval_next']].values
                norm_3d = (raw - g_mean_11d[:3]) / g_std_11d[:3]
                x_tensor = torch.tensor(norm_3d, dtype=torch.float32)
                sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
                true_ata = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
                
                pred_ata, _ = model_3d(x_tensor, sta_tensor)
                val_mse += mse_loss_fn(pred_ata, true_ata).item()
                
        val_mse /= len(df_val['day_id'].unique())
        scheduler.step(val_mse)

        if val_mse < best_val_mse - 1e-4:
            best_val_mse = val_mse
            best_weights_3d = model_3d.linear.weight.data.clone()
            best_bias_3d = model_3d.linear.bias.data.clone()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 30:
            break

    state_dict_11d = {
        'linear.weight': torch.cat([best_weights_3d, torch.zeros(1, 8)], dim=1),
        'linear.bias': best_bias_3d
    }

    torch.save({'state_dict': state_dict_11d, 'g_mean': g_mean_11d, 'g_std': g_std_11d}, f"{prefix}-PO_Best.pth")
    print(f"PO 训练结束 (最低 MSE: {best_val_mse:.2f})。")



if __name__ == "__main__":
    train_po_baseline(prefix = "toy_data/D50-F20-K10")

