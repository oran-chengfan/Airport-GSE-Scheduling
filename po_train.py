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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    import copy
    from nn import LinearDelayPredictor
    from utils import rebuild_instance_from_group

    df_train = pd.read_csv(f"{prefix}-Train.csv")
    df_val = pd.read_csv(f"{prefix}-Val.csv")
    
    # 提取 3 维基础特征
    raw_features = df_train[['feat_weather', 'buffer', 'interval_next']].values
    global_mean = np.mean(raw_features, axis=0)
    global_std = np.std(raw_features, axis=0) + 1e-6

    model = LinearDelayPredictor(input_dim=3)
    # 严格去除 weight_decay，确保获取无偏估计
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss_fn = nn.MSELoss()

    best_val_mse = float('inf')
    best_model_state = None
    patience = 50
    epochs_no_improve = 0
    num_epochs = 10000000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for day in df_train['day_id'].unique():
            day_df = df_train[df_train['day_id'] == day].sort_values('flight_id')
            
            x_raw = day_df[['feat_weather', 'buffer', 'interval_next']].values
            x_norm = (x_raw - global_mean) / global_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32)
            
            sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
            true_ata_tensor = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
            
            pred_ata, _ = model(x_tensor, sta_tensor)
            loss = mse_loss_fn(pred_ata, true_ata_tensor)
            
            loss.backward()
            epoch_loss += loss.item()
            
        optimizer.step()

        # 验证集前向
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for day in df_val['day_id'].unique():
                day_df = df_val[df_val['day_id'] == day].sort_values('flight_id')
                x_raw = day_df[['feat_weather', 'buffer', 'interval_next']].values
                x_norm = (x_raw - global_mean) / global_std
                
                x_tensor = torch.tensor(x_norm, dtype=torch.float32)
                sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
                true_ata_tensor = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
                
                pred_ata, _ = model(x_tensor, sta_tensor)
                val_mse += mse_loss_fn(pred_ata, true_ata_tensor).item()
                
        val_mse /= len(df_val['day_id'].unique())

        if val_mse < best_val_mse - 1e-4:
            best_val_mse = val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"--> [PO 早停] Epoch {epoch+1} 触发。")
            break

    torch.save({'state_dict': best_model_state, 'g_mean': global_mean, 'g_std': global_std}, f"{prefix}-PO_Best.pth")
    print(f"PO训练完成：最佳 Val MSE: {best_val_mse:.2f}")

if __name__ == "__main__":
    train_po_baseline(prefix = "toy_data/D50-F20-S42")

