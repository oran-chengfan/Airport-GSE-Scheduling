import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from gurobipy import GRB

from nn import LinearDelayPredictor
from solver import GSESolver
from utils import rebuild_instance_from_group,augment_features

def evaluate_model(model_name, model_path, test_df, config, test_k):
    import torch
    import torch.nn as nn
    from gurobipy import GRB
    from nn import LinearDelayPredictor
    from solver import GSESolver
    from utils import rebuild_instance_from_group
    import numpy as np

    checkpoint = torch.load(model_path, weights_only=False)
    g_mean = checkpoint['g_mean']
    g_std = checkpoint['g_std']
    
    # [修复点 1]：初始化新版无参的残差网络
    model = LinearDelayPredictor()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    mse_loss_fn = nn.MSELoss()
    days_test = test_df['day_id'].unique()
    
    total_mse, total_surr, total_regret = 0.0, 0.0, 0.0
    solver = GSESolver(config)
    
    with torch.no_grad():
        for day in days_test:
            day_df = test_df[test_df['day_id'] == day].sort_values('flight_id')
            instance = rebuild_instance_from_group(day, day_df)
            flights = sorted(instance['flights'])
            
            raw_features = day_df[['feat_weather', 'buffer', 'interval_next']].values
            norm_features = (raw_features - g_mean) / g_std
            x_base = torch.tensor(norm_features, dtype=torch.float32)
            
            # 计算当前测试环境的拥挤度
            c_val = 10.0 / test_k
            x_cong = torch.full((len(day_df), 1), c_val, dtype=torch.float32)
            
            sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
            true_ata_tensor = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
            
            # 前向传播需要同时传入 base 和 cong
            pred_ata_tensor, _ = model(x_base, x_cong, sta_tensor)
            
            day_mse = mse_loss_fn(pred_ata_tensor, true_ata_tensor).item()
            total_mse += day_mse
            
            pred_ata = {fn: pred_ata_tensor[j].item() for j, fn in enumerate(flights)}
            true_ata = {fn: true_ata_tensor[j].item() for j, fn in enumerate(flights)}
            
            model_mip, vars_mip, *_ = solver.build_model(instance, pred_ata, relax=False)
            model_mip.setParam("OutputFlag", 0)
            model_mip.setParam("Threads", 1)
            model_mip.optimize()
            
            if model_mip.SolCount == 0:
                day_surr, day_regret = 100000.0, 100000.0
            else:
                x_mip = vars_mip['x']
                active_edges = [(u, v) for (u, v) in x_mip if x_mip[u, v].X > 0.5]
                
                model_1, vars_1 = solver.build_reduced_model(instance, pred_ata, active_edges)
                model_1.setParam("OutputFlag", 0)
                model_1.optimize()
                X1_vals = {k: var.X for k, var in vars_1['t'].items()}
                
                model_2, vars_2 = solver.build_reduced_model(instance, true_ata, active_edges)
                model_2.setParam("OutputFlag", 0)
                for k, var in vars_2['t'].items():
                    var.LB = -GRB.INFINITY
                    var.UB = GRB.INFINITY
                    model_2.addConstr(var == X1_vals[k])
                model_2.optimize()
                day_surr = model_2.ObjVal
                
                model_3, _ = solver.build_reduced_model(instance, true_ata, active_edges)
                model_3.setParam("OutputFlag", 0)
                model_3.optimize()
                day_regret = model_3.ObjVal
                
            total_surr += day_surr
            total_regret += day_regret

    num_days = len(days_test)
    return total_mse / num_days, total_regret / num_days, total_surr / num_days

if __name__ == "__main__":
    prefix = "toy_data/D50-F20-K10"
    config_path = "./toy_data/config.json"
    test_csv_path = f"{prefix}-Test.csv"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    test_df = pd.read_csv(test_csv_path)
    num_vehicles = len(config.get("fleets",{}))

    print(f"=== 开始在测试集上对抗评估 (Test Days: {len(test_df['day_id'].unique())}) ===")
    
    # 评估 PO 基线模型
    po_model_path = f"{prefix}-PO_Best.pth"
    po_mse, po_regret, po_surr = evaluate_model("PO (Baseline)", po_model_path, test_df, config)
    
    # 评估 DFL 模型
    dfl_model_path = f"{prefix}-DFL_Best.pth"
    dfl_mse, dfl_regret, dfl_surr = evaluate_model("DFL (Proposed)", dfl_model_path, test_df, config)

    print("\n" + "="*50)
    print(f"对比结果: {num_vehicles}辆车")
    print("="*50)
    # 使用制表符对齐输出，保留 4 位小数
    print(f"{'':<10} | {'MSE':<12} | {'Regret':<12} | {'Surrogate':<12}")
    print("-" * 50)
    print(f"{'PO':<10} | {po_mse:<12.4f} | {po_regret:<12.4f} | {po_surr:<12.4f}")
    print(f"{'DFL':<10} | {dfl_mse:<12.4f} | {dfl_regret:<12.4f} | {dfl_surr:<12.4f}")
    print("="*50)
    