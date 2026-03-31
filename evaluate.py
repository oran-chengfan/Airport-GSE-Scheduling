import torch
import torch.nn as nn
from gurobipy import GRB
import numpy as np
from nn import LinearDelayPredictor
from solver import GSESolver
from utils import rebuild_instance_from_group, augment_features

def evaluate_model(model_name, model_path, test_df, config, test_k, num_flights=20):
    checkpoint = torch.load(model_path, weights_only=False)
    g_mean = checkpoint['g_mean']
    g_std = checkpoint['g_std']
    
    model = LinearDelayPredictor(input_dim=11)
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
            
            raw = day_df[['feat_weather', 'buffer', 'interval_next']].values
            
            # [核心修复点]：执行 11 维特征升维与物理截断
            aug = augment_features(raw, test_k, num_flights)
            norm = (aug - g_mean) / g_std
            
            x_tensor = torch.tensor(norm, dtype=torch.float32)
            sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
            true_ata_tensor = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
            
            pred_ata_tensor, _ = model(x_tensor, sta_tensor)
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