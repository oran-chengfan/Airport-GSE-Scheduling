import torch
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from gurobipy import GRB
from solver import GSESolver
from utils import rebuild_instance_from_group
from nn import LinearDelayPredictor

class DFL_Surrogate_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_ata_tensor, true_ata_tensor, instance, config):
        """
        前向传播：计算 Surrogate Loss，并为反向传播准备雅可比矩阵
        """
        flights = sorted(instance['flights'])
        num_flights = len(flights)
        
        # 将 PyTorch Tensor 转换为 Gurobi 友好的字典
        pred_ata = {fn: pred_ata_tensor[i].item() for i, fn in enumerate(flights)}
        true_ata = {fn: true_ata_tensor[i].item() for i, fn in enumerate(flights)}

        solver = GSESolver(config)

        # =========================================================
        # [加速策略 1] Phase 1: MILP 寻路 (找出活跃子图)
        # =========================================================
        model_mip, vars_mip, *_ = solver.build_model(instance, pred_ata, relax=False)
        model_mip.setParam("OutputFlag", 0)
        model_mip.setParam("Threads", 1)
        model_mip.setParam("MIPGap", 0.05) # 允许 5% 的 gap 极速出解
        model_mip.optimize()
        
        if model_mip.SolCount == 0:
            # 如果极端情况下无解，熔断并返回 0 梯度
            ctx.grad_pred_ata = torch.zeros_like(pred_ata_tensor)
            return torch.tensor(0.0, dtype=torch.float32)

        x_mip = vars_mip['x']
        active_edges = [(i, j) for (i, j) in x_mip if x_mip[i, j].X > 0.5]

        # =========================================================
        # [加速策略 2] Phase 2 & 3: 名义 LP 与 真实评估 LP
        # =========================================================
        # 构建极简多面体 (Reduced LP)，剔除所有无用连边与 Big-M
        model_1, vars_1 = solver.build_reduced_model(instance, pred_ata, active_edges)
        model_1.setParam("OutputFlag", 0)
        model_1.setParam("Threads", 1)
        model_1.setParam("Method", 2)    # 强制使用内点法 (Barrier)
        model_1.setParam("Crossover", 0) # [关键加速] 禁用交叉验证，省去大量寻点时间
        model_1.optimize()
        
        # 记录名义环境下的最优时间分配
        X1_vals = {k: var.X for k, var in vars_1['t'].items()}

        # 在真实环境下评估成本，并提取影子价格 (Shadow Prices)
        model_2, vars_2 = solver.build_reduced_model(instance, true_ata, active_edges)
        model_2.setParam("OutputFlag", 0)
        model_2.setParam("Threads", 1)
        model_2.setParam("Method", 2)
        model_2.setParam("Crossover", 0)
        
        fix_constrs = {}
        for k, var in vars_2['t'].items():
            var.LB = -GRB.INFINITY
            var.UB = GRB.INFINITY
            # 强制固定时间，计算事后补救成本
            fix_constrs[k] = model_2.addConstr(var == X1_vals[k])
                
        model_2.optimize()
        surr_cost = model_2.ObjVal
        gx_dict = {k: c.Pi for k, c in fix_constrs.items()} # 提取梯度方向

        # =========================================================
        # [数学锚点] Phase 4: KKT 伴随法求导
        # =========================================================
        all_vars = model_1.getVars() 
        var2idx = {v: i for i, v in enumerate(all_vars)}
        n_vars = len(all_vars)

        g = np.zeros(n_vars)
        for k, var in vars_1['t'].items():
            g[var2idx[var]] = gx_dict[k]

        A = model_1.getA() 
        senses = np.array(model_1.getAttr('Sense', model_1.getConstrs())) 
        slacks = np.array(model_1.getAttr('Slack', model_1.getConstrs())) 

        eq_idx = np.where(senses == '=')[0]
        ineq_le_idx = np.where(senses == '<')[0]
        ineq_ge_idx = np.where(senses == '>')[0]

        A_eq = A[eq_idx, :] if len(eq_idx) > 0 else sparse.csr_matrix((0, n_vars))
        A_le = A[ineq_le_idx, :] if len(ineq_le_idx) > 0 else sparse.csr_matrix((0, n_vars))
        A_ge = -A[ineq_ge_idx, :] if len(ineq_ge_idx) > 0 else sparse.csr_matrix((0, n_vars))
        A_ineq = sparse.vstack([A_le, A_ge]) 

        # 松弛变量裁剪防除零爆炸
        S_ineq = np.concatenate([slacks[ineq_le_idx], slacks[ineq_ge_idx]]) 
        S_ineq = np.clip(S_ineq, 1e-4, None) 

        # 构造 Hessian 矩阵
        mu = 1e-3 
        S_inv2 = 1.0 / (S_ineq**2) 
        H = mu * A_ineq.T.dot(sparse.diags(S_inv2)).dot(A_ineq)
        
        # [加速与稳定策略 3] 严谨的 Tikhonov 正则化
        rho = 1e-5
        delta = 1e-6
        m_eq = A_eq.shape[0]
        
        K = sparse.bmat([
            [H + sparse.eye(n_vars) * rho, A_eq.T], 
            [A_eq, -delta * sparse.eye(m_eq)]
        ], format='csc')
        
        b_adj = np.concatenate([g, np.zeros(m_eq)])
        
        # 利用稀疏矩阵极速求逆
        z = splinalg.spsolve(K, b_adj)
        z1 = z[:n_vars]

        # =========================================================
        # [数学锚点] 提取复合雅可比矩阵 (Jacobian)
        # =========================================================
        J_b_orig = np.zeros((A.shape[0], num_flights))
        constrs = model_1.getConstrs()
        
        for i, c in enumerate(constrs):
            if c.ConstrName.startswith("pred_ata_svc_"):
                f_idx = flights.index(c.ConstrName.replace("pred_ata_svc_", ""))
                J_b_orig[i, f_idx] = 1.0 
            elif c.ConstrName.startswith("pred_ata_wait_"):
                f_idx = flights.index(c.ConstrName.replace("pred_ata_wait_", ""))
                J_b_orig[i, f_idx] = -1.0

        J_b_ineq = np.vstack([
            J_b_orig[ineq_le_idx, :] if len(ineq_le_idx) > 0 else np.zeros((0, num_flights)),
            -J_b_orig[ineq_ge_idx, :] if len(ineq_ge_idx) > 0 else np.zeros((0, num_flights))
        ])

        V_a = mu * A_ineq.T.dot(sparse.diags(S_inv2)).dot(J_b_ineq)
        grad_pred_ata = z1.dot(V_a)
            
        # 梯度截断，防止由于极端天气导致某航班梯度爆炸带偏整个网络
        grad_norm = np.linalg.norm(grad_pred_ata)
        if grad_norm > 10.0:
            grad_pred_ata = grad_pred_ata / grad_norm * 10.0

        ctx.grad_pred_ata = torch.tensor(grad_pred_ata, dtype=torch.float32)
        return torch.tensor(surr_cost, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        # 链式法则：运筹梯度 * 网络传来的上游梯度
        return ctx.grad_pred_ata * grad_output, None, None, None

def train_dfl(prefix):
    print("=== 开始 DFL 级联延误防守联合训练 ===")
    import copy
    from gurobipy import GRB
    
    # 1. 加载训练期配置 (恒定 K=10 环境)
    config = json.load(open('./toy_data/config.json', 'r'))
    
    df_train = pd.read_csv(f"{prefix}-Train.csv")
    df_val = pd.read_csv(f"{prefix}-Val.csv")
    days_train = df_train['day_id'].unique()
    days_val = df_val['day_id'].unique()
    
    # 2. [核心修复] 严格继承 PO 的特征归一化空间
    model = LinearDelayPredictor(input_dim=3)
    po_checkpoint = torch.load(f"{prefix}-PO_Best.pth", weights_only=False)
    global_mean = po_checkpoint['g_mean']
    global_std = po_checkpoint['g_std']
    model.load_state_dict(po_checkpoint['state_dict'])
    
    # 3. [稳定策略] 降低学习率，强化权重锚定
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    num_epochs = 100
    batch_size = 4
    patience = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    best_val_surr = float('inf')
    best_val_regret = float('inf')
    epochs_no_improve_regret = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # ==========================================================
        # 阶段一：训练集反向传播
        # ==========================================================
        model.train()
        epoch_train_surr = 0.0
        optimizer.zero_grad()
        
        for i, day in enumerate(days_train):            
            day_df = df_train[df_train['day_id'] == day].sort_values('flight_id')
            instance = rebuild_instance_from_group(day, day_df)
            
            raw_features = day_df[['feat_weather', 'buffer', 'interval_next']].values
            norm_features = (raw_features - global_mean) / global_std
            
            x_tensor = torch.tensor(norm_features, dtype=torch.float32)
            sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
            true_ata_tensor = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
            
            pred_ata_tensor, pred_delay = model(x_tensor, sta_tensor)
            
            loss = DFL_Surrogate_Function.apply(pred_ata_tensor, true_ata_tensor, instance, config)
            (loss / batch_size).backward()            
            epoch_train_surr += loss.item()

            if (i + 1) % batch_size == 0 or (i + 1) == len(days_train):
                # 防止单日天气极端导致的梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                
        scheduler.step()       
        avg_train_surr = epoch_train_surr / len(days_train)

        # ==========================================================
        # 阶段二：验证集前向评测 (物理仿真)
        # ==========================================================
        model.eval()
        epoch_val_surr = 0.0
        epoch_val_regret = 0.0
        
        with torch.no_grad():
            for day in days_val:
                day_df = df_val[df_val['day_id'] == day].sort_values('flight_id')
                instance = rebuild_instance_from_group(day, day_df)
                flights = sorted(instance['flights'])
                
                raw_features = day_df[['feat_weather', 'buffer', 'interval_next']].values
                norm_features = (raw_features - global_mean) / global_std
                x_tensor = torch.tensor(norm_features, dtype=torch.float32)
                sta_tensor = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
                
                pred_ata_tensor, _ = model(x_tensor, sta_tensor)
                pred_ata = {fn: pred_ata_tensor[j].item() for j, fn in enumerate(flights)}
                true_ata = {fn: day_df['ata_min'].values[j] for j, fn in enumerate(flights)}
                
                solver = GSESolver(config)
                model_mip, vars_mip, *_ = solver.build_model(instance, pred_ata, relax=False)
                model_mip.setParam("OutputFlag", 0)
                model_mip.optimize()
                
                if model_mip.SolCount == 0:
                    val_surr, val_regret = 100000.0, 100000.0
                else:
                    active_edges = [(u, v) for (u, v) in vars_mip['x'] if vars_mip['x'][u, v].X > 0.5]
                    
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
                    val_surr = model_2.ObjVal
                    
                    model_3, _ = solver.build_reduced_model(instance, true_ata, active_edges)
                    model_3.setParam("OutputFlag", 0)
                    model_3.optimize()
                    val_regret = model_3.ObjVal
                
                epoch_val_surr += val_surr
                epoch_val_regret += val_regret

        avg_val_surr = epoch_val_surr / len(days_val)
        avg_val_regret = epoch_val_regret / len(days_val)

        # ==========================================================
        # 阶段三：基于物理代价 (Regret) 的单点验证
        # ==========================================================
        if avg_val_regret < best_val_regret - 1e-4:
            best_val_regret = avg_val_regret
            best_val_surr = avg_val_surr
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve_regret = 0
        else:
            epochs_no_improve_regret += 1

        print(f"Epoch [{epoch+1:03d}/{num_epochs}] | LR: {scheduler.get_last_lr()[0]:.4f} | "
              f"Tr Surr: {avg_train_surr:.1f} | Val Surr: {avg_val_surr:.1f} | Val Regret: {avg_val_regret:.1f}")
        
        weights = model.linear.weight.data.numpy().flatten()
        print(f"  [物理权重追踪] -> 天气: {weights[0]:.2f}, 缓冲: {weights[1]:.2f}, 间隔: {weights[2]:.2f}")

        if epochs_no_improve_regret >= patience:
            print(f"--> [物理截断] Epoch {epoch+1}: 连续 {patience} 轮 Regret 未降低，停止训练。")
            break

    torch.save({'state_dict': best_model_state, 'g_mean': global_mean, 'g_std': global_std}, f"{prefix}-DFL_Best.pth")
    print(f"DFL 模型就绪。最佳物理代价 (Regret): {best_val_regret:.2f}")


if __name__ == "__main__":
    train_dfl(prefix = "toy_data/Cascade_D50-F20-K10")
