# gradient_check.py
import torch
import json
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from solver import GSESolver
from gurobipy import GRB
from utils import rebuild_instance_from_group

class DFL_Surrogate_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_ata_tensor, true_ata_tensor, instance, config):
        flights = sorted(instance['flights'])
        pred_ata = {fn: pred_ata_tensor[i].item() for i, fn in enumerate(flights)}
        true_ata = {fn: true_ata_tensor[i].item() for i, fn in enumerate(flights)}

        solver = GSESolver(config)

        # =========================================================
        # Phase 1: MILP 寻路 (找出严苛物理条件下的活跃子图)
        # =========================================================
        model_mip, vars_mip, *_ = solver.build_model(instance, pred_ata, relax=False)
        model_mip.setParam("Outputflag", 0)
        model_mip.optimize()
        
        if model_mip.SolCount == 0:
            ctx.grad_pred_ata = torch.zeros_like(pred_ata_tensor)
            return torch.tensor(0.0)

        x_mip = vars_mip['x']
        active_edges = [(i, j) for (i, j) in x_mip if x_mip[i, j].X > 0.5]

        # =========================================================
        # Phase 2 & 3: 名义 LP 与 事后评估 LP (严格降维)
        # =========================================================
        model_1, vars_1 = solver.build_reduced_model(instance, pred_ata, active_edges)
        model_1.setParam("Method", 2)
        model_1.setParam("Crossover", 0)
        model_1.setParam("Outputflag", 0)
        model_1.optimize()
        
        X1_vals = {k: var.X for k, var in vars_1['t'].items()}

        model_2, vars_2 = solver.build_reduced_model(instance, true_ata, active_edges)
        model_2.setParam("Outputflag", 0)
        
        fix_constrs = {}
        for k, var in vars_2['t'].items():
            var.LB = -GRB.INFINITY
            var.UB = GRB.INFINITY
            fix_constrs[k] = model_2.addConstr(var == X1_vals[k])
                
        model_2.optimize()
        surr_cost = model_2.ObjVal
        gx_dict = {k: c.Pi for k, c in fix_constrs.items()}

        # =========================================================
        # Phase 4: 数值稳定的 KKT 伴随法求导
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

        S_ineq = np.concatenate([slacks[ineq_le_idx], slacks[ineq_ge_idx]]) 
        S_ineq = np.clip(S_ineq, 1e-4, None) # 放宽裁断阈值，避免除零爆炸

        # 构造 Hessian 矩阵
        mu = 1e-3 # Barrier 参数
        S_inv2 = 1.0 / (S_ineq**2) 
        H = mu * A_ineq.T.dot(sparse.diags(S_inv2)).dot(A_ineq)
        
        # 严谨的 Tikhonov 正则化 (原始空间与对偶空间双重加固)
        rho = 1e-5
        delta = 1e-6
        H = H + sparse.eye(n_vars) * rho
        m_eq = A_eq.shape[0]
        
        # 构建非奇异准定矩阵
        K = sparse.bmat([
            [H, A_eq.T], 
            [A_eq, -delta * sparse.eye(m_eq)]
        ], format='csc')
        
        b_adj = np.concatenate([g, np.zeros(m_eq)])
        
        # 使用 spsolve 或 lsqr 求解
        z = splinalg.spsolve(K, b_adj)
        z1 = z[:n_vars]

        num_flights = len(flights)
        J_b_orig = np.zeros((A.shape[0], num_flights))
        constrs = model_1.getConstrs()
        
        # 精确提取预测参数在约束 RHS (右端项) 中的雅可比矩阵
        for i, c in enumerate(constrs):
            if c.ConstrName.startswith("pred_ata_svc_"):
                # 约束: s[arr_n] >= ata  => RHS = ata  => 导数为 1.0
                f_idx = flights.index(c.ConstrName.replace("pred_ata_svc_", ""))
                J_b_orig[i, f_idx] = 1.0 
                
            elif c.ConstrName.startswith("pred_ata_wait_"):
                # 约束: plane_wait[fn] - s[arr_n] >= -ata => RHS = -ata => 导数为 -1.0
                f_idx = flights.index(c.ConstrName.replace("pred_ata_wait_", ""))
                J_b_orig[i, f_idx] = -1.0

        J_b_ineq = np.vstack([
            J_b_orig[ineq_le_idx, :] if len(ineq_le_idx) > 0 else np.zeros((0, num_flights)),
            -J_b_orig[ineq_ge_idx, :] if len(ineq_ge_idx) > 0 else np.zeros((0, num_flights))
        ])

        V_a = mu * A_ineq.T.dot(sparse.diags(S_inv2)).dot(J_b_ineq)
        grad_pred_ata = z1.dot(V_a)
            
        ctx.grad_pred_ata = torch.tensor(grad_pred_ata, dtype=torch.float32)
        return torch.tensor(surr_cost, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pred_ata * grad_output, None, None, None


if __name__ == "__main__":
    # ==========================================
    # 核心测试：梯度能否指向关键飞机？
    # ==========================================
    config = json.load(open('./toy_data/config.json', 'r'))
    df = pd.read_csv("toy_data/dfl_train_data.csv")
    
    # 提取 Day 0 的数据
    day_df = df[df['day_id'] == 0].sort_values('flight_id')
    instance = rebuild_instance_from_group(0, day_df)
    flights = sorted(instance['flights'])
    
    sta = torch.tensor(day_df['sta_min'].values, dtype=torch.float32)
    true_ata = torch.tensor(day_df['ata_min'].values, dtype=torch.float32)
    
    # 人为构造一个错误的预测：假设所有航班预测均为 STA
    # 我们设置 requires_grad=True，观察反向传播的梯度
    pred_ata = sta.clone().detach().requires_grad_(True)
    
    print("真实 ATA: ", true_ata.tolist())
    print("预测 ATA: ", pred_ata.tolist())
    
    # 执行前向传播计算 Surrogate Loss
    surr_loss = DFL_Surrogate_Function.apply(pred_ata, true_ata, instance, config)
    print(f"\nSurrogate Loss: {surr_loss.item():.4f}")
    
    # 执行反向传播
    surr_loss.backward()
    
    # 打印梯度ga
    print("\n========= 梯度检查 (Gradient Check) =========")
    grads = pred_ata.grad.tolist()
    for f_id, g in zip(flights, grads):
        # 寻找梯度绝对值最大的航班
        marker = " <--- 关键航班 (Critical Flight)!" if abs(g) == max(map(abs, grads)) and abs(g) > 1e-4 else ""
        print(f"{f_id}: 梯度 = {g:.6f} {marker}")

