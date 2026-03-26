import torch
import torch.nn as nn
from torch.autograd import Function
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import warnings

warnings.filterwarnings("ignore")

class GSE_LP_Builder:
    """
    核心工具类：将简化的 GSE 调度问题转化为标准 LP 矩阵形式。
    Minimize c^T * xi
    s.t.     A_eq * xi = b_eq
             A_in * xi <= b_in (包含动态参数 pred_a)
    """
    def __init__(self, instance, vehicles, big_M=10000000):
        self.instance = instance
        self.vehicles = vehicles
        self.M = big_M
        
        self.nodes = instance['nodes']
        self.flights = instance['flights']
        self.tau = instance.get('travel_times', {})
        self.d_plan = instance['std']
        
        self._build_indices()
        self._build_static_matrices()

    def _is_capable(self, v_id, n_id):
        if n_id == 0 or n_id == len(self.nodes) + 1:
            return True
        task_types = self.nodes[n_id]['type']
        v_types = self.vehicles[v_id]['task_type']
        return any(t in v_types for t in task_types)

    def _build_indices(self):
        self.idx_map = {}
        counter = 0
        
        self.all_vehicles = list(self.vehicles.keys())
        self.V_nodes = [0] + list(self.nodes.keys()) + [len(self.nodes) + 1]
        self.real_nodes = list(self.nodes.keys())
        
        # 1. x_{ijk}
        self.x_keys = []
        for k in self.all_vehicles:
            valid_nodes = [i for i in self.V_nodes if self._is_capable(k, i)]
            for i in valid_nodes:
                if i == self.V_nodes[-1]: continue
                for j in valid_nodes:
                    if j == 0 or i == j: continue
                    self.x_keys.append((i, j, k))
        self.idx_map['x'] = (counter, counter + len(self.x_keys))
        counter += len(self.x_keys)
        
        # 2. z_{ik}
        self.z_keys = []
        for k in self.all_vehicles:
            for i in self.real_nodes:
                if self._is_capable(k, i):
                    self.z_keys.append((i, k))
        self.idx_map['z'] = (counter, counter + len(self.z_keys))
        counter += len(self.z_keys)
        
        # 3. t_i
        self.idx_map['t'] = (counter, counter + len(self.real_nodes))
        counter += len(self.real_nodes)
        
        # 4. w_i
        self.idx_map['w'] = (counter, counter + len(self.real_nodes))
        counter += len(self.real_nodes)
        
        # 5. r_{ik}
        self.r_keys = []
        for k in self.all_vehicles:
            for i in self.V_nodes:
                if self._is_capable(k, i):
                    self.r_keys.append((i, k))
        self.idx_map['r'] = (counter, counter + len(self.r_keys))
        counter += len(self.r_keys)
        
        # 6. v_i (仅针对 arr 节点)
        self.arr_nodes = [i for i, info in self.nodes.items() if 'arr' in info['type']]
        self.idx_map['v'] = (counter, counter + len(self.arr_nodes))
        counter += len(self.arr_nodes)
        
        # 7. y_n
        self.idx_map['y'] = (counter, counter + len(self.flights))
        counter += len(self.flights)
        
        self.num_vars = counter
        
        # Maps for quick indexing
        self.x_map = {k: i + self.idx_map['x'][0] for i, k in enumerate(self.x_keys)}
        self.z_map = {k: i + self.idx_map['z'][0] for i, k in enumerate(self.z_keys)}
        self.t_map = {k: i + self.idx_map['t'][0] for i, k in enumerate(self.real_nodes)}
        self.w_map = {k: i + self.idx_map['w'][0] for i, k in enumerate(self.real_nodes)}
        self.r_map = {k: i + self.idx_map['r'][0] for i, k in enumerate(self.r_keys)}
        self.v_map = {k: i + self.idx_map['v'][0] for i, k in enumerate(self.arr_nodes)}
        self.y_map = {k: i + self.idx_map['y'][0] for i, k in enumerate(self.flights)}

    def _build_static_matrices(self):
        rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
        rows_in, cols_in, data_in, b_in = [], [], [], []
        
        eq_ptr, in_ptr = 0, 0
        
        # ==================================
        # 等式约束 A_eq * x = b_eq
        # ==================================
        # Eq 1: Coverage
        for i in self.real_nodes:
            edges = [e for e in self.x_keys if e[1] == i]
            for e in edges:
                rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(1.0)
            b_eq.append(1.0)
            eq_ptr += 1
            
        # Eq 2: Start/End
        source, sink = 0, self.V_nodes[-1]
        for k in self.all_vehicles:
            # Start
            edges_start = [e for e in self.x_keys if e[0] == source and e[2] == k]
            for e in edges_start:
                rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(1.0)
            b_eq.append(1.0)
            eq_ptr += 1
            # End
            edges_end = [e for e in self.x_keys if e[1] == sink and e[2] == k]
            for e in edges_end:
                rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(1.0)
            b_eq.append(1.0)
            eq_ptr += 1
            
        # Eq 3: Flow
        for k in self.all_vehicles:
            for h in self.real_nodes:
                if not self._is_capable(k, h): continue
                in_edges = [e for e in self.x_keys if e[1] == h and e[2] == k]
                out_edges = [e for e in self.x_keys if e[0] == h and e[2] == k]
                for e in in_edges:
                    rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(1.0)
                for e in out_edges:
                    rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(-1.0)
                b_eq.append(0.0)
                eq_ptr += 1

        # Eq 4: z = sum(x)
        for k in self.all_vehicles:
            for i in self.real_nodes:
                if not self._is_capable(k, i): continue
                rows_eq.append(eq_ptr); cols_eq.append(self.z_map[(i, k)]); data_eq.append(1.0)
                in_edges = [e for e in self.x_keys if e[1] == i and e[2] == k]
                for e in in_edges:
                    rows_eq.append(eq_ptr); cols_eq.append(self.x_map[e]); data_eq.append(-1.0)
                b_eq.append(0.0)
                eq_ptr += 1

        # ==================================
        # 不等式约束 A_in * x <= b_in
        # ==================================
        # Eq 7: Time Prop (t_i - r_jk + M*x <= M - s_i - tau)
        for (i, j, k) in self.x_keys:
            if i == 0 or j == sink: continue
            rows_in.append(in_ptr); cols_in.append(self.t_map[i]); data_in.append(1.0)
            rows_in.append(in_ptr); cols_in.append(self.r_map[(j, k)]); data_in.append(-1.0)
            rows_in.append(in_ptr); cols_in.append(self.x_map[(i, j, k)]); data_in.append(self.M)
            
            s_i = self.instance['duration'][i]
            tau_ij = self.tau.get((i, j), 15)
            b_in.append(self.M - s_i - tau_ij)
            in_ptr += 1

        # Eq 9: Task Start (r_ik - t_i + M*z <= M)
        for k in self.all_vehicles:
            for i in self.real_nodes:
                if not self._is_capable(k, i): continue
                rows_in.append(in_ptr); cols_in.append(self.r_map[(i, k)]); data_in.append(1.0)
                rows_in.append(in_ptr); cols_in.append(self.t_map[i]); data_in.append(-1.0)
                rows_in.append(in_ptr); cols_in.append(self.z_map[(i, k)]); data_in.append(self.M)
                b_in.append(self.M)
                in_ptr += 1
                
        # Eq 11: Veh Wait (t_i - r_ik - w_i + M*z <= M)
        for k in self.all_vehicles:
            for i in self.real_nodes:
                if not self._is_capable(k, i): continue
                rows_in.append(in_ptr); cols_in.append(self.t_map[i]); data_in.append(1.0)
                rows_in.append(in_ptr); cols_in.append(self.r_map[(i, k)]); data_in.append(-1.0)
                rows_in.append(in_ptr); cols_in.append(self.w_map[i]); data_in.append(-1.0)
                rows_in.append(in_ptr); cols_in.append(self.z_map[(i, k)]); data_in.append(self.M)
                b_in.append(self.M)
                in_ptr += 1

        # Eq 15 & 16: Task Chain (t_u - t_v <= -s_u)
        task_map = self.instance['task_map']
        for fn, tasks in task_map.items():
            u_arr = tasks.get('arr')
            v_dep = tasks.get('dep')
            h_push = tasks.get('push')
            
            if u_arr and v_dep:
                rows_in.append(in_ptr); cols_in.append(self.t_map[u_arr]); data_in.append(1.0)
                rows_in.append(in_ptr); cols_in.append(self.t_map[v_dep]); data_in.append(-1.0)
                b_in.append(-self.instance['duration'][u_arr])
                in_ptr += 1
                
            if v_dep and h_push:
                rows_in.append(in_ptr); cols_in.append(self.t_map[v_dep]); data_in.append(1.0)
                rows_in.append(in_ptr); cols_in.append(self.t_map[h_push]); data_in.append(-1.0)
                b_in.append(-self.instance['duration'][v_dep])
                in_ptr += 1

        # Eq 18: Delay (t_push - y_n <= d_n^0 - s_push)
        for fn, tasks in task_map.items():
            if 'push' in tasks:
                h = tasks['push']
                s_h = self.instance['duration'][h]
                d = self.d_plan[fn]
                rows_in.append(in_ptr); cols_in.append(self.t_map[h]); data_in.append(1.0)
                rows_in.append(in_ptr); cols_in.append(self.y_map[fn]); data_in.append(-1.0)
                b_in.append(d - s_h)
                in_ptr += 1

        # Bounds Constraints (x>=0, z>=0, ... => -xi <= 0)
        for i in range(self.num_vars):
            rows_in.append(in_ptr); cols_in.append(i); data_in.append(-1.0); b_in.append(0.0)
            in_ptr += 1
            
        # Upper bounds for relax variables (x<=1, z<=1 => xi <= 1)
        for name in ['x', 'z']:
            start, end = self.idx_map[name]
            for i in range(start, end):
                rows_in.append(in_ptr); cols_in.append(i); data_in.append(1.0); b_in.append(1.0)
                in_ptr += 1

        self.A_eq = sp.csr_matrix((data_eq, (rows_eq, cols_eq)), shape=(eq_ptr, self.num_vars))
        self.b_eq = np.array(b_eq)
        self.A_in_static = sp.csr_matrix((data_in, (rows_in, cols_in)), shape=(in_ptr, self.num_vars))
        self.b_in_static = np.array(b_in)

    def get_full_constraints(self, pred_a_dict):
        """拼接动态预测参数 a_n 对应的约束"""
        rows, cols, data = [], [], []
        b = []
        
        static_rows, static_cols = self.A_in_static.nonzero()
        static_data = self.A_in_static.data
        rows.extend(static_rows); cols.extend(static_cols); data.extend(static_data)
        b.extend(self.b_in_static)
        
        current_row = self.A_in_static.shape[0]
        self.a_sensitivity_map = {} 
        
        task_map = self.instance['task_map']
        for fn, tasks in task_map.items():
            arr_time = pred_a_dict[fn]
            if 'arr' in tasks:
                i = tasks['arr']
                
                # Eq 13: t_i - v_i <= a_n
                rows.append(current_row); cols.append(self.t_map[i]); data.append(1.0)
                rows.append(current_row); cols.append(self.v_map[i]); data.append(-1.0)
                b.append(arr_time)
                self.a_sensitivity_map[current_row] = (fn, 1.0) # a_n 的系数是 1.0
                current_row += 1
                
                # Eq 14: -t_i <= -a_n
                rows.append(current_row); cols.append(self.t_map[i]); data.append(-1.0)
                b.append(-arr_time)
                self.a_sensitivity_map[current_row] = (fn, -1.0) # a_n 的系数是 -1.0
                current_row += 1

        A_in = sp.csr_matrix((data, (rows, cols)), shape=(current_row, self.num_vars))
        b_in = np.array(b)
        return A_in, b_in

    def build_c_vector(self, weights):
        """与目标函数完全对应"""
        c = np.zeros(self.num_vars)
        for fn, idx in self.y_map.items(): c[idx] = weights['pi']
        for i, idx in self.w_map.items(): c[idx] = weights['alpha']
        for i, idx in self.v_map.items(): c[idx] = weights['beta']
        return c

class SPO_Function(Function):
    @staticmethod
    def forward(ctx, pred_a_tensor, instance_builders, weights, mu):
        ctx.builders = instance_builders
        ctx.mu = mu
        device = pred_a_tensor.device
        
        batch_size = pred_a_tensor.shape[0]
        optimal_xi_list = []
        ctx.solutions = [] 
        
        pred_a_numpy = pred_a_tensor.detach().cpu().numpy()
        
        for b in range(batch_size):
            builder = instance_builders[b]
            pred_a_row = pred_a_numpy[b]
            
            flight_list = list(builder.flights)
            pred_a_dict = {fn: pred_a_row[i] for i, fn in enumerate(flight_list)}
            
            A_eq, b_eq = builder.A_eq, builder.b_eq
            A_in, b_in = builder.get_full_constraints(pred_a_dict)
            c = builder.build_c_vector(weights)
            
            x_var = cp.Variable(builder.num_vars)
            objective = cp.Minimize(c @ x_var - mu * cp.sum(cp.log(b_in - A_in @ x_var)))
            constraints = [A_eq @ x_var == b_eq]
            prob = cp.Problem(objective, constraints)
            
            solve_success = False
            try:
                prob.solve(solver=cp.ECOS, verbose=False, max_iters=200)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x_var.value is not None:
                    xi_val = x_var.value
                    solve_success = True
                else:
                    xi_val = np.zeros(builder.num_vars)
            except Exception:
                xi_val = np.zeros(builder.num_vars)
            
            optimal_xi_list.append(torch.tensor(xi_val, dtype=torch.float32, device=device))
            
            slacks = np.maximum(b_in - A_in @ xi_val, 1e-6) if solve_success else np.ones(b_in.shape)

            ctx.solutions.append({
                'xi': xi_val,
                'slacks': slacks,
                'A_in': A_in,
                'A_eq': A_eq,
                'a_map': builder.a_sensitivity_map,
                'flights': flight_list,
                'success': solve_success
            })
            
        return torch.stack(optimal_xi_list)

    @staticmethod
    def backward(ctx, grad_output):
        mu = ctx.mu
        device = grad_output.device
        final_grads = []
        
        for b, sol in enumerate(ctx.solutions):
            if not sol['success']:
                final_grads.append(torch.zeros(len(sol['flights']), dtype=torch.float32, device=device))
                continue

            A_eq, A_in, S = sol['A_eq'], sol['A_in'], sol['slacks']
            
            try:
                # [核心修正] 针对 M=10000 的自适应屏蔽 (Active Constraint Masking)
                # 只有 slack < 1000 的约束被认为是“真正起作用的物理约束”
                mask = (S < 1000.0).astype(float)  
                
                # 被屏蔽的约束对角线元素强行置零，彻底消除大M带来的 10^-7 级噪声
                D_diag = mask * (mu / (S**2 + 1e-6))
                D = sp.diags(D_diag)
                
                H = A_in.T @ D @ A_in
                
                num_vars, num_eq = H.shape[0], A_eq.shape[0]
                
                # 构建屏蔽大 M 后的精简且良态(well-conditioned)的 KKT 系统
                KKT = sp.vstack([
                    sp.hstack([H, A_eq.T]),
                    sp.hstack([A_eq, sp.csr_matrix((num_eq, num_eq))])
                ]) + sp.eye(num_vars + num_eq) * 1e-6 
                
                rhs_vec = np.concatenate([grad_output[b].detach().cpu().numpy(), np.zeros(num_eq)])
                adjoint_sol = spsolve(KKT, rhs_vec)
                lambda_x = adjoint_sol[:num_vars]
                
                # 梯度的传导也只通过真实的有效约束
                temp = (A_in @ lambda_x) * D_diag
                
                flight_to_idx = {fn: i for i, fn in enumerate(sol['flights'])}
                grad_a_row = np.zeros(len(sol['flights']))
                
                for row_idx, (fn, val) in sol['a_map'].items():
                    grad_a_row[flight_to_idx[fn]] += temp[row_idx] * val
                    
                final_grads.append(torch.tensor(grad_a_row, dtype=torch.float32, device=device))
            
            except Exception:
                final_grads.append(torch.zeros(len(sol['flights']), dtype=torch.float32, device=device))

        return torch.stack(final_grads), None, None, None

class SPO_Layer(nn.Module):
    def __init__(self, weights={'pi': 100, 'alpha': 2, 'beta': 10}, mu=10.0):
        super(SPO_Layer, self).__init__()
        self.weights = weights
        self.mu = mu
        
    def forward(self, pred_a, instance_raw_data, fleet_config):
        # [核心修复] 将传进来的字典(Instance)列表，转化为 GSE_LP_Builder 对象列表
        builders = [GSE_LP_Builder(inst, fleet_config) for inst in instance_raw_data]
        return SPO_Function.apply(pred_a, builders, self.weights, self.mu)

