# import gurobipy as gp
# from gurobipy import GRB
# import random

# class GSESolver:
#     def __init__(self, config):
#         self.pi = config.get('weight_delay', 100)    # 航班离港延误惩罚
#         self.beta = config.get('weight_resp', 50)    # 飞机等车惩罚
#         self.alpha = config.get('weight_wait', 1)    # 车等飞机惩罚
#         self.M = config.get('big_M', 1440) 
#         self.vehicles = config.get('fleets', config.get('vehicles', {}))
#         self.all_vehicles = list(self.vehicles.keys())

#     def solve(self, instance, predicted_a, relax=False, fixed_x=None):
#         model = gp.Model('GSE_Scheduling_Pruned')
        
#         model.setParam('OutputFlag', 1)
#         model.setParam('Threads', 4)
#         if not relax:
#             # model.setParam('TimeLimit', 60)
#             # model.setParam('MIPGap', 0.05)
#             # model.setParam('MIPFocus', 1)
#             model.setParam('Heuristics', 0.10)
#         flights = instance['flights']       
#         nodes = instance['nodes']           
#         task_map = instance['task_map']     
#         tau = instance.get('travel_times', {}) 
#         durations = instance['duration']    
#         tau_time = 15
#         N = len(nodes)
#         source_node = 0
#         sink_node = N + 1
#         V_nodes = [source_node] + list(nodes.keys()) + [sink_node]

#         def is_capable(v_id, n_id):
#             if n_id == source_node or n_id == sink_node: return True
#             return any(t in self.vehicles[v_id]['task_type'] for t in nodes[n_id]['type'])

#         # ==========================================
#         # 0. 预处理：构建节点映射与计算 EST (最早开始时间)
#         # ==========================================
#         node_to_flight = {}
#         node_to_task = {}
#         for fn, tmap in task_map.items():
#             for t_name, n_id in tmap.items():
#                 node_to_flight[n_id] = fn
#                 node_to_task[n_id] = t_name

#         EST = {source_node: 0, sink_node: 0}
#         for fn in flights:
#             arr_n = task_map[fn].get('arr')
#             dep_n = task_map[fn].get('dep')
#             push_n = task_map[fn].get('push')
            
#             ata = predicted_a[fn]
#             if arr_n: EST[arr_n] = ata
#             if dep_n: EST[dep_n] = ata + durations.get(arr_n, 0)
#             if push_n: EST[push_n] = ata + durations.get(arr_n, 0) + durations.get(dep_n, 0)

#         # 核心剪枝函数
#         def is_edge_valid(i, j):
#             if i == j or j == source_node or i == sink_node: 
#                 return False
                
#             # 剪枝 1：同航班的逆序逻辑不可能
#             # 任务顺序必须是: arr -> dep -> push
#             f_i = node_to_flight.get(i)
#             f_j = node_to_flight.get(j)
#             if f_i and f_j and f_i == f_j:
#                 order = {'arr': 0, 'dep': 1, 'push': 2}
#                 if order[node_to_task[i]] >= order[node_to_task[j]]:
#                     return False
            
#             # 剪枝 2：时间窗强迫延误剪枝
#             # 如果先做 i 再做 j，导致 j 的理论最早开始时间被推迟了超过 180 分钟 (3小时)
#             # 这种极其颠倒时间顺序的边绝对不可能是最优解
#             if i != source_node and j != sink_node:
#                 travel_t = tau.get((i, j), tau_time)
#                 forced_j_start = EST[i] + durations[i] + travel_t
#                 forced_delay = forced_j_start - EST[j]
#                 if forced_delay > 180:
#                     return False
            
#             return True

#         # ==========================================
#         # 1. 定义决策变量 (加入剪枝逻辑)
#         # ==========================================
#         vtype_bin = GRB.CONTINUOUS if relax else GRB.BINARY
        
#         x, z = {}, {}
#         edge_count = 0  # 统计最终生成的有效边数量
        
#         for k in self.all_vehicles:
#             valid_nodes = [i for i in V_nodes if is_capable(k, i)]
#             for i in valid_nodes:
#                 if i != source_node and i != sink_node:
#                     z[i, k] = model.addVar(lb=0, ub=1, vtype=vtype_bin, name=f"z_{i}_{k}")
                
#                 if i == sink_node: continue
#                 for j in valid_nodes:
#                     # [关键动作] 应用剪枝
#                     if not is_edge_valid(i, j): 
#                         continue
                    
#                     lb_val, ub_val = 0.0, 1.0
#                     if fixed_x is not None and (i, j, k) in fixed_x:
#                         val = round(fixed_x[i, j, k])
#                         lb_val, ub_val = val, val
                        
#                     x[i, j, k] = model.addVar(lb=lb_val, ub=ub_val, vtype=vtype_bin, name=f"x_{i}_{j}_{k}")
#                     edge_count += 1

#         t, w, r = {}, {}, {}
#         for i in nodes:
#             t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{i}")
#             w[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}")

#         for k in self.all_vehicles:
#             for i in V_nodes:
#                 if is_capable(k, i):
#                     r[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_{i}_{k}")
        
#         v_delay = {}
#         arr_nodes = [i for i, info in nodes.items() if 'arr' in info['type']]
#         for i in arr_nodes:
#             v_delay[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_{i}")

#         y = {}
#         for fn in flights:
#             y[fn] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{fn}")

#         # ==========================================
#         # 2. 约束条件 (依赖于剪枝后的 x)
#         # ==========================================
#         for i in nodes:
#             # 必须有一个车流入 i
#             model.addConstr(gp.quicksum(x[j, i, k] for k in self.all_vehicles for j in V_nodes if (j, i, k) in x) == 1)

#         for k in self.all_vehicles:
#             # 源点流出与汇点流入
#             model.addConstr(gp.quicksum(x[source_node, j, k] for j in nodes if (source_node, j, k) in x) == 1)
#             model.addConstr(gp.quicksum(x[i, sink_node, k] for i in nodes if (i, sink_node, k) in x) == 1)

#             for h in nodes:
#                 if not is_capable(k, h): continue
#                 flow_in = gp.quicksum(x[i, h, k] for i in V_nodes if (i, h, k) in x)
#                 flow_out = gp.quicksum(x[h, j, k] for j in V_nodes if (h, j, k) in x)
#                 model.addConstr(flow_in == flow_out)
#                 model.addConstr(z[h, k] == flow_in)

#         for k in self.all_vehicles:
#             for i in V_nodes:
#                 if i == sink_node or not is_capable(k, i): continue
#                 for j in V_nodes:
#                     if (i, j, k) not in x: continue
#                     travel_time = tau.get((i, j), tau_time)
#                     duration_i = durations[i] if i != source_node else 0
#                     start_i = t[i] if i != source_node else 0
#                     model.addConstr(r[j, k] >= start_i + duration_i + travel_time - self.M * (1 - x[i, j, k]))

#         for i in nodes:
#             for k in self.all_vehicles:
#                 if not is_capable(k, i): continue
#                 model.addConstr(t[i] >= r[i, k] - self.M * (1 - z[i, k]))
#                 model.addConstr(w[i] >= t[i] - r[i, k] - self.M * (1 - z[i, k]))

#         for fn in flights:
#             if 'arr' in task_map[fn]:
#                 node_arr = task_map[fn]['arr']
#                 pred_arrival = predicted_a[fn]
#                 model.addConstr(v_delay[node_arr] >= t[node_arr] - pred_arrival)
#                 model.addConstr(t[node_arr] >= pred_arrival)

#         for fn in flights:
#             node_arr = task_map[fn].get('arr')
#             node_dep = task_map[fn].get('dep')
#             node_push = task_map[fn].get('push')

#             if node_arr and node_dep:
#                 model.addConstr(t[node_dep] >= t[node_arr] + durations[node_arr])
#             if node_dep and node_push:
#                 model.addConstr(t[node_push] >= t[node_dep] + durations[node_dep])
#             if node_push:
#                 plan_d = instance.get('std', {}).get(fn) 
#                 if plan_d is None:
#                     plan_d = predicted_a[fn] + durations[node_arr] + durations[node_dep] + durations[node_push]
#                 model.addConstr(y[fn] >= t[node_push] + durations[node_push] - plan_d)
#                 model.addConstr(t[node_push] + durations[node_push] >= plan_d - 5, name=f"earliest_dep_{fn}")

#         baggage_vehicles = [k for k in self.all_vehicles if self.vehicles[k]['type'] == 'baggage']
#         for v_idx in range(len(baggage_vehicles) - 1):
#             v_curr = baggage_vehicles[v_idx]
#             v_next = baggage_vehicles[v_idx + 1]
#             tasks_curr = gp.quicksum(z[i, v_curr] for i in V_nodes if (i, v_curr) in z)
#             tasks_next = gp.quicksum(z[i, v_next] for i in V_nodes if (i, v_next) in z)
#             model.addConstr(tasks_curr >= tasks_next, name=f"sym_break_baggage_{v_curr}_{v_next}")

#         # 2. 破除牵引车 (Pushback) 的对称性
#         pushback_vehicles = [k for k in self.all_vehicles if self.vehicles[k]['type'] == 'pushback']
#         for v_idx in range(len(pushback_vehicles) - 1):
#             v_curr = pushback_vehicles[v_idx]
#             v_next = pushback_vehicles[v_idx + 1]
#             tasks_curr = gp.quicksum(z[i, v_curr] for i in V_nodes if (i, v_curr) in z)
#             tasks_next = gp.quicksum(z[i, v_next] for i in V_nodes if (i, v_next) in z)
#             model.addConstr(tasks_curr >= tasks_next, name=f"sym_break_push_{v_curr}_{v_next}")

#         # objective function
#         obj_delay = gp.quicksum(self.pi * y[fn] for fn in flights)
#         obj_wait = gp.quicksum(self.alpha * w[i] for i in nodes)
#         obj_resp = gp.quicksum(self.beta * v_delay[i] for i in arr_nodes)
        
#         model.setObjective(obj_delay + obj_wait + obj_resp, GRB.MINIMIZE)
#         model.optimize()

#         if model.SolCount > 0:
#             val_delay = sum(y[fn].X for fn in flights)
#             val_wait = sum(w[i].X for i in nodes)
#             val_resp = sum(v_delay[i].X for i in arr_nodes)
            
#             t_values = {i: t[i].X for i in nodes}
#             z_values = {(i, k): z[i, k].X for (i, k) in z if z[i, k].X > 0.5}
            
#             return model.objVal, {
#                 'delay_cost': val_delay * self.pi,
#                 'wait_cost': val_wait * self.alpha,
#                 'resp_cost': val_resp * self.beta,
#                 'x_values': {(i, j, k): x[i, j, k].X for (i, j, k) in x},
#                 'z_values': z_values,
#                 't_values': t_values
#             }
#         else:
#             return float('inf'), {}


import gurobipy as gp
from gurobipy import GRB

class GSESolver:
    def __init__(self, config):
        self.pi = config.get('weight_delay', 100)    
        self.beta = config.get('weight_resp', 50)    
        self.alpha = config.get('weight_wait', 1)    
        self.M = config.get('big_M', 1440) 
        self.vehicles = config.get('fleets', config.get('vehicles', {}))
        self.all_vehicles = list(self.vehicles.keys())

    def solve(self, instance, predicted_a, relax=False, fixed_x=None):
        model = gp.Model('GSE_Scheduling_NoPush')
        model.setParam('OutputFlag', 1)
        model.setParam('Threads', 8)
        
        if not relax:
            # model.setParam('TimeLimit', 60)
            # model.setParam('MIPGap', 0.05)
            # model.setParam('MIPFocus', 1)
            model.setParam('Heuristics', 0.15)

        flights = instance['flights']       
        nodes = instance['nodes']           
        task_map = instance['task_map']     
        durations = instance['duration']    
        tau_time = 5  # 固定物理行驶时间
        
        N = len(nodes)
        source_node, sink_node = 0, N + 1
        V_nodes = [source_node] + list(nodes.keys()) + [sink_node]

        def is_capable(v_id, n_id):
            if n_id in (source_node, sink_node): return True
            return any(t in self.vehicles[v_id]['task_type'] for t in nodes[n_id]['type'])

        node_to_flight, node_to_task = {}, {}
        for fn, tmap in task_map.items():
            for t_name, n_id in tmap.items():
                node_to_flight[n_id] = fn
                node_to_task[n_id] = t_name

        EST = {source_node: 0, sink_node: 0}
        max_possible_time = max(EST.values()) + 180 # 假设最晚的任务不会超过最后到达的飞机+180分钟
        tight_M = max_possible_time

        for fn in flights:
            arr_n = task_map[fn].get('arr')
            dep_n = task_map[fn].get('dep')
            ata = predicted_a[fn]
            if arr_n: EST[arr_n] = ata
            if dep_n: EST[dep_n] = ata + durations.get(arr_n, 0)

        def is_edge_valid(i, j):
            if i == j or j == source_node or i == sink_node: return False
            f_i, f_j = node_to_flight.get(i), node_to_flight.get(j)
            if f_i and f_j and f_i == f_j:
                order = {'arr': 0, 'dep': 1}
                if order[node_to_task[i]] >= order[node_to_task[j]]: return False
            
            if i != source_node and j != sink_node:
                if EST[i] + durations[i] + tau_time - EST[j] > 60: return False
            return True

        vtype_bin = GRB.CONTINUOUS if relax else GRB.BINARY
        x, z = {}, {}
        for k in self.all_vehicles:
            valid_nodes = [i for i in V_nodes if is_capable(k, i)]
            for i in valid_nodes:
                if i not in (source_node, sink_node):
                    z[i, k] = model.addVar(lb=0, ub=1, vtype=vtype_bin)
                if i == sink_node: continue
                for j in valid_nodes:
                    if not is_edge_valid(i, j): continue
                    lb_val = ub_val = round(fixed_x[i, j, k]) if fixed_x and (i, j, k) in fixed_x else (0.0, 1.0)[1] if fixed_x is None else 0.0
                    x[i, j, k] = model.addVar(lb=lb_val if fixed_x else 0, ub=ub_val if fixed_x else 1, vtype=vtype_bin)

        t={}
        for i in nodes:
            t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=EST.get(i, 0), name=f"t_{i}")
        w = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0) for i in nodes}
        r = {(i, k): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for k in self.all_vehicles for i in V_nodes if is_capable(k, i)}
        
        arr_nodes = [i for i, info in nodes.items() if 'arr' in info['type']]
        v_delay = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0) for i in arr_nodes}
        y = {fn: model.addVar(vtype=GRB.CONTINUOUS, lb=0) for fn in flights}

        for i in nodes:
            model.addConstr(gp.quicksum(x[j, i, k] for k in self.all_vehicles for j in V_nodes if (j, i, k) in x) == 1)

        for k in self.all_vehicles:
            model.addConstr(gp.quicksum(x[source_node, j, k] for j in nodes if (source_node, j, k) in x) == 1)
            model.addConstr(gp.quicksum(x[i, sink_node, k] for i in nodes if (i, sink_node, k) in x) == 1)
            for h in nodes:
                if not is_capable(k, h): continue
                flow_in = gp.quicksum(x[i, h, k] for i in V_nodes if (i, h, k) in x)
                flow_out = gp.quicksum(x[h, j, k] for j in V_nodes if (h, j, k) in x)
                model.addConstr(flow_in == flow_out)
                model.addConstr(z[h, k] == flow_in)

            for i in V_nodes:
                if i == sink_node or not is_capable(k, i): continue
                for j in V_nodes:
                    if (i, j, k) not in x: continue
                    dur_i = durations[i] if i != source_node else 0
                    start_i = t[i] if i != source_node else 0
                    model.addConstr(r[j, k] >= start_i + dur_i + tau_time - tight_M * (1 - x[i, j, k]))

        for i in nodes:
            for k in self.all_vehicles:
                if not is_capable(k, i): continue
                model.addConstr(t[i] >= r[i, k] - tight_M * (1 - z[i, k]))
                model.addConstr(w[i] >= t[i] - r[i, k] - tight_M * (1 - z[i, k]))

        for fn in flights:
            node_arr = task_map[fn].get('arr')
            pred_arrival = predicted_a[fn]
            model.addConstr(v_delay[node_arr] >= t[node_arr] - pred_arrival)
            model.addConstr(t[node_arr] >= pred_arrival)

            node_dep = task_map[fn].get('dep')
            if node_arr and node_dep:
                model.addConstr(t[node_dep] >= t[node_arr] + durations[node_arr])
            if node_dep:
                plan_d = instance.get('std', {}).get(fn) or (pred_arrival + 60)
                # 装货结束必须在 STD - 5 之前
                model.addConstr(y[fn] >= t[node_dep] + durations[node_dep] - (plan_d - 5))

        # 行李车的对称性破除
        baggage_vs = [k for k in self.all_vehicles if self.vehicles[k]['type'] == 'baggage']
        for v_idx in range(len(baggage_vs) - 1):
            tasks_curr = gp.quicksum(z[i, baggage_vs[v_idx]] for i in V_nodes if (i, baggage_vs[v_idx]) in z)
            tasks_next = gp.quicksum(z[i, baggage_vs[v_idx+1]] for i in V_nodes if (i, baggage_vs[v_idx+1]) in z)
            model.addConstr(tasks_curr >= tasks_next)

        obj_delay = gp.quicksum(self.pi * y[fn] for fn in flights)
        obj_wait = gp.quicksum(self.alpha * w[i] for i in nodes)
        obj_resp = gp.quicksum(self.beta * v_delay[i] for i in arr_nodes)
        
        model.setObjective(obj_delay + obj_wait + obj_resp, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount > 0:
            return model.objVal, {
                'delay_cost': sum(y[fn].X for fn in flights) * self.pi,
                'wait_cost': sum(w[i].X for i in nodes) * self.alpha,
                'resp_cost': sum(v_delay[i].X for i in arr_nodes) * self.beta,
                'x_values': {(i, j, k): x[i, j, k].X for (i, j, k) in x},
                'z_values': {(i, k): z[i, k].X for (i, k) in z if z[i, k].X > 0.5},
                't_values': {i: t[i].X for i in nodes}
            }
        return float('inf'), {}
