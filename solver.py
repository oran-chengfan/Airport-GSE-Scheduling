import gurobipy as gp
from gurobipy import GRB

class GSESolver:
    def __init__(self, config):
        self.pi = config.get('weight_delay', 100)    
        # 最终的起飞延误 y_n
        self.num_vehicles = len(config.get('fleets', config.get('vehicles', {})))

    def build_model(self, instance, predicted_a, relax=False, fixed_x=None):
        import gurobipy as gp
        from gurobipy import GRB
        
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model('VRPTW_Task_Level', env=env)
        model.setParam('Threads', 1)
        model.setParam("MIPGap", 0.05)

        flights = sorted(instance['flights'])
        task_map = instance['task_map']
        durations = instance['duration']
        
        tasks = []
        EST = {'SOURCE': 0, 'SINK': 0}
        
        for fn in flights:
            arr_n = task_map[fn].get('arr')
            dep_n = task_map[fn].get('dep')
            std_time = instance.get('std', {}).get(fn, 1000)
            if arr_n: 
                tasks.append(arr_n)
                EST[arr_n] = std_time - 60
            if dep_n: 
                tasks.append(dep_n)
                EST[dep_n] = std_time - 60
            
        source_node, sink_node = 'SOURCE', 'SINK'
        V_nodes = [source_node] + tasks + [sink_node]
        tau_time = 10 
        max_possible_time = 3000
        def is_edge_valid(i, j):
            if i == j or j == source_node or i == sink_node: return False
            if i != source_node and j != sink_node:
                # 如果车辆做完 i，赶到 j 的时间，比 j 的计划时间晚了 4 个小时以上 (240分钟)
                # 这种连边在现实中绝不可能发生，直接剪枝！
                if EST.get(i, 0) + durations.get(i, 0) + tau_time > EST.get(j, 0) + 240: 
                    return False
            return True

        vtype_bin = GRB.CONTINUOUS if relax else GRB.BINARY
        x = {}
        for i in V_nodes:
            if i == sink_node: continue
            for j in V_nodes:
                if not is_edge_valid(i, j): continue
                lb_val = round(fixed_x[i, j]) if fixed_x and (i, j) in fixed_x else 0.0
                ub_val = round(fixed_x[i, j]) if fixed_x and (i, j) in fixed_x else 1.0
                x[i, j] = model.addVar(lb=lb_val if fixed_x else 0, ub=ub_val if fixed_x else 1, vtype=vtype_bin, name=f"x_{i}_{j}")

        t = {n: model.addVar(lb=0, ub=max_possible_time, name=f"t_{n}") for n in tasks}
        s = {n: model.addVar(lb=0, ub=max_possible_time, name=f"s_{n}") for n in tasks}
        veh_wait = {n: model.addVar(lb=0, ub=max_possible_time, name=f"veh_wait_{n}") for n in tasks}
        
        plane_wait = {f: model.addVar(lb=0, ub=max_possible_time, name=f"plane_wait_{f}") for f in flights}
        y = {f: model.addVar(lb=0, ub=max_possible_time, name=f"y_{f}") for f in flights}

        for n in tasks:
            model.addConstr(gp.quicksum(x[j, n] for j in V_nodes if (j, n) in x) == 1)
            model.addConstr(gp.quicksum(x[n, j] for j in V_nodes if (n, j) in x) == 1)

        model.addConstr(gp.quicksum(x[source_node, j] for j in tasks if (source_node, j) in x) <= self.num_vehicles)
        model.addConstr(gp.quicksum(x[source_node, j] for j in tasks if (source_node, j) in x) == 
                        gp.quicksum(x[i, sink_node] for i in tasks if (i, sink_node) in x))

        for i in V_nodes:
            if i == sink_node: continue
            for j in V_nodes:
                if (i, j) not in x or j == source_node or j == sink_node: continue
                p_i = durations.get(i, 0) if i != source_node else 0
                t_i = t[i] if i != source_node else 0
                s_i = s[i] if i != source_node else 0
                
                # 【快刀 2：抛弃 Big-M，使用 Gurobi 原生指示器约束】
                # 这在数学上等价，但能让求解器速度起飞！
                model.addConstr((x[i, j] == 1) >> (t[j] >= t_i + p_i + tau_time))
                model.addConstr((x[i, j] == 1) >> (s[j] >= s_i + p_i + tau_time))

        # ---- 下面的物理约束与之前完全一样，原封不动 ----
        for fn in flights:
            arr_n = task_map[fn]['arr']
            dep_n = task_map[fn]['dep']
            ata = predicted_a[fn]
            std = instance.get('std', {}).get(fn, 1000)
            ddl = std - 5
            
            model.addConstr(s[arr_n] >= ata, name=f"pred_ata_svc_{fn}")
            model.addConstr(s[arr_n] >= t[arr_n], name=f"svc_after_veh_{arr_n}")
            model.addConstr(s[dep_n] >= t[dep_n], name=f"svc_after_veh_{dep_n}")
            model.addConstr(s[dep_n] >= s[arr_n] + durations[arr_n], name=f"prec_{fn}")
            
            model.addConstr(veh_wait[arr_n] >= s[arr_n] - t[arr_n], name=f"def_veh_wait_{arr_n}")
            model.addConstr(veh_wait[dep_n] >= s[dep_n] - t[dep_n], name=f"def_veh_wait_{dep_n}")
            
            model.addConstr(plane_wait[fn] >= s[arr_n] - ata, name=f"pred_ata_wait_{fn}")
            model.addConstr(y[fn] >= s[dep_n] + durations[dep_n] - ddl, name=f"def_delay_{fn}")

        pi_wt = self.pi
        alpha_wt = self.pi * 0.1
        beta_wt = self.pi * 0.05
        
        obj = gp.quicksum(pi_wt * y[fn] + beta_wt * plane_wait[fn] for fn in flights)
        obj += gp.quicksum(alpha_wt * veh_wait[n] for n in tasks)
        obj += gp.quicksum(0.001 * t[n] for n in tasks) 
        model.setObjective(obj, GRB.MINIMIZE)

        vars_dict = {'x': x, 't': t, 's': s, 'veh_wait': veh_wait, 'plane_wait': plane_wait, 'y': y}
        return model, vars_dict, V_nodes, source_node, sink_node, durations


    def build_reduced_model(self, instance, ata_dict, active_edges):
        """
        DFL 专用代理模型 (Reduced LP)
        必须保持与 build_model 完全同构的 Task-level 节点拆解！
        """
        import gurobipy as gp
        from gurobipy import GRB
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model('Reduced_LP', env=env)
        model.setParam('Threads', 1)

        flights = sorted(instance['flights'])
        task_map = instance['task_map']
        durations = instance['duration']
        
        # 1. 将航班拆解为 arr 和 dep 任务节点，与 MILP 保持一致
        tasks = []
        for fn in flights:
            arr_n = task_map[fn].get('arr')
            dep_n = task_map[fn].get('dep')
            if arr_n: tasks.append(arr_n)
            if dep_n: tasks.append(dep_n)
            
        tau_time = 10
        
        # 2. 时间变量 t, s, veh_wait 必须挂载在 tasks 上！(修复 KeyError 的核心)
        t = {n: model.addVar(lb=0, name=f"t_{n}") for n in tasks} 
        s = {n: model.addVar(lb=0, name=f"s_{n}") for n in tasks} 
        veh_wait = {n: model.addVar(lb=0, name=f"veh_wait_{n}") for n in tasks}
        
        # 飞机等待和最终延误是针对整体 flight 的
        plane_wait = {f: model.addVar(lb=0, name=f"plane_wait_{f}") for f in flights}
        y = {f: model.addVar(lb=0, name=f"y_{f}") for f in flights}
        
        # 3. 添加基于活跃连边的物理路由约束 (时间传导)
        for (i, j) in active_edges:
            if i == 'SOURCE' or j == 'SINK': continue 
            p_i = durations.get(i, 0)
            model.addConstr(t[j] >= t[i] + p_i + tau_time, name=f"route_t_{i}_{j}")
            model.addConstr(s[j] >= s[i] + p_i + tau_time, name=f"route_s_{i}_{j}")
            
        # 4. 添加节点内部的业务逻辑约束 (严格复刻 build_model)
        for fn in flights:
            arr_n = task_map[fn]['arr']
            dep_n = task_map[fn]['dep']
            ata = ata_dict[fn]
            std = instance.get('std', {}).get(fn, 1000)
            ddl = std - 5
            
            # 预测到达时间与服务开始时间的约束
            model.addConstr(s[arr_n] >= ata, name=f"pred_ata_svc_{fn}")
            
            # 物理因果：服务必须在车辆(t)到达后才能开始
            model.addConstr(s[arr_n] >= t[arr_n], name=f"svc_after_veh_{arr_n}")
            model.addConstr(s[dep_n] >= t[dep_n], name=f"svc_after_veh_{dep_n}")
            
            # 任务时序：装货必须在卸货完成之后
            model.addConstr(s[dep_n] >= s[arr_n] + durations[arr_n], name=f"prec_{fn}")
            
            # 计算各项等待与延误代价
            model.addConstr(veh_wait[arr_n] >= s[arr_n] - t[arr_n], name=f"def_veh_wait_{arr_n}")
            model.addConstr(veh_wait[dep_n] >= s[dep_n] - t[dep_n], name=f"def_veh_wait_{dep_n}")
            
            model.addConstr(plane_wait[fn] >= s[arr_n] - ata, name=f"pred_ata_wait_{fn}")
            model.addConstr(y[fn] >= s[dep_n] + durations[dep_n] - ddl, name=f"def_delay_{fn}")

        # 5. 目标函数设定
        pi_wt = self.pi
        alpha_wt = self.pi * 0.1
        beta_wt = self.pi * 0.05
        
        obj = gp.quicksum(pi_wt * y[fn] + beta_wt * plane_wait[fn] for fn in flights)
        obj += gp.quicksum(alpha_wt * veh_wait[n] for n in tasks)
        obj += gp.quicksum(0.001 * t[n] for n in tasks)
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        vars_dict = {'t': t, 's': s, 'veh_wait': veh_wait, 'plane_wait': plane_wait, 'y': y}
        return model, vars_dict


    def solve(self, instance, predicted_a, relax=False, fixed_x=None):
        model, vars_dict, V_nodes, source_node, sink_node, P = self.build_model(instance, predicted_a, relax, fixed_x)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            x, t, y = vars_dict['x'], vars_dict['t'], vars_dict['y']
            flights = instance['flights']
            
            x_values_mapped = {}
            active_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
            for i, j in active_edges:
                x_values_mapped[(i, j)] = 1.0

            return model.objVal, {
                'delay_cost': sum(y[fn].X for fn in flights) * self.pi,
                'x_values': x_values_mapped,
                't_values': {f: t[f].X for f in flights}
            }
        
        return float('inf'), {}



# import gurobipy as gp
# from gurobipy import GRB

# class GSESolver:
#     def __init__(self, config):
#         self.pi = config.get('weight_delay', 100)    
#         self.beta = config.get('weight_resp', 50)    
#         self.alpha = config.get('weight_wait', 1)    
#         self.M = config.get('big_M', 1440) 
#         self.vehicles = config.get('fleets', config.get('vehicles', {}))
#         self.all_vehicles = sorted([v for v, info in self.vehicles.items() if info['type'] == 'baggage'])
#         self.num_vehicles = len(self.all_vehicles)

#     def build_model(self, instance, predicted_a, relax=False, fixed_x=None):
#         env = gp.Env(empty=True)
#         env.setParam('OutputFlag', 0)
#         env.start()
#         model = gp.Model('GSE_Scheduling_AnonymousFlow', env=env)
#         # model.setParam('OutputFlag', 1 if not relax else 0)
#         model.setParam('OutputFlag', 0)
#         model.setParam('Threads', 1)
        
#         if not relax:
#             model.setParam('MIPGap', 0.03)
#             model.setParam('Heuristics', 0.15)

#         flights = instance['flights']       
#         nodes = instance['nodes']           
#         task_map = instance['task_map']     
#         durations = instance['duration']    
#         tau_time = 10  
        
#         N = len(nodes)
#         source_node, sink_node = 0, N + 1
#         V_nodes = [source_node] + list(nodes.keys()) + [sink_node]

#         node_to_flight, node_to_task = {}, {}
#         for fn, tmap in task_map.items():
#             for t_name, n_id in tmap.items():
#                 node_to_flight[n_id] = fn
#                 node_to_task[n_id] = t_name

#         static_EST = {source_node: 0, sink_node: 0}
#         for fn in flights:
#             arr_n = task_map[fn].get('arr')
#             dep_n = task_map[fn].get('dep')
#             # 使用 instance 中自带的 std 反推 sta，作为静态锚点
#             sta = instance.get('std', {}).get(fn, 1000) - 60 
#             if arr_n: static_EST[arr_n] = sta
#             if dep_n: static_EST[dep_n] = sta + durations.get(arr_n, 0)

#         def is_edge_valid(i, j):
#             if i == j or j == source_node or i == sink_node: return False
#             f_i, f_j = node_to_flight.get(i), node_to_flight.get(j)
            
#             # 同一航班的拓扑顺序剪枝
#             if f_i and f_j and f_i == f_j:
#                 order = {'arr': 0, 'dep': 1}
#                 if order[node_to_task[i]] >= order[node_to_task[j]]: return False
            
#             # 跨航班的时间绝对不可能剪枝 (使用 static_EST 剪枝)
#             if i != source_node and j != sink_node:
#                 # 稍微放宽一点静态剪枝范围到 120，确保各种早到晚到情况的路径都被包含进模型中
#                 if static_EST[i] + durations[i] + tau_time - static_EST[j] > 120: return False
#             return True

#         # [保持动态紧致]: 用传入的 predicted_a (ATA) 计算真实的 EST 和 tight_M (仅用于大M和时间变量LB)
#         EST = {source_node: 0, sink_node: 0}
#         for fn in flights:
#             arr_n = task_map[fn].get('arr')
#             dep_n = task_map[fn].get('dep')
#             ata = predicted_a[fn]
#             if arr_n: EST[arr_n] = ata
#             if dep_n: EST[dep_n] = ata + durations.get(arr_n, 0)
            
#         tight_M = max(EST.values()) + 240

#         vtype_bin = GRB.CONTINUOUS if relax else GRB.BINARY
        
#         x = {}
#         for i in V_nodes:
#             if i == sink_node: continue
#             for j in V_nodes:
#                 if not is_edge_valid(i, j): continue
#                 lb_val = ub_val = round(fixed_x[i, j]) if fixed_x and (i, j) in fixed_x else (0.0, 1.0)[1] if fixed_x is None else 0.0
#                 x[i, j] = model.addVar(lb=lb_val if fixed_x else 0, ub=ub_val if fixed_x else 1, vtype=vtype_bin, name=f"x_{i}_{j}")

#         t = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=EST.get(i, 0), name=f"t_{i}") for i in nodes}
#         w = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}") for i in nodes}
#         arr_nodes = [i for i, info in nodes.items() if 'arr' in info['type']]
#         v_delay = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_delay_{i}") for i in arr_nodes}
#         y = {fn: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{fn}") for fn in flights}

#         for i in nodes:
#             model.addConstr(gp.quicksum(x[j, i] for j in V_nodes if (j, i) in x) == 1, name=f"flow_in_{i}")
#             model.addConstr(gp.quicksum(x[i, j] for j in V_nodes if (i, j) in x) == 1, name=f"flow_out_{i}")

#         model.addConstr(gp.quicksum(x[source_node, j] for j in nodes if (source_node, j) in x) <= self.num_vehicles, name="max_fleet")
#         model.addConstr(gp.quicksum(x[source_node, j] for j in nodes if (source_node, j) in x) == 
#                         gp.quicksum(x[i, sink_node] for i in nodes if (i, sink_node) in x), name="depot_balance")

#         for i in V_nodes:
#             if i == sink_node: continue
#             for j in V_nodes:
#                 if (i, j) not in x or j == source_node or j == sink_node: continue
#                 dur_i = durations[i] if i != source_node else 0
#                 start_i = t[i] if i != source_node else 0
#                 arrival_at_j = start_i + dur_i + tau_time
#                 model.addConstr(t[j] >= arrival_at_j - tight_M * (1 - x[i, j]))
#                 model.addConstr(w[j] >= t[j] - arrival_at_j - tight_M * (1 - x[i, j]))

#         for fn in flights:
#             node_arr = task_map[fn].get('arr')
#             node_dep = task_map[fn].get('dep')
#             pred_arrival = predicted_a[fn]
            
#             if node_arr:
#                 # 为预测参数打上唯一的 name，给梯度提取当做锚点
#                 model.addConstr(v_delay[node_arr] >= t[node_arr] - pred_arrival, name=f"pred_vdelay_{fn}")
#                 model.addConstr(t[node_arr] >= pred_arrival, name=f"pred_ata_{fn}")
            
#             if node_arr and node_dep:
#                 model.addConstr(t[node_dep] >= t[node_arr] + durations[node_arr])
                
#             if node_dep:
#                 plan_d = instance.get('std', {}).get(fn) or (pred_arrival + 60)
#                 model.addConstr(y[fn] >= t[node_dep] + durations[node_dep] - (plan_d - 5))

#         obj_delay = gp.quicksum(self.pi * y[fn] for fn in flights)
#         obj_wait = gp.quicksum(self.alpha * w[i] for i in nodes)
#         obj_resp = gp.quicksum(self.beta * v_delay[i] for i in arr_nodes)
#         model.setObjective(obj_delay + obj_wait + obj_resp, GRB.MINIMIZE)

#         # 返回模型、变量字典以及后处理需要的图信息
#         vars_dict = {'x': x, 't': t, 'w': w, 'v_delay': v_delay, 'y': y}
#         return model, vars_dict, V_nodes, source_node, sink_node

#     def build_reduced_model(self, instance, ata_dict, active_edges):
#         env = gp.Env(empty=True)
#         env.setParam('OutputFlag', 0)
#         env.start()
#         model = gp.Model('Reduced_LP', env=env)
#         model.setParam('OutputFlag', 0)
#         flights = instance['flights']
#         nodes = instance['nodes']
#         task_map = instance['task_map']
#         durations = instance['duration']
#         tau_time = 10
        
#         t = {i: model.addVar(lb=0, name=f"t_{i}") for i in nodes} 
#         w = {i: model.addVar(lb=0, name=f"w_{i}") for i in nodes}
#         arr_nodes = [i for i, info in nodes.items() if 'arr' in info['type']]
#         v_delay = {i: model.addVar(lb=0, name=f"v_delay_{i}") for i in arr_nodes}
#         y = {fn: model.addVar(lb=0, name=f"y_{fn}") for fn in flights}
        
#         # [核心改动 1]: 引入松弛变量，允许时间预测出现失误，不让系统崩溃
#         slack_ata = {fn: model.addVar(lb=0, name=f"slack_ata_{fn}") for fn in flights}
        
#         for (i, j) in active_edges:
#             if i == 0 or j == len(nodes) + 1: 
#                 continue 
#             dur_i = durations[i]
#             model.addConstr(t[j] >= t[i] + dur_i + tau_time, name=f"route_{i}_{j}")
#             model.addConstr(w[j] >= t[j] - (t[i] + dur_i + tau_time), name=f"wait_{i}_{j}")
            
#         for fn in flights:
#             arr_n = task_map[fn].get('arr')
#             dep_n = task_map[fn].get('dep')
#             ata = ata_dict[fn]
            
#             if arr_n:
#                 # [核心改动 2]: 如果 t 小于 ata，用 slack_ata 垫上，保持数学上的 Feasible
#                 model.addConstr(t[arr_n] + slack_ata[fn] >= ata, name=f"pred_ata_{fn}")
#                 model.addConstr(v_delay[arr_n] >= t[arr_n] - ata, name=f"pred_vdelay_{fn}")
#             if arr_n and dep_n:
#                 model.addConstr(t[dep_n] >= t[arr_n] + durations[arr_n])
#             if dep_n:
#                 plan_d = instance.get('std', {}).get(fn) or (ata + 60)
#                 model.addConstr(y[fn] >= t[dep_n] + durations[dep_n] - (plan_d - 5))
                
#         obj_delay = gp.quicksum(self.pi * y[fn] for fn in flights)
#         obj_wait = gp.quicksum(self.alpha * w[i] for i in nodes)
#         obj_resp = gp.quicksum(self.beta * v_delay[i] for i in arr_nodes)
        
#         # [核心改动 3]: 给予松弛变量天价罚款！这会成为逼迫神经网络预测准确的最强反向梯度！
#         obj_slack = gp.quicksum(1000 * slack_ata[fn] for fn in flights)
        
#         model.setObjective(obj_delay + obj_wait + obj_resp + obj_slack, GRB.MINIMIZE)
        
#         vars_dict = {'t': t, 'w': w, 'v_delay': v_delay, 'y': y, 'slack_ata': slack_ata}
#         return model, vars_dict


#     def solve(self, instance, predicted_a, relax=False, fixed_x=None):
#         """
#         原有可视化与基准求解接口：调用 build_model 后执行 optimize 和结果后处理。
#         """
#         model, vars_dict, V_nodes, source_node, sink_node = self.build_model(instance, predicted_a, relax, fixed_x)
        
#         model.optimize()

#         if model.SolCount > 0:
#             x, t, w, v_delay, y = vars_dict['x'], vars_dict['t'], vars_dict['w'], vars_dict['v_delay'], vars_dict['y']
#             flights, nodes, arr_nodes = instance['flights'], instance['nodes'], [i for i, info in instance['nodes'].items() if 'arr' in info['type']]
            
#             active_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
#             adj = {i: [] for i in V_nodes}
#             for i, j in active_edges: adj[i].append(j)
                
#             paths = []
#             for start_node in adj[source_node]:
#                 curr = start_node
#                 path_nodes = []
#                 while curr != sink_node:
#                     path_nodes.append(curr)
#                     if adj[curr]: curr = adj[curr][0]
#                     else: break
#                 paths.append(path_nodes)
                
#             paths.sort(key=lambda p: len(p), reverse=True)
            
#             z_values_mapped, x_values_mapped = {}, {}
#             for idx, path_nodes in enumerate(paths):
#                 v_name = self.all_vehicles[idx]
#                 curr = source_node
#                 for node in path_nodes:
#                     z_values_mapped[(node, v_name)] = 1.0
#                     x_values_mapped[(curr, node, v_name)] = 1.0
#                     curr = node
#                 x_values_mapped[(curr, sink_node, v_name)] = 1.0

#             return model.objVal, {
#                 'delay_cost': sum(y[fn].X for fn in flights) * self.pi,
#                 'wait_cost': sum(w[i].X for i in nodes) * self.alpha,
#                 'resp_cost': sum(v_delay[i].X for i in arr_nodes) * self.beta,
#                 'x_values': x_values_mapped, 
#                 'z_values': z_values_mapped,
#                 't_values': {i: t[i].X for i in nodes}
#             }
        
#         return float('inf'), {}