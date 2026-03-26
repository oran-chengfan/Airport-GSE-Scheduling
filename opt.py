import pandas as pd
import json
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from solver import GSESolver

def load_and_solve(csv_path, config_path='toy_data/config.json'):
    df = pd.read_csv(csv_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    solver = GSESolver(config)
    flights = sorted(df['flight_id'].unique())
    instance = {
        'flights': flights, 
        'task_map': {}, 
        'duration': {}, 
        'std': {}
    }
    
    ata_dict, sta_dict = {}, {}
    for _, row in df.iterrows():
        f = row['flight_id']
        arr_id = f"{f}_arr"
        dep_id = f"{f}_dep"
        instance['task_map'][f] = {'arr': arr_id, 'dep': dep_id}
        instance['duration'][arr_id] = 25
        instance['duration'][dep_id] = 25
        instance['std'][f] = row['std_min']
        ata_dict[f] = row['ata_min']
        sta_dict[f] = row['sta_min']

    print(f"\n>>> 正在求解场景: {csv_path} ...")
    model, vars_dict, V_nodes, source_node, sink_node, durations = solver.build_model(instance, ata_dict, relax=False)
    
    model.setParam("OutputFlag", 0)  
    model.setParam("MIPFocus", 1)    
    model.setParam("MIPGap", 0.00)   
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise ValueError(f"求解失败，状态码: {model.Status}")

    # 提取按 Task 索引的连续变量
    s_vals = {n: vars_dict['s'][n].X for n in durations.keys()}
    y_vals = {f: vars_dict['y'][f].X for f in flights}
    
    # 提取离散边，追踪 Task 级别的车辆分配
    x_edges = [(i, j) for (i, j), var in vars_dict['x'].items() if var.X > 0.5]
    veh_assignments = {}
    veh_id = 1
    for src, dst in x_edges:
        if src == source_node:
            curr = dst
            while curr != sink_node:
                veh_assignments[curr] = f"V{veh_id}"
                next_nodes = [nxt for curr_node, nxt in x_edges if curr_node == curr]
                curr = next_nodes[0] if next_nodes else sink_node
            veh_id += 1

    print(f"求解完成！系统总物理起飞延误 (Sum of y): {sum(y_vals.values()):.2f} 分钟")
    
    return flights, sta_dict, ata_dict, s_vals, durations, y_vals, solver.num_vehicles, veh_assignments, instance['task_map']

def calculate_cascading_impact(y_base, y_pert, delayed_idx, flights):
    total_impact = 0.0
    impact_details = {}
    print(f"\n=== 级联扰动物理分析 ===")
    source_flight = flights[delayed_idx]
    print(f"扰动源: {source_flight}")
    
    for i, f in enumerate(flights):
        if i > delayed_idx:
            diff = max(0, y_pert[f] - y_base[f])
            total_impact += diff
            impact_details[f] = diff
            if diff > 0:
                print(f"  [物理传导] -> {f} 被波及，新增起飞延误: {diff:.2f} 分钟")
                
    print(f"[{source_flight}] 对系统下游造成的总级联伤害: {total_impact:.2f} 分钟\n")
    return total_impact, impact_details

def visualize_comprehensive(flights, sta, ata, s_vals, durations, y, K_total, veh_assign, task_map, tau=10, save_path=None):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    ax_gantt = fig.add_subplot(gs[0, :])      
    ax_supply = fig.add_subplot(gs[1, 0])     
    ax_bar = fig.add_subplot(gs[1, 1])        
    
    # ==========================================
    # 1. 绘制复刻版甘特图 (按 Task 级拆分，修复标尺错位)
    # ==========================================
    f_idx = {fn: i for i, fn in enumerate(flights)}
    short_flights = [f.replace('F', 'Flight ') for f in flights]
    
    for fn in flights:
        y_coord = f_idx[fn]
        sta_time = sta[fn]
        std_time = sta_time + 60
        deadline = std_time - 5 
        
        # 理想时间窗与缓冲
        ax_gantt.broken_barh([(sta_time, 55)], (y_coord - 0.4, 0.8), 
                       facecolors='none', edgecolor='gray', linestyle='--', linewidth=2, zorder=1)
        ax_gantt.broken_barh([(deadline, 5)], (y_coord - 0.4, 0.8), 
                       facecolors='#FFEBEE', edgecolor='red', linestyle=':', linewidth=1.5, zorder=1)
        ax_gantt.text(std_time - 2.5, y_coord, "Gate Close", color='red', fontsize=8, ha='center', va='center')

        arr_n = task_map[fn]['arr']
        dep_n = task_map[fn]['dep']
        
        start_arr = s_vals[arr_n]
        start_dep = s_vals[dep_n]
        dur_arr = durations[arr_n]
        dur_dep = durations[dep_n]
        
        v_id_arr = veh_assign.get(arr_n, "Unk")
        v_id_dep = veh_assign.get(dep_n, "Unk")
        
        # 独立绘制 ARR
        ax_gantt.broken_barh([(start_arr, dur_arr)], (y_coord - 0.4, 0.8), 
                       facecolors='#FF9999', edgecolor='black', linewidth=1, alpha=0.9, zorder=3)
        ax_gantt.text(start_arr + dur_arr/2, y_coord, v_id_arr, ha='center', va='center', color='black', fontsize=10, fontweight='bold', zorder=4)

        # 独立绘制 DEP 
        ax_gantt.broken_barh([(start_dep, dur_dep)], (y_coord - 0.4, 0.8), 
                       facecolors='#99CCFF', edgecolor='black', linewidth=1, alpha=0.9, zorder=3)
        ax_gantt.text(start_dep + dur_dep/2, y_coord, v_id_dep, ha='center', va='center', color='black', fontsize=10, fontweight='bold', zorder=4)

        ata_time = ata[fn]
        end_t = start_dep + dur_dep
        
        # 车等飞机/飞机等车 (Wait veh)
        if start_arr > ata_time + 0.1:
            ax_gantt.plot([ata_time, start_arr], [y_coord + 0.45, y_coord + 0.45], color='red', linestyle='-', linewidth=2, zorder=5)
            ax_gantt.scatter([ata_time], [y_coord + 0.45], color='red', marker='>', zorder=5)
            ax_gantt.text((ata_time + start_arr)/2, y_coord + 0.5, f"wait {start_arr - ata_time:.1f}m", color='red', fontsize=9, ha='center', va='bottom')

        # 【核心修复 1】：起飞延误必须严格从 STD -5 开始画，这样才能和右下角、求解器完全一致！
        ddl=std_time - 5
        if end_t > ddl + 0.1:
            ax_gantt.plot([ddl, end_t], [y_coord - 0.45, y_coord - 0.45], color='darkred', linestyle='-', linewidth=2, zorder=5)
            ax_gantt.scatter([ddl], [y_coord - 0.45], color='darkred', marker='|', s=100, zorder=5)
            ax_gantt.text((ddl + end_t)/2, y_coord - 0.5, f"delay {end_t - std_time:.1f}m", color='darkred', fontsize=9, ha='center', va='top')
   
    ax_gantt.set_yticks(range(len(flights)))
    ax_gantt.set_yticklabels(short_flights)
    ax_gantt.set_xlabel('Time')
    ax_gantt.set_title('GSE Scheduling Gantt Chart', fontsize=14, fontweight='bold')
    ax_gantt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    custom_lines = [
        mpatches.Patch(facecolor='#FF9999', edgecolor='black', label= 'ARR Task'),
        mpatches.Patch(facecolor='#99CCFF', edgecolor='black', label= 'DEP Task'),
        mpatches.Patch(facecolor='none', edgecolor='gray', linestyle='--', label='Scheduled time window'),
        mpatches.Patch(facecolor='#FFEBEE', edgecolor='red', linestyle=':', label='Gate close'),
        Line2D([0], [0], color='red', lw=2, label='Plane wait veh'),
        Line2D([0], [0], color='darkred', lw=2, label='Takeoff delay')
    ]
    ax_gantt.legend(handles=custom_lines, loc='lower right')

    # ==========================================
    # 2. 绘制时空供需挤兑图 (展现原始未平抑的物理需求)
    # ==========================================
    min_time = int(min(ata.values()))
    max_end_times = [s_vals[task_map[f]['dep']] + durations[task_map[f]['dep']] + tau for f in flights]
    max_time = int(max(max_end_times)) + 30
    timeline = np.arange(min_time, max_time)
    
    demand_curve, active_curve = np.zeros_like(timeline), np.zeros_like(timeline)
    
    for i, t in enumerate(timeline):
        d_count, a_count = 0, 0
        
        # 【核心修复】：计算“原始理想需求”
        for f in flights:
            arr_n = task_map[f]['arr']
            dep_n = task_map[f]['dep']
            d_arr, d_dep = durations[arr_n], durations[dep_n]
            deadline = sta[f] + 60 - 5 # 关门生死线
            
            # 原始卸货需求：飞机一落地就需要车。占用时间 = 作业 + 撤离空驶(tau)
            if ata[f] <= t < ata[f] + d_arr + tau:
                d_count += 1
                
            # 原始装货需求：为了不延误，最迟必须在这个时段要车。占用时间 = 作业 + 撤离空驶(tau)
            if deadline - d_dep <= t < deadline + tau:
                d_count += 1

        # 实际活跃车辆：Gurobi 削峰填谷后的结果
        for n in s_vals.keys():
            if s_vals[n] <= t < s_vals[n] + durations[n] + tau:
                a_count += 1
                
        demand_curve[i] = d_count
        active_curve[i] = a_count

    # 图表绘制保持不变
    ax_supply.fill_between(timeline, 0, demand_curve, color='salmon', alpha=0.4, label='Demand')
    ax_supply.plot(timeline, active_curve, color='royalblue', linewidth=2.5, label='Active vehicles')
    ax_supply.axhline(y=K_total, color='black', linestyle='--', linewidth=2, label=f'Capacity (K={K_total})')
    ax_supply.fill_between(timeline, K_total, demand_curve, where=(demand_curve > K_total), 
                     color='darkred', alpha=0.7, label='Over demand area')
                     
    ax_supply.set_title("Spatio-temporal S-D relationship", fontsize=12, fontweight='bold')
    ax_supply.set_xlabel("Time (minutes)")
    ax_supply.set_ylabel("Vehicles / Tasks")
    ax_supply.legend(loc='upper left')
    ax_supply.grid(True, linestyle=':', alpha=0.6)
    
    # ==========================================
    # 3. 绘制级联延误传播柱状图
    # ==========================================
    f_labels = [f.replace("F", "") for f in flights]
    arr_delays = [ata[f] - sta[f] for f in flights]
    dep_delays = [y[f] for f in flights]
    
    x = np.arange(len(flights))
    width = 0.35
    ax_bar.bar(x - width/2, arr_delays, width, color='darkgray', edgecolor='black', label='Arrival Disturbance')
    ax_bar.bar(x + width/2, dep_delays, width, color='darkorange', edgecolor='black', label='Departure Delay')
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(f_labels)
    ax_bar.set_title("One-flight-caused cascading effect", fontsize=12, fontweight='bold')
    ax_bar.set_xlabel("Index")
    ax_bar.set_ylabel("Delay (minutes)")
    ax_bar.legend()
    ax_bar.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    DELAYED_IDX = 5  
    
    print("--------------------------------------------------")
    print("阶段 1: 求解 Baseline (全准点参照系)")
    print("--------------------------------------------------")
    f_bl, sta_bl, ata_bl, s_bl, dur_bl, y_bl, K, v_assign_bl, t_map_bl = load_and_solve("toy_data/cascade_F05_D1_bl.csv")
    
    print("\n--------------------------------------------------")
    print("阶段 2: 求解扰动场景 (包含单点延误)")
    print("--------------------------------------------------")
    f_pert, sta_pert, ata_pert, s_pert, dur_pert, y_pert, K, v_assign_pert, t_map_pert = load_and_solve("toy_data/cascade_F05_D0.csv")
    
    calculate_cascading_impact(y_base=y_bl, y_pert=y_pert, delayed_idx=DELAYED_IDX, flights=f_pert)
    visualize_comprehensive(f_pert, sta_pert, ata_pert, s_pert, dur_pert, y_pert, K, v_assign_pert, t_map_pert)