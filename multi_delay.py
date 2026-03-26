import os
import sys
import pandas as pd
import numpy as np

from opt import load_and_solve, visualize_comprehensive

def generate_multi_scenario_csv(output_path, delay_dict):
    """
    构造场景，支持同时指定多架飞机的延误
    delay_dict: dict, 例如 {4: 30, 5: 30} 表示 F04 和 F05 同时延误 30 分钟
    """
    intervals = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7]
    
    ata_planned = [720]
    for iv in intervals:
        ata_planned.append(ata_planned[-1] + iv)
        
    records = []
    for i in range(15):
        f_id = f"F{i:02d}"
        sta = ata_planned[i]
        
        delay_min = delay_dict.get(i, 0)
        ata = sta + delay_min
        std = sta + 60 
        
        records.append({
            'day_id': 1,
            'flight_id': f_id,
            'sta_min': sta,
            'ata_min': ata,
            'std_min': std,
            'dur_arr': 25,
            'dur_dep': 25,
            'is_large': 0,
            'feat_weather': 0.0,
            'delay_min': max(0, ata - sta)
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

def solve_silently(csv_path):
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = load_and_solve(csv_path)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return result

def get_delay_impact(y_pert, y_bl, flights, target_flight_indices):
    """结算指标"""
    affected_flights_info = []
    total_cascading_delay = 0.0
    self_delay = 0.0
    
    max_idx = max(target_flight_indices) if target_flight_indices else -1
    
    for idx in target_flight_indices:
        self_delay += max(0, y_pert[flights[idx]] - y_bl[flights[idx]])
        
    for i, f in enumerate(flights):
        if i > max_idx: # 只统计最晚发生的扰动航班之后的波及
            diff = max(0, y_pert[f] - y_bl[f])
            if diff > 0:
                affected_flights_info.append(f"{f}(+{diff:.1f}m)")
                total_cascading_delay += diff
                
    affected_str = "无" if not affected_flights_info else " | ".join(affected_flights_info)
    return self_delay, total_cascading_delay, affected_str

if __name__ == "__main__":
    OUTPUT_DIR = "toy_data/multi_delay_experiments"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 求解 Baseline
    bl_csv = f"{OUTPUT_DIR}/multi_delay_baseline.csv"
    generate_multi_scenario_csv(bl_csv, {})
    flights, sta_bl, ata_bl, s_bl, dur_bl, y_bl, K, v_assign_bl, t_map_bl = solve_silently(bl_csv)
    
    # 2. 定义测试组 (测试相邻航班对)

    delay_list = [20,30,40]
    test_pairs = []
    for pair in range(15):
        if pair < 14:
            test_pairs.append((pair, pair+1))
    
    analysis_records = []
    
    for (f_a, f_b) in test_pairs:
        for delay_val in delay_list:
            scenarios = [
                (f"仅 F{f_a:02d} 延误", {f_a: delay_val}),
                (f"仅 F{f_b:02d} 延误", {f_b: delay_val}),
                (f"F{f_a:02d} & F{f_b:02d} 同时延误", {f_a: delay_val, f_b: delay_val})
            ]
        
            pair_results = {}
        
            for sc_name, delay_dict in scenarios:
                csv_name = f"combo_{sc_name}.csv"
                img_name = f"plot_{sc_name}.png"
                csv_path = f"{OUTPUT_DIR}/{csv_name}"
                img_path = f"{OUTPUT_DIR}/{img_name}"
            
                generate_multi_scenario_csv(csv_path, delay_dict)
                try:
                    f_pert, sta_pert, ata_pert, s_pert, dur_pert, y_pert, K_pert, v_assign_pert, t_map_pert = solve_silently(csv_path)
                except Exception:
                    continue
                
                self_delay, total_cascading_delay, affected_str = get_delay_impact(y_pert, y_bl, flights, list(delay_dict.keys()))
                sys_total = self_delay + total_cascading_delay
                
                pair_results[sc_name] = sys_total
                
                analysis_records.append({
                    '场景配置': sc_name,
                    '自身起飞新增延误': f"{self_delay:.1f}m",
                    '被波及的下游飞机及延误': affected_str,
                    '下游延误总和': total_cascading_delay,
                    '系统新增总延误': sys_total
                })
                
                visualize_comprehensive(
                    flights=f_pert, sta=sta_pert, ata=ata_pert, s_vals=s_pert, 
                    durations=dur_pert, y=y_pert, K_total=K_pert, veh_assign=v_assign_pert, 
                    task_map=t_map_pert, tau=10, save_path=img_path
                )
            
            # 计算非线性放大系数
            val_a = pair_results[f"仅 F{f_a:02d} 延误"]
            val_b = pair_results[f"仅 F{f_b:02d} 延误"]
            val_combo = pair_results[f"F{f_a:02d} & F{f_b:02d} 同时延误"]
            print(f"处理完成: F{f_a:02d} & F{f_b:02d} -> 被波及的下游飞机及延误: {affected_str}，下游延误总和: {total_cascading_delay:.1f}m")
            
            analysis_records.append({
                '场景配置': f"-> 【非线性检验】",
                '自身起飞新增延误': "-",
                '被波及的下游飞机及延误': f"A+B 独立相加: {val_a + val_b:.1f}m",
                '下游延误总和': "-",
                '系统新增总延误': f"同时发生: {val_combo:.1f}m"
            })
            analysis_records.append({'场景配置': "", '自身起飞新增延误': "", '被波及的下游飞机及延误': "", '下游延误总和': "", '系统新增总延误': ""})

    df_analysis = pd.DataFrame(analysis_records)
    summary_csv = f"{OUTPUT_DIR}/00_multi_delay_analysis_summary.csv"
    df_analysis.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(df_analysis.to_markdown(index=False))

