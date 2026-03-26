import os
import sys
import pandas as pd
import numpy as np

from opt import load_and_solve, visualize_comprehensive

def generate_scenario_csv(output_path, delayed_idx, delay_min):
    """
    构造 15 架飞机的场景，到达间隔呈现显著的“入峰”特征。
    """
    # 14 个间隔，模拟从宽松到极度密集的演变
    intervals = [20,15,12,12,12,10,10,10,8,8,8,7,6,5] 
    
    ata_planned = [720]
    for iv in intervals:
        ata_planned.append(ata_planned[-1] + iv)
        
    records = []
    for i in range(15):  # 扩展至 15 架飞机
        f_id = f"F{i:02d}"
        sta = ata_planned[i]
        
        ata = sta + delay_min if i == delayed_idx else sta
        std = sta + 55
        
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
    """静默求解，屏蔽 Gurobi 标准输出"""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = load_and_solve(csv_path)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return result

if __name__ == "__main__":
    OUTPUT_DIR = "toy_data/15flights_experiments"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 求解 Baseline
    bl_csv = f"{OUTPUT_DIR}/15flights_baseline.csv"
    generate_scenario_csv(bl_csv, delayed_idx=-1, delay_min=0)
    
    print("--------------------------------------------------")
    print("求解全局 Baseline (全准点参照系)...")
    
    flights, sta_bl, ata_bl, s_bl, dur_bl, y_bl, K, v_assign_bl, t_map_bl = solve_silently(bl_csv)
    
    baseline_delay = sum(y_bl.values())
    print(f"Baseline 系统总起飞延误: {baseline_delay:.2f} 分钟")
    if baseline_delay > 0.1:
        print("[Warning] Baseline 存在非零延误，可能影响后续分析的准确性！")
    print("--------------------------------------------------")
        
    
    # 2. 实验参数设定
    delay_values = [10, 20, 30, 40, 50]
    num_flights = 15
    analysis_records = []

    print("开始执行航班敏感度批量实验...")
    
    # 3. 批量循环
    for f_idx in range(num_flights):
        for d_min in delay_values:
            csv_name = f"delay_F{f_idx:02d}_{d_min}min.csv"
            img_name = f"plot_F{f_idx:02d}_{d_min}min.png"
            
            csv_path = f"{OUTPUT_DIR}/{csv_name}"
            img_path = f"{OUTPUT_DIR}/{img_name}"
            
            generate_scenario_csv(csv_path, f_idx, d_min)
            
            try:
                f_pert, sta_pert, ata_pert, s_pert, dur_pert, y_pert, K_pert, v_assign_pert, t_map_pert = solve_silently(csv_path)
            except Exception as e:
                print(f"[Error] 场景 {csv_name} 求解失败: {e}")
                continue
                
            affected_flights_info = []
            total_cascading_delay = 0.0
            self_delay = max(0, y_pert[flights[f_idx]] - y_bl[flights[f_idx]])
            
            for i, f in enumerate(flights):
                if i > f_idx: 
                    diff = max(0, y_pert[f] - y_bl[f])
                    if diff > 0:
                        affected_flights_info.append(f"{f}(+{diff:.1f}m)")
                        total_cascading_delay += diff
                        
            affected_str = "无" if not affected_flights_info else " | ".join(affected_flights_info)
            
            analysis_records.append({
                '延误飞机': f"F{f_idx:02d}",
                '到达延误时间': f"{d_min}m",
                '自身起飞新增延误': f"{self_delay:.1f}m",
                '被波及的下游飞机及延误': affected_str,
                '下游延误总和': total_cascading_delay,
                '系统新增总延误': self_delay + total_cascading_delay
            })
            
            # 渲染并保存物理仿真图表
            visualize_comprehensive(
                flights=f_pert, 
                sta=sta_pert, 
                ata=ata_pert, 
                s_vals=s_pert, 
                durations=dur_pert, 
                y=y_pert, 
                K_total=K_pert, 
                veh_assign=v_assign_pert, 
                task_map=t_map_pert, 
                tau=10, 
                save_path=img_path
            )
            
            print(f"处理完成: F{f_idx:02d} 延误 {d_min}m -> {img_name}")

    # 4. 数据汇总与排序输出
    df_analysis = pd.DataFrame(analysis_records)
    
    # 按 系统新增总延误 和 下游延误总和 降序排列
    df_sorted = df_analysis.sort_values(by=['下游延误总和'], ascending=[False])
    
    summary_csv = f"{OUTPUT_DIR}/15flights_ranking.csv"
    df_sorted.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    
    print("--------------------------------------------------")
    print("敏感度分析报表")
    print("--------------------------------------------------")
    print(df_sorted.to_markdown(index=False))
    print(f"\n完整统计报表已保存至: {summary_csv}")