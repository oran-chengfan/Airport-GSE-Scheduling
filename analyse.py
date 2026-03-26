import pandas as pd
import numpy as np
import os
from opt import load_and_solve

def generate_scenario_csv(output_path, delayed_idx, delay_min):
    """在内存或本地快速构造扰动场景"""
    intervals = [25,23,21,19,17,15,13,11,9]
    ata_planned = [720]
    for iv in intervals:
        ata_planned.append(ata_planned[-1] + iv)
        
    records = []
    for i in range(10):
        f_id = f"F{i:02d}"
        sta = ata_planned[i]
        ata = sta + delay_min if i == delayed_idx else sta
        std = sta + 60 
        records.append({
            'day_id': 1, 'flight_id': f_id, 'sta_min': sta, 'ata_min': ata, 'std_min': std,
            'dur_arr': 25, 'dur_dep': 25, 'is_large': 0, 'feat_weather': 0.0,
            'delay_min': max(0, ata - sta)
        })
    pd.DataFrame(records).to_csv(output_path, index=False)

def analyze_critical_flights():
    OUTPUT_DIR = "toy_data/sensitivity_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 计算绝对的 Baseline (全准点)
    bl_csv = f"{OUTPUT_DIR}/baseline.csv"
    generate_scenario_csv(bl_csv, -1, 0)
    print("正在求解全局 Baseline...")
    flights, _, _, _, _, y_bl, _, _, _ = load_and_solve(bl_csv)
    
    delay_values = [20, 30, 40, 50]
    analysis_records = []
    
    print("\n🚀 开始进行航班敏感度遍历 (找出关键飞机)...\n")
    for f_idx in range(10):
        for d_min in delay_values:
            csv_path = f"{OUTPUT_DIR}/sim_F{f_idx:02d}_D{d_min}.csv"
            generate_scenario_csv(csv_path, f_idx, d_min)
            
            try:
                # 隐藏内部打印，保持控制台整洁
                import sys, os as sys_os
                old_stdout = sys.stdout
                sys.stdout = open(sys_os.devnull, 'w')
                _, _, _, _, _, y_pert, _, _, _ = load_and_solve(csv_path)
                sys.stdout.close()
                sys.stdout = old_stdout
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[Error] F{f_idx:02d} 延误 {d_min}m 求解失败: {e}")
                continue
                
            # 2. 核心逻辑：精准计算谁被波及了，波及了多少？
            affected_flights_info = []
            total_cascading_delay = 0.0
            self_delay = max(0, y_pert[flights[f_idx]] - y_bl[flights[f_idx]])
            
            for i, f in enumerate(flights):
                if i > f_idx: # 只看下游
                    diff = max(0, y_pert[f] - y_bl[f])
                    if diff > 0:
                        affected_flights_info.append(f"{f}(+{diff:.1f}m)")
                        total_cascading_delay += diff
                        
            affected_str = "无" if not affected_flights_info else " | ".join(affected_flights_info)
            
            analysis_records.append({
                '延误飞机': f"F{f_idx:02d}",
                '到达延误时间': f"{d_min}m",
                '自身起飞新增延误': f"{self_delay:.1f}m",
                '影响的下游飞机及延误': affected_str,
                '下游总和': total_cascading_delay,
                '新增成本': self_delay + total_cascading_delay
            })
            
    # 3. 转化为 DataFrame 并进行终极排序
    df_analysis = pd.DataFrame(analysis_records)
    
    # 【排序逻辑】：优先看它对全局系统增加的“总代价”，其次看它造成的“级联伤害”
    df_sorted = df_analysis.sort_values(by=['下游总和'], ascending=[False])
    
    # 格式化输出
    print(df_sorted.to_markdown(index=False))
    df_sorted.to_csv(f"{OUTPUT_DIR}/critical_flights_ranking.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ 关键航班已生成: {OUTPUT_DIR}/critical_flights_ranking.csv")

if __name__ == "__main__":
    analyze_critical_flights()