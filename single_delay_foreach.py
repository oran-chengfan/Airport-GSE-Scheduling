import os
import sys
import pandas as pd
import numpy as np

# 直接引用我们完善好的 opt.py 工具库
from opt import load_and_solve, visualize_comprehensive

def generate_scenario_csv(output_path, delayed_idx, delay_min):
    """在本地快速构造扰动场景"""
    # 航班间隔时间逐渐缩短，增加级联风险 (基于你最新的设置)
    intervals = [25, 23, 21, 19, 17, 15, 13, 11, 9] 
    
    ata_planned = [720]
    for iv in intervals:
        ata_planned.append(ata_planned[-1] + iv)
        
    records = []
    for i in range(10):
        f_id = f"F{i:02d}"
        sta = ata_planned[i]
        
        # 赋予到达延误
        ata = sta + delay_min if i == delayed_idx else sta
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
    """静默求解，防止控制台被大量日志刷屏"""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = load_and_solve(csv_path)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return result

if __name__ == "__main__":
    # 统一输出目录
    OUTPUT_DIR = "toy_data/batch_experiments"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 生成并求解全局 Baseline (全准点)
    bl_csv = f"{OUTPUT_DIR}/cascade_baseline.csv"
    generate_scenario_csv(bl_csv, delayed_idx=-1, delay_min=0)
    
    print("="*60)
    print("正在求解全局 Baseline")
    print("="*60)
    
    # 提取 Baseline 的延误情况
    flights, sta_bl, ata_bl, s_bl, dur_bl, y_bl, K, v_assign_bl, t_map_bl = solve_silently(bl_csv)
    
    # 2. 联合循环参数 (融合 10~50 分钟)
    delay_values = [10, 20, 30, 40, 50]
    num_flights = 10
    
    analysis_records = []

    print("\n 开始foreach\n")
    
    # 3. 启动批量实验
    for f_idx in range(num_flights):
        for d_min in delay_values:
            csv_name = f"delay_F{f_idx:02d}_{d_min}min.csv"
            img_name = f"plot_F{f_idx:02d}_{d_min}min.png"
            
            csv_path = f"{OUTPUT_DIR}/{csv_name}"
            img_path = f"{OUTPUT_DIR}/{img_name}"
            
            # A. 生成数据
            generate_scenario_csv(csv_path, f_idx, d_min)
            
            # B. 静默执行运筹求解
            try:
                f_pert, sta_pert, ata_pert, s_pert, dur_pert, y_pert, K_pert, v_assign_pert, t_map_pert = solve_silently(csv_path)
            except Exception as e:
                print(f"[严重错误] 场景 {csv_name} 求解失败: {e}")
                continue
                
            # C. 结算级联扰动物理量 (融合 analyse.py 的详细统计逻辑)
            affected_flights_info = []
            total_cascading_delay = 0.0
            self_delay = max(0, y_pert[flights[f_idx]] - y_bl[flights[f_idx]])
            
            for i, f in enumerate(flights):
                if i > f_idx: # 只看下游被波及的航班
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
            
            # D. 静默保存三合一分析图表
            print(f"F{f_idx:02d} 延误 {d_min}m -> 生成图表 {img_name}")
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
                save_path=img_path   # 传入 save_path 静默保存
            )

    # 4. 转化为 DataFrame 并进行终极排序
    df_analysis = pd.DataFrame(analysis_records)
    
    # 排序逻辑：优先看它对全局系统增加的“总代价”，其次看它造成的“级联伤害”
    df_sorted = df_analysis.sort_values(by=['系统新增总代价', '下游延误总和'], ascending=[False, False])
    
    # 保存排名表
    summary_csv = f"{OUTPUT_DIR}/00_critical_flights_ranking.csv"
    df_sorted.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("="*80)
    print(df_sorted.head(10).to_markdown(index=False))
    print(f"保存在: {OUTPUT_DIR}/")