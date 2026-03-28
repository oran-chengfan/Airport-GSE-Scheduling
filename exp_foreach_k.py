import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_gen import generate_dynamic_wind_tunnel
from po_train import train_po_baseline
from dfl_train import train_dfl
from evaluate import evaluate_model

def create_config_for_k(k, config_path="./toy_data/config.json"):
    """动态生成指定车数 K 的 config.json"""
    fleets = {}
    for i in range(1, k + 1):
        fleets[f"B{i}"] = {"type": "baggage", "task_type": ["arr", "dep"]}
    
    config = {
        "fleets": fleets,
        "params": {
            "default_travel_time": 10,
            "big_M": 1440
        }
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n 这次有 {k} 辆车")
    return config

def main():
    days = 100
    flights = 30
    seed = 42
    prefix = f"toy_data/D{days}-F{flights}-S{seed}"
    K_list = [13,14,15,18,20] # 给定数据集，探索K的变化
    
    print(f"\n生成数据集{days}天，每天 {flights}航班，随机种子 {seed}")
    val_days = max(1, int(days * 0.2))
    test_days = max(1, int(days * 0.3))
    
    df_train = generate_dynamic_wind_tunnel(days, flights, seed, target_K=15)
    df_val = generate_dynamic_wind_tunnel(val_days, flights, seed + 1, target_K=15)
    df_test = generate_dynamic_wind_tunnel(test_days, flights, seed + 2, target_K=15)
    
    df_train.to_csv(f"{prefix}-Train.csv", index=False)
    df_val.to_csv(f"{prefix}-Val.csv", index=False)
    df_test.to_csv(f"{prefix}-Test.csv", index=False)
    
    # PO训练
    print("\n Train PO baseline")
    if not os.path.exists(f"{prefix}-PO_Best.pth"):
        train_po_baseline(prefix)
    else:
        print(f"检测到已有 PO 模型 {prefix}-PO_Best.pth")

    # 用于记录结果的字典
    results = {
        'K': [],
        'PO_Regret': [], 'DFL_Regret': [],
        'PO_MSE': [], 'DFL_MSE': []
    }

    test_df = pd.read_csv(f"{prefix}-Test.csv")

    # 4. 遍历 K 进行核心测试
    print("\n 遍历K：")
    for k in K_list:
        print(f"\n{'-'*40}")
        print(f" 评估 K = {k} 辆车")
        print(f"{'-'*40}")
        
        # 覆写环境 config
        config = create_config_for_k(k)
        
        # 训练专门针对 K 辆车的 DFL 模型 (内部将调用 PO 作为 Warm Start)
        train_dfl(prefix)
        
        # 评估对抗
        po_model_path = f"{prefix}-PO_Best.pth"
        dfl_model_path = f"{prefix}-DFL_Best.pth"
        
        print(f"\n[测试集评测] K={k} ...")
        po_mse, po_regret, _ = evaluate_model("PO Baseline", po_model_path, test_df, config)
        dfl_mse, dfl_regret, _ = evaluate_model("DFL Model", dfl_model_path, test_df, config)
        
        results['K'].append(k)
        results['PO_Regret'].append(po_regret)
        results['DFL_Regret'].append(dfl_regret)
        results['PO_MSE'].append(po_mse)
        results['DFL_MSE'].append(dfl_mse)

    print("\n\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"{'K (Vehicles)':<15} | {'PO Regret':<15} | {'DFL Regret':<15} | {'Winner'}")
    print("-" * 60)
    for i in range(len(K_list)):
        k = results['K'][i]
        po_r = results['PO_Regret'][i]
        dfl_r = results['DFL_Regret'][i]
        winner = "DFL ★" if dfl_r < po_r else "PO"
        print(f"{k:<15} | {po_r:<15.2f} | {dfl_r:<15.2f} | {winner}")
    print("="*60)

# 6. 绘制顶会级别的相变交叉曲线
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(results['K'], results['PO_Regret'], label='PO (MSE Driven)', marker='o', linewidth=2.5, linestyle='--', color='gray')
    plt.plot(results['K'], results['DFL_Regret'], label='DFL (Regret Driven)', marker='s', linewidth=3, color='#d62728')
    
    # [核心修正] 动态计算相变阴影区域，禁止硬编码
    k_array = np.array(results['K'])
    po_regret_array = np.array(results['PO_Regret'])
    dfl_regret_array = np.array(results['DFL_Regret'])
    
    # 找到 DFL 胜出的连续区间
    dfl_wins = dfl_regret_array < po_regret_array
    for i in range(len(k_array) - 1):
        if dfl_wins[i] or dfl_wins[i+1]:
            # 仅当相邻两点中存在 DFL 胜出时，进行区间高亮
            plt.axvspan(k_array[i], k_array[i+1], color='red', alpha=0.1)
            
    # 添加一次图例标签
    if np.any(dfl_wins):
        plt.axvspan(-1, -1, color='red', alpha=0.1, label='Phase Transition (DFL Wins)')
    
    plt.xlim(min(k_array) - 1, max(k_array) + 1)
    plt.title('System Regret vs. Fleet Capacity (K)', fontsize=14, fontweight='bold')
    plt.xlabel('Fleet Capacity K (Number of Vehicles)', fontsize=12)
    plt.ylabel('System Regret (Delay Cost)', fontsize=12)
    plt.xticks(k_array)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.tight_layout()
    plt.savefig(f'Phase_Transition_Curve_{days}Days_{flights}Flights_{seed}.png')

if __name__ == "__main__":
    main()
