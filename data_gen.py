import pandas as pd
import numpy as np
import os

def generate_cascade_scenario(num_days=50, num_flights=20, seed=42, target_K=10):
    """
    生成针对级联延误阻断机制的紧平衡测试风洞。
    物理参数设定: 真实服务周期 W = 70. 期望利用率 rho = 0.85.
    """
    records = []
    np.random.seed(seed)
    
    # 根据排队论推导紧平衡到达间隔
    W_true = 70.0
    rho_optimal = 0.85
    critical_interval = W_true / (target_K * rho_optimal)
    
    for day in range(num_days):
        feat_weather = np.random.rand() # 0~1 恶劣程度
        
        # 构造波峰到达间隔
        intervals = []
        peak_center = num_flights // 2
        for i in range(num_flights):
            dist = abs(i - peak_center)
            density = np.exp(- (dist ** 2) / (0.1 * num_flights ** 2))
            iv = int(25 - (25 - critical_interval) * density + np.random.normal(0, 1))
            intervals.append(max(4, iv))
            
        base_sta = [720]
        for iv in intervals[:-1]:
            base_sta.append(base_sta[-1] + iv)

        for i in range(num_flights):
            sta = base_sta[i]
            
            # 异质性构造：强制设定核心波峰区为 Critical，边缘区为 Robust
            is_critical = 1 if abs(i - peak_center) <= (num_flights * 0.2) else 0
            
            if is_critical:
                buffer_plan = int(np.random.uniform(0, 5))
            else:
                buffer_plan = int(np.random.uniform(15, 35))
                
            std = sta + 60 + buffer_plan
            
            # 随机延误生成机制 (发生概率与强度均受天气影响)
            delay_prob = 0.2 + 0.6 * feat_weather
            if np.random.rand() < delay_prob:
                # 发生延误
                delay_mean = feat_weather * 40.0 + (15.0 if is_critical else 0.0)
                delay_std = 5.0 + 10.0 * feat_weather
                delay_min = int(np.random.normal(delay_mean, delay_std))
                delay_min = max(5, delay_min) # 至少延误 5 分钟
            else:
                # 准点或早到 (负延误)
                delay_min = int(np.random.normal(-10, 10))
                delay_min = min(0, delay_min)
                
            records.append({
                'day_id': day, 
                'flight_id': f"F{i:03d}",
                'sta_min': sta, 
                'ata_min': sta + delay_min, 
                'std_min': std,
                'dur_arr': 25, 
                'dur_dep': 25, 
                'buffer': buffer_plan,
                'interval_next': intervals[i], 
                'feat_weather': round(feat_weather, 3),
                'delay_min': delay_min
            })
            
    return pd.DataFrame(records)

if __name__ == "__main__":

    num_train = 50
    num_val = int(0.3*num_train)
    num_test = int(0.3*num_train)

    num_flights = 20
    seed = 42
    K = 10
    df_train = generate_cascade_scenario(num_days=num_train, num_flights=20, seed=seed, target_K=K)
    df_val = generate_cascade_scenario(num_days=num_val, num_flights=20, seed=seed, target_K=K)
    df_test = generate_cascade_scenario(num_days=num_test, num_flights=20, seed=seed, target_K=K)
    prefix = f"toy_data/D{num_train}_F{num_flights}_K{K}"
    os.makedirs("toy_data", exist_ok=True)
    
    df_train.to_csv(f"{prefix}-Train.csv", index=False)
    df_val.to_csv(f"{prefix}-Val.csv", index=False)
    df_test.to_csv(f"{prefix}-Test.csv", index=False)
