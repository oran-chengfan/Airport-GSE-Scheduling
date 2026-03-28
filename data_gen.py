import pandas as pd
import numpy as np
import os
import argparse

def generate_dynamic_wind_tunnel(num_days, num_flights, seed,target_K,num_peaks=2):
    records = []

    np.random.seed(seed)
    rho_optimal = 0.85
    critical_interval = 60/(target_K * rho_optimal)
    for day in range(num_days):
        peak_centers = np.random.uniform(0.25, 0.75, num_peaks) * num_flights
        intervals = []
        for i in range(num_flights):
            min_dist = min([abs(i - pc) for pc in peak_centers])
            density_factor = np.exp(- (min_dist ** 2) / (0.05 * num_flights ** 2))
            iv = int(25 - (25 - critical_interval) * density_factor + np.random.normal(0, 1))
            intervals.append(max(5, iv))
            
        base_sta = [720]
        for iv in intervals[:-1]:
            base_sta.append(base_sta[-1] + iv)

        downstream_densities = []
        for i in range(num_flights):
            avg_iv = np.mean(intervals[i:i+3]) if i < num_flights - 3 else 30
            downstream_densities.append(avg_iv)
        critical_idx = np.argmin(downstream_densities)

        feat_weather = np.random.rand()

        for i in range(num_flights):
            dist_to_crit = abs(i - critical_idx)
            crit_weight = np.exp(-dist_to_crit)
            buffer_plan = int((1 - crit_weight) * np.random.randint(15, 30) + crit_weight * np.random.randint(0, 5))
            
            sta = base_sta[i]
            std = sta + 60 + buffer_plan

            delay_mean = feat_weather * (10 + 40 * crit_weight)
            delay_std = 2 + 8 * crit_weight
            delay_min = int(np.random.normal(delay_mean, delay_std))
            if crit_weight > 0.9: delay_min += 15 
            delay_min = max(0, delay_min)
            
            records.append({
                'day_id': day, 'flight_id': f"F{i:03d}",
                'sta_min': sta, 'ata_min': sta + delay_min, 'std_min': std,
                'dur_arr': 25, 'dur_dep': 25, 'buffer': buffer_plan,
                'interval_next': intervals[i], 'feat_weather': round(feat_weather, 3),
                'delay_min': delay_min
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成数据集")
    parser.add_argument("-D", "--days", type=int, default=30, help="训练集的天数")
    parser.add_argument("-F", "--flights", type=int, default=15, help="每天的航班数")
    parser.add_argument("-S", "--seed", type=int, default=42, help="随机种子")
    parser.add_argument("-K", "--target_K", type=int, default=10, help="车数量")
    args = parser.parse_args()

    os.makedirs("toy_data", exist_ok=True)
    
    val_days = max(1, int(args.days * 0.2))
    test_days = max(1, int(args.days * 0.3))

    df_train = generate_dynamic_wind_tunnel(args.days, args.flights,args.target_K, args.seed)
    df_val = generate_dynamic_wind_tunnel(val_days, args.flights,args.target_K, args.seed + 1)
    df_test = generate_dynamic_wind_tunnel(test_days, args.flights,args.target_K, args.seed + 2)

    base_name = f"toy_data/D{args.days}-F{args.flights}-S{args.seed}"
    df_train.to_csv(f"{base_name}-Train.csv", index=False)
    df_val.to_csv(f"{base_name}-Val.csv", index=False)
    df_test.to_csv(f"{base_name}-Test.csv", index=False)
    
    print(f"生成训练集 {args.days} 天 x {args.flights} 航班; 验证集 {val_days} 天 x {args.flights} 航班; 测试集 {test_days} 天 x {args.flights} 航班")

