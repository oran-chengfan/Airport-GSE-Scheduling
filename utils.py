import numpy as np
import pandas as pd
import os
import json


def rebuild_instance_from_group(day_id, group_df):
    instance = {
        'flights': [],
        'task_map': {},
        'duration': {},
        'std': {},
        'sta': {}
    }
    
    for _, row in group_df.iterrows():
        f_id = row['flight_id']
        instance['flights'].append(f_id)
        arr_node = f"{f_id}_arr"
        dep_node = f"{f_id}_dep"
        instance['task_map'][f_id] = {'arr': arr_node, 'dep': dep_node}
        instance['duration'][arr_node] = row['dur_arr']
        instance['duration'][dep_node] = row['dur_dep']
        instance['std'][f_id] = row['std_min']
        instance['sta'][f_id] = row['sta_min']
        
    return instance


def augment_features(raw_features, k_vehicles, num_flights):
    """
    恢复 11 维特征空间：X_new = [X_base, r, 1/r, X_base * r, X_base * (1/r)]
    【防爆炸截断】：在特征映射时，运力上限截断为 15。
    因为对于 20 架航班，15 辆车已是绝对宽裕，更高 K 物理意义相同，防数学外推。
    """
    n_samples = raw_features.shape[0]
    
    # 物理截断，防止 K=100 时的线性项爆炸
    effective_k = min(k_vehicles, num_flights) 
    
    r = effective_k / float(num_flights)
    r_inv = 1.0 / r
    
    r_vec = np.full((n_samples, 1), r)
    r_inv_vec = np.full((n_samples, 1), r_inv)
    cross_r = raw_features * r
    cross_r_inv = raw_features * r_inv
    
    aug_features = np.hstack([raw_features, r_vec, r_inv_vec, cross_r, cross_r_inv])
    return aug_features

def create_dynamic_config(k, config_path="./toy_data/config.json"):
    fleets = {}
    for i in range(1, k + 1):
        fleets[f"B{i}"] = {"type": "baggage", "task_type": ["arr", "dep"]}
    config = {
        "fleets": fleets,
        "params": {"default_travel_time": 10, "big_M": 1440}
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config