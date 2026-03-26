import numpy as np
import pandas as pd


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

