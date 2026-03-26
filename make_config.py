import json
import os
import argparse

def generate_config(num_baggage, num_pushback, data_dir):
    """
    动态生成车队和全局参数配置
    """
    config = {
        "fleets": {},
        "params": {
            # 这里可以放置一些全局通用参数，方便未来集中管理
            "default_travel_time": 15,
            "big_M": 10000
        }
    }

    # 1. 批量生成行李车 (Baggage Vehicles)
    # 职责：负责进港卸货 (arr) 和出港装货 (dep)
    for i in range(1, num_baggage + 1):
        vid = f"B{i}"
        config["fleets"][vid] = {
            "type": "baggage",
            "task_type": ["arr", "dep"] 
        }

    # 2. 批量生成牵引车 (Pushback Vehicles)
    # 职责：负责起飞前的推出任务 (push)
    for i in range(1, num_pushback + 1):
        vid = f"P{i}"
        config["fleets"][vid] = {
            "type": "pushback",
            "task_type": ["push"] 
        }

    # 3. 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    config_path = os.path.join(data_dir, 'config.json')

    # 4. 保存为 JSON 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"行李车: {num_baggage} 辆")
    print(f"牵引车: {num_pushback} 辆")
    print(f"已保存至: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成飞行区保障车辆调度配置")
    parser.add_argument('--b', type=int, default=3, help='行李车(Baggage)的数量')
    parser.add_argument('--p', type=int, default=2, help='牵引车(Pushback)的数量')
    parser.add_argument('--dir', type=str, default='./toy_data', help='配置文件保存目录')

    args = parser.parse_args()

    generate_config(num_baggage=args.b, num_pushback=args.p, data_dir=args.dir)



