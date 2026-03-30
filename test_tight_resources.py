import pandas as pd
from evaluate import evaluate_model
from utils import create_dynamic_config

def main():
    prefix = "toy_data/D50-F20-K10"
    num_flights = 20
    test_df = pd.read_csv(f"{prefix}-Test.csv")
    po_model_path = f"{prefix}-PO_Best.pth"
    dfl_model_path = f"{prefix}-DFL_Best.pth"
    
    test_k_list = [7,8, 9, 10, 11,12,15,20,100]
    results = []

    for k in test_k_list:
        config = create_dynamic_config(k)
        po_mse, po_reg, po_surr = evaluate_model("PO", po_model_path, test_df, config, k)
        dfl_mse, dfl_reg, dfl_surr = evaluate_model("DFL", dfl_model_path, test_df, config,k)
        
        winner = "DFL ★" if dfl_reg < po_reg else "PO"
        results.append((k, po_mse, dfl_mse, po_reg, dfl_reg, winner))

    print("\n" + "="*70)
    print(f"{'K':<5} | {'PO MSE':<8} | {'DFL MSE':<8} | {'PO Regret':<10} | {'DFL Regret':<10} | {'Winner'}")
    print("-" * 70)
    for row in results:
        k, p_m, d_m, p_r, d_r, w = row
        print(f"{k:<5} | {p_m:<8.1f} | {d_m:<8.1f} | {p_r:<10.1f} | {d_r:<10.1f} | {w}")
    print("="*70)

if __name__ == "__main__":
    main()
