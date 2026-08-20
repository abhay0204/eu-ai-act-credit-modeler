import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    output_path = 'data/clean_taiwan_credit.csv'
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        print(f"[INFO] Using existing cleaned dataset at '{output_path}'.")
        return
        
    print("[INFO] Requesting Credit Card Default dataset from UCI Repository...")
    
    try:
        credit_data = fetch_ucirepo(id=350)
    except Exception as e:
        print(f"[WARN] UCI ID 350 fetch issue: {e}. Falling back to ID 277...")
        credit_data = fetch_ucirepo(id=277)
    
    X = credit_data.data.features.copy()
    y = credit_data.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    
    if hasattr(credit_data, 'variables') and credit_data.variables is not None:
        var_map = dict(zip(credit_data.variables['name'], credit_data.variables['description']))
        var_map = {k: v for k, v in var_map.items() if isinstance(v, str) and v.strip()}
        df.rename(columns=var_map, inplace=True)
    
    target_renames = {
        'Y': 'default_payment_next_month',
        'default payment next month': 'default_payment_next_month',
        'default.payment.next.month': 'default_payment_next_month'
    }
    df.rename(columns=target_renames, inplace=True)
    
    if 'default_payment_next_month' not in df.columns:
        for col in df.columns:
            if 'default' in col.lower() or col.lower() == 'y':
                df.rename(columns={col: 'default_payment_next_month'}, inplace=True)
                break
    
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
        
    os.makedirs('data', exist_ok=True)
    output_path = 'data/clean_taiwan_credit.csv'
    df.to_csv(output_path, index=False)
    
    print(f"[INFO] Cleaned dataset saved to '{output_path}' ({len(df)} rows, {len(df.columns)} columns).")

if __name__ == '__main__':
    main()
