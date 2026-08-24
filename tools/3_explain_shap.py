import os
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CreditMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(CreditMLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    data_path = 'data/clean_taiwan_credit.csv'
    model_path = 'outputs/baseline_model.pt'
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Missing data or model checkpoint.")
        
    df = pd.read_csv(data_path)
    checkpoint = torch.load(model_path, weights_only=False)
    
    target_col = 'default_payment_next_month'
    feature_cols = checkpoint['feature_names']
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    
    X_test_scaled = scaler.transform(X_test)
    
    model = CreditMLP(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    background_samples = X_test_scaled[:100]
    eval_samples = X_test_scaled[100:300]
    
    background_t = torch.tensor(background_samples, dtype=torch.float32)
    eval_t = torch.tensor(eval_samples, dtype=torch.float32)
    
    print("[INFO] Computing SHAP values...")
    try:
        explainer = shap.DeepExplainer(model, background_t)
        shap_values = explainer.shap_values(eval_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e:
        print(f"[WARN] DeepExplainer fallback: {e}. Using GradientExplainer...")
        explainer = shap.GradientExplainer(model, background_t)
        shap_values = explainer.shap_values(eval_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
    plt.style.use('dark_background')
    shap.summary_plot(
        shap_values, 
        features=eval_samples, 
        feature_names=feature_cols, 
        max_display=10,
        plot_size=(10, 6),
        show=False
    )
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    out_plot = 'outputs/shap_summary.png'
    plt.savefig(out_plot, transparent=True, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[INFO] SHAP summary plot saved to '{out_plot}'.")

if __name__ == '__main__':
    main()
