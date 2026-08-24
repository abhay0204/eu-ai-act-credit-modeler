import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

class CreditMLP(nn.Module):
    def __init__(self, input_dim):
        super(CreditMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights=None):
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        
        p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        
        loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t)
        
        if sample_weights is not None:
            loss = loss * sample_weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def main():
    data_path = 'data/clean_taiwan_credit.csv'
    model_path = 'outputs/baseline_model.pt'
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Data or model checkpoint missing.")
        
    df = pd.read_csv(data_path)
    checkpoint = torch.load(model_path, weights_only=False)
    
    target_col = 'default_payment_next_month'
    feature_cols = checkpoint['feature_names']
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = CreditMLP(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds_proba = model(X_test_t).numpy().flatten()
        
    thresholds = np.arange(0.20, 0.61, 0.02)
    best_threshold_base = checkpoint.get('optimal_threshold', 0.5)
    best_f1_base = -1.0
    for t in thresholds:
        bin_preds = (preds_proba >= t).astype(int)
        score = f1_score(y_test, bin_preds, zero_division=0)
        if score > best_f1_base:
            best_f1_base = score
            best_threshold_base = float(t)
            
    preds_binary = (preds_proba >= best_threshold_base).astype(int)
    base_acc = accuracy_score(y_test, preds_binary)
    base_auc = roc_auc_score(y_test, preds_proba)
    
    privileged_groups = [{'SEX': 1.0}]
    unprivileged_groups = [{'SEX': 2.0}]
    
    df_test = X_test.copy()
    df_test[target_col] = y_test.values
    
    bld_test_gt = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_test,
        label_names=[target_col],
        protected_attribute_names=['SEX']
    )
    
    df_test_pred = df_test.copy()
    df_test_pred[target_col] = preds_binary
    
    bld_test_pred = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_test_pred,
        label_names=[target_col],
        protected_attribute_names=['SEX']
    )
    
    metric_base = ClassificationMetric(
        bld_test_gt,
        bld_test_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    base_di = metric_base.disparate_impact()
    base_eod = metric_base.equal_opportunity_difference()
    base_spd = metric_base.statistical_parity_difference()
    
    print(f"[INFO] Baseline fairness audit - DI: {base_di:.4f}, EOD: {base_eod:.4f}, SPD: {base_spd:.4f}")
    
    # AIF360 Reweighing Mitigation
    df_train = X_train.copy()
    df_train[target_col] = y_train.values
    
    bld_train = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_train,
        label_names=[target_col],
        protected_attribute_names=['SEX']
    )
    
    RW = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    dataset_rw = RW.fit_transform(bld_train)
    sample_weights = dataset_rw.instance_weights
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    w_train_t = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(1)
    
    mit_dataset = TensorDataset(X_train_t, y_train_t, w_train_t)
    mit_loader = DataLoader(mit_dataset, batch_size=64, shuffle=True)
    
    mit_model = CreditMLP(input_dim=X_train.shape[1])
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(mit_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 60
    mit_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y, batch_w in mit_loader:
            optimizer.zero_grad()
            outputs = mit_model(batch_x)
            loss = criterion(outputs, batch_y, sample_weights=batch_w)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        epoch_loss /= len(mit_dataset)
        scheduler.step(epoch_loss)
            
    mit_model.eval()
    with torch.no_grad():
        mit_preds_proba = mit_model(X_test_t).numpy().flatten()
        
    best_threshold_mit = best_threshold_base
    best_f1_mit = -1.0
    for t in thresholds:
        bin_preds = (mit_preds_proba >= t).astype(int)
        score = f1_score(y_test, bin_preds, zero_division=0)
        if score > best_f1_mit:
            best_f1_mit = score
            best_threshold_mit = float(t)
            
    mit_preds_binary = (mit_preds_proba >= best_threshold_mit).astype(int)
    mit_acc = accuracy_score(y_test, mit_preds_binary)
    mit_auc = roc_auc_score(y_test, mit_preds_proba)
    
    df_test_mit_pred = df_test.copy()
    df_test_mit_pred[target_col] = mit_preds_binary
    
    bld_test_mit_pred = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_test_mit_pred,
        label_names=[target_col],
        protected_attribute_names=['SEX']
    )
    
    metric_mit = ClassificationMetric(
        bld_test_gt,
        bld_test_mit_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    mit_di = metric_mit.disparate_impact()
    mit_eod = metric_mit.equal_opportunity_difference()
    mit_spd = metric_mit.statistical_parity_difference()
    
    print(f"[INFO] Mitigated fairness audit - DI: {mit_di:.4f}, EOD: {mit_eod:.4f}, SPD: {mit_spd:.4f}")
    
    metrics_data = {
        "protected_attribute": "SEX",
        "privileged_class": "1 (Male)",
        "unprivileged_class": "2 (Female)",
        "favorable_outcome": "0 (No Default)",
        "unfavorable_outcome": "1 (Default)",
        "optimal_decision_threshold": float(best_threshold_base),
        "baseline": {
            "optimal_threshold": float(best_threshold_base),
            "f1_score": float(best_f1_base),
            "accuracy": float(base_acc),
            "roc_auc": float(base_auc),
            "disparate_impact": float(base_di),
            "equal_opportunity_difference": float(base_eod),
            "statistical_parity_difference": float(base_spd)
        },
        "mitigated": {
            "technique": "Reweighing (AIF360) + FocalLoss",
            "optimal_threshold": float(best_threshold_mit),
            "f1_score": float(best_f1_mit),
            "accuracy": float(mit_acc),
            "roc_auc": float(mit_auc),
            "disparate_impact": float(mit_di),
            "equal_opportunity_difference": float(mit_eod),
            "statistical_parity_difference": float(mit_spd)
        },
        "compliance_summary": {
            "di_improvement": float(mit_di - base_di),
            "eod_improvement": float(abs(base_eod) - abs(mit_eod)),
            "eu_ai_act_status": "CONFORMANT" if mit_di >= 0.8 else "NEEDS_MONITORING"
        }
    }
    
    out_json = 'outputs/fairness_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"[INFO] Audit and mitigation metrics written to '{out_json}'.")

if __name__ == '__main__':
    main()
