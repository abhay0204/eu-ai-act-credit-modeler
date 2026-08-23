import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

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
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset missing at '{data_path}'.")
        
    df = pd.read_csv(data_path)
    target_col = 'default_payment_next_month'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('outputs', exist_ok=True)
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = CreditMLP(input_dim=X_train.shape[1])
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 60
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        epoch_loss /= len(train_dataset)
        scheduler.step(epoch_loss)
        if (epoch + 1) % 15 == 0:
            print(f"[INFO] Epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f} (lr: {optimizer.param_groups[0]['lr']:.6f})")
            
    model.eval()
    with torch.no_grad():
        preds_proba = model(X_test_t).numpy().flatten()
        
    thresholds = np.arange(0.20, 0.61, 0.02)
    best_threshold = 0.5
    best_f1 = -1.0
    
    for t in thresholds:
        bin_preds = (preds_proba >= t).astype(int)
        score = f1_score(y_test, bin_preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
            
    print(f"[INFO] Optimal threshold: {best_threshold:.2f} (max F1: {best_f1:.4f})")
    
    preds_binary = (preds_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds_binary)
    auc = roc_auc_score(y_test, preds_proba)
    f1 = f1_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary)
    rec = recall_score(y_test, preds_binary)
    
    print(f"[INFO] Evaluation metrics - Acc: {acc:.4f}, AUC: {auc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    
    model_path = 'outputs/baseline_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'feature_names': feature_cols,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'optimal_threshold': float(best_threshold),
        'accuracy': float(acc),
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec)
    }, model_path)
    print(f"[INFO] Model checkpoint saved to '{model_path}'.")

if __name__ == '__main__':
    main()
