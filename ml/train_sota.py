#!/usr/bin/env python3
"""
SOTA Deep Learning for Cricket Ball Outcome Prediction
- Transformer attention mechanism
- Focal Loss for class imbalance
- Mixup data augmentation
- Cosine annealing with warm restarts
- Model ensemble
- Residual connections
"""

import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import math

# Paths
ML_DIR = Path(__file__).parent

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

OUTCOME_CLASSES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET']
NUM_CLASSES = len(OUTCOME_CLASSES)

FEATURE_COLS = [
    'over', 'ball_in_over', 'innings', 'phase_powerplay', 'phase_middle', 'phase_death',
    'runs_scored', 'wickets_lost', 'run_rate', 'required_rate', 'balls_remaining',
    'batsman_sr', 'batsman_avg', 'batsman_experience',
    'bowler_economy', 'bowler_sr', 'bowler_experience',
    'matchup_sr', 'matchup_balls', 'venue_avg_score', 'recent_form'
]
NUM_FEATURES = len(FEATURE_COLS)


# ============ FOCAL LOSS ============
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============ LABEL SMOOTHING ============
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), confidence)
        smooth_target += self.smoothing / self.num_classes
        return F.kl_div(F.log_softmax(pred, dim=1), smooth_target, reduction='batchmean')


# ============ DATASET ============
class CricketDataset(Dataset):
    def __init__(self, filepath):
        self.X, self.y = [], []
        with open(filepath, 'r') as f:
            for row in csv.DictReader(f):
                self.X.append([float(row[c]) for c in FEATURE_COLS])
                self.y.append(int(row['outcome_idx']))
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        self.mean = self.X.mean(0)
        self.std = self.X.std(0) + 1e-8
        
    def normalize(self, mean=None, std=None):
        self.mean = mean if mean is not None else self.mean
        self.std = std if std is not None else self.std
        self.X = (self.X - self.mean) / self.std
        
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
    
    def get_class_weights(self):
        counts = torch.bincount(self.y, minlength=NUM_CLASSES).float()
        weights = 1.0 / (counts + 1)
        return weights / weights.sum() * NUM_CLASSES


# ============ ADVANCED MODELS ============

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(F.gelu(x + self.block(x)))


class SelfAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N = x.shape[0], 1
        x = x.unsqueeze(1)  # Add sequence dim
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x.squeeze(1)


class TransformerMLP(nn.Module):
    """Transformer-style MLP with attention and residuals"""
    def __init__(self, in_dim, hidden=256, num_blocks=4, heads=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': SelfAttention(hidden, heads, dropout),
                'norm1': nn.LayerNorm(hidden),
                'res': ResidualBlock(hidden, dropout),
                'norm2': nn.LayerNorm(hidden),
            })
            for _ in range(num_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, NUM_CLASSES)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block['norm1'](x + block['attn'](x))
            x = block['norm2'](x + block['res'](x))
        
        return self.head(x)


class WideDeepMLP(nn.Module):
    """Wide & Deep architecture"""
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(in_dim, NUM_CLASSES)
        
        # Deep component
        self.deep = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.wide(x) + self.deep(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with learned weights"""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
    def forward(self, x):
        outputs = torch.stack([m(x) for m in self.models], dim=0)
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
        return (outputs * weights).sum(0)


# ============ MIXUP AUGMENTATION ============
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ TRAINING ============
def train_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        if use_mixup and np.random.random() > 0.5:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=0.2)
            optimizer.zero_grad()
            out = model(X)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += criterion(out, y).item()
            
            probs = F.softmax(out, dim=1)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return total_loss / len(loader), correct / total, all_preds, all_probs


def train_model(model, train_loader, val_loader, epochs=60, lr=0.001, warmup=5, use_mixup=True):
    """Train with cosine annealing and warmup"""
    model = model.to(DEVICE)
    
    # Get class weights from training data
    class_weights = train_loader.dataset.get_class_weights().to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_acc, best_state = 0, None
    
    print(f"\n{'Ep':>3} {'LR':>8} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7}")
    print("-" * 50)
    
    for ep in range(epochs):
        # Warmup
        if ep < warmup:
            for g in optimizer.param_groups:
                g['lr'] = lr * (ep + 1) / warmup
        
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, use_mixup)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        
        if ep >= warmup:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{ep+1:>3} {current_lr:>8.6f} {tr_loss:>8.4f} {tr_acc:>7.4f} {vl_loss:>8.4f} {vl_acc:>7.4f}")
        
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_acc


def create_lookup(model, mean, std):
    model.eval()
    lookup = {}
    with torch.no_grad():
        for over in [0, 3, 6, 10, 15, 18]:
            for wkts in [0, 2, 4, 6]:
                for runs in [0, 40, 80, 120, 160]:
                    pp = 1 if over < 6 else 0
                    mid = 1 if 6 <= over < 15 else 0
                    death = 1 if over >= 15 else 0
                    rr = runs / (over * 6 + 3) * 6 if over > 0 else 0
                    feat = torch.FloatTensor([[over, 3, 1, pp, mid, death, runs, wkts, rr, 8.0, (20-over)*6, 130, 30, 2.0, 8.0, 20, 2.0, 130, 0.5, 165, 10]])
                    feat = ((feat - mean) / std).to(DEVICE)
                    probs = F.softmax(model(feat), 1).cpu().numpy()[0].tolist()
                    lookup[f"{over}_{wkts}_{runs}"] = probs
    return lookup


def main():
    print("=" * 60)
    print("🚀 SOTA Deep Learning Training")
    print("=" * 60)
    
    # Load data
    train_ds = CricketDataset(ML_DIR / 'train_data.csv')
    val_ds = CricketDataset(ML_DIR / 'val_data.csv')
    test_ds = CricketDataset(ML_DIR / 'test_data.csv')
    
    mean, std = train_ds.mean, train_ds.std
    train_ds.normalize()
    val_ds.normalize(mean, std)
    test_ds.normalize(mean, std)
    
    print(f"📊 Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    
    # Class distribution
    class_counts = torch.bincount(train_ds.y, minlength=NUM_CLASSES)
    print(f"📊 Class distribution:")
    for i, c in enumerate(OUTCOME_CLASSES):
        print(f"   {c}: {class_counts[i].item():,} ({class_counts[i].item()/len(train_ds)*100:.1f}%)")
    
    train_dl = DataLoader(train_ds, 256, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, 512, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, 512, num_workers=4, pin_memory=True)
    
    results = []
    trained_models = {}
    
    # ===== 1. Transformer MLP =====
    print("\n" + "=" * 60)
    print("🔷 Training Transformer MLP")
    print("=" * 60)
    
    trans = TransformerMLP(NUM_FEATURES, hidden=256, num_blocks=4, heads=4, dropout=0.3)
    print(f"   Params: {sum(p.numel() for p in trans.parameters()):,}")
    
    t0 = time.time()
    trans, _ = train_model(trans, train_dl, val_dl, epochs=50, lr=0.0005)
    trans_time = time.time() - t0
    
    _, trans_acc, _, _ = evaluate(trans, test_dl, FocalLoss())
    results.append(('Transformer MLP', trans_acc, trans_time))
    trained_models['transformer'] = trans
    print(f"✅ Test Acc: {trans_acc:.4f}")
    
    # ===== 2. Wide & Deep =====
    print("\n" + "=" * 60)
    print("🔷 Training Wide & Deep")
    print("=" * 60)
    
    wd = WideDeepMLP(NUM_FEATURES, dropout=0.35)
    print(f"   Params: {sum(p.numel() for p in wd.parameters()):,}")
    
    t0 = time.time()
    wd, _ = train_model(wd, train_dl, val_dl, epochs=50, lr=0.0008)
    wd_time = time.time() - t0
    
    _, wd_acc, _, _ = evaluate(wd, test_dl, FocalLoss())
    results.append(('Wide & Deep', wd_acc, wd_time))
    trained_models['widedeep'] = wd
    print(f"✅ Test Acc: {wd_acc:.4f}")
    
    # ===== 3. Ensemble =====
    print("\n" + "=" * 60)
    print("🔷 Creating Ensemble")
    print("=" * 60)
    
    ensemble = EnsembleModel([trained_models['transformer'], trained_models['widedeep']])
    ensemble = ensemble.to(DEVICE)
    
    _, ens_acc, _, _ = evaluate(ensemble, test_dl, FocalLoss())
    results.append(('Ensemble', ens_acc, 0))
    print(f"✅ Ensemble Test Acc: {ens_acc:.4f}")
    
    # ===== RESULTS =====
    print("\n" + "=" * 60)
    print("📊 SOTA RESULTS")
    print("=" * 60)
    
    results.append(('LightGBM (baseline)', 0.413, 0))
    results.append(('Deep MLP (previous)', 0.4144, 0))
    
    best = max(results, key=lambda x: x[1])
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Time':>8}")
    print("-" * 45)
    for name, acc, t in sorted(results, key=lambda x: -x[1]):
        mark = " 🏆" if name == best[0] else ""
        print(f"{name:<25} {acc:>10.4f} {t:>8.1f}s{mark}")
    
    # Export best model
    best_name = best[0]
    if best_name == 'Ensemble':
        best_model = ensemble
    elif best_name == 'Transformer MLP':
        best_model = trans
    else:
        best_model = wd
    
    lookup = create_lookup(best_model, mean, std)
    
    model_data = {
        'model_type': best_name.lower().replace(' ', '_').replace('&', 'and'),
        'n_classes': NUM_CLASSES,
        'classes': OUTCOME_CLASSES,
        'features': FEATURE_COLS,
        'test_accuracy': best[1],
        'scenario_lookup': lookup,
        'phase_adjustments': {
            'powerplay': {'DOT': 0.95, 'FOUR': 1.15, 'SIX': 1.10, 'WICKET': 0.90},
            'middle': {'DOT': 1.05, 'FOUR': 0.90, 'SIX': 0.85, 'WICKET': 1.00},
            'death': {'DOT': 0.90, 'FOUR': 1.10, 'SIX': 1.20, 'WICKET': 1.15}
        }
    }
    
    outpath = ML_DIR / 'sota_model.json'
    with open(outpath, 'w') as f:
        json.dump(model_data, f)
    print(f"\n✅ Exported {outpath.name} ({outpath.stat().st_size/1024:.1f} KB)")
    
    # Update main model if SOTA beats previous
    if best[1] > 0.4144:
        import shutil
        shutil.copy(outpath, ML_DIR / 'ball_outcome_model.json')
        print(f"🏆 SOTA beats previous best! Updated ball_outcome_model.json")
        improvement = (best[1] - 0.413) / 0.413 * 100
        print(f"   Improvement over LightGBM: +{improvement:.2f}%")
    
    print("\n✅ SOTA Training Complete!")


if __name__ == "__main__":
    main()
