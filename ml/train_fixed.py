#!/usr/bin/env python3
"""
FIXED SOTA Training - No Weighted Sampler
Key insight: Weighted sampler causes train/val distribution mismatch
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

ML_DIR = Path(__file__).parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {DEVICE}")

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


# ============ SIMPLE BUT EFFECTIVE MLP ============
class SimpleMLP(nn.Module):
    """Simple MLP - the baseline that works"""
    def __init__(self, in_dim, hidden=[256, 128, 64], out_dim=7, dropout=0.3):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


# ============ DEEP MLP WITH RESIDUAL ============
class DeepResMLP(nn.Module):
    """Deep MLP with residual connections"""
    def __init__(self, in_dim, hidden=256, n_blocks=3, out_dim=7, dropout=0.3):
        super().__init__()
        
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden)
            )
            for _ in range(n_blocks)
        ])
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )
        
    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = F.relu(x + block(x))
        return self.output(x)


# ============ WIDE & DEEP ============
class WideDeep(nn.Module):
    """Wide & Deep with interaction features"""
    def __init__(self, in_dim, deep_dims=[256, 128, 64], out_dim=7, dropout=0.3):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(in_dim)
        
        # Wide: linear
        self.wide = nn.Linear(in_dim, out_dim)
        
        # Deep: MLP with GELU
        deep = []
        prev = in_dim
        for dim in deep_dims:
            deep += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        deep.append(nn.Linear(prev, out_dim))
        self.deep = nn.Sequential(*deep)
        
    def forward(self, x):
        x = self.bn(x)
        return 0.5 * self.wide(x) + 0.5 * self.deep(x)


# ============ TRAINING ============
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
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


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    model = model.to(DEVICE)
    
    # Standard CrossEntropy (no class weighting for now - let's establish baseline)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_acc, best_state, no_improve = 0, None, 0
    
    print(f"{'Ep':>3} {'TrAcc':>7} {'VlAcc':>7} {'LR':>10}")
    print("-" * 32)
    
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_acc = evaluate(model, val_loader)
        
        scheduler.step(vl_acc)
        lr_now = optimizer.param_groups[0]['lr']
        
        if (ep + 1) % 10 == 0 or ep < 10:
            print(f"{ep+1:>3} {tr_acc:>7.4f} {vl_acc:>7.4f} {lr_now:>10.6f}")
        
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"   Early stopping at epoch {ep+1}")
            break
    
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
    print("🎯 FIXED SOTA Training (No Weighted Sampler)")
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
    
    # Standard data loaders - NO weighted sampler
    train_loader = DataLoader(train_ds, 256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, 512, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 512, num_workers=4, pin_memory=True)
    
    results = []
    best_model, best_name, best_acc = None, None, 0
    
    # ===== 1. Simple MLP (baseline) =====
    print("\n" + "=" * 60)
    print("🔷 Model 1: Simple MLP [256, 128, 64]")
    print("=" * 60)
    
    mlp = SimpleMLP(NUM_FEATURES, [256, 128, 64], NUM_CLASSES, dropout=0.3)
    print(f"   Params: {sum(p.numel() for p in mlp.parameters()):,}")
    
    t0 = time.time()
    mlp, _ = train_model(mlp, train_loader, val_loader, epochs=100, lr=0.001, patience=20)
    test_acc = evaluate(mlp, test_loader)
    t = time.time() - t0
    
    results.append(('Simple MLP', test_acc, t))
    print(f"   ✅ Test: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, mlp, 'Simple MLP'
    
    # ===== 2. Deep MLP =====
    print("\n" + "=" * 60)
    print("🔷 Model 2: Deep MLP [512, 256, 128, 64]")
    print("=" * 60)
    
    deep = SimpleMLP(NUM_FEATURES, [512, 256, 128, 64], NUM_CLASSES, dropout=0.4)
    print(f"   Params: {sum(p.numel() for p in deep.parameters()):,}")
    
    t0 = time.time()
    deep, _ = train_model(deep, train_loader, val_loader, epochs=100, lr=0.0005, patience=20)
    test_acc = evaluate(deep, test_loader)
    t = time.time() - t0
    
    results.append(('Deep MLP', test_acc, t))
    print(f"   ✅ Test: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, deep, 'Deep MLP'
    
    # ===== 3. Residual MLP =====
    print("\n" + "=" * 60)
    print("🔷 Model 3: Residual MLP (3 blocks)")
    print("=" * 60)
    
    res = DeepResMLP(NUM_FEATURES, hidden=256, n_blocks=3, out_dim=NUM_CLASSES, dropout=0.35)
    print(f"   Params: {sum(p.numel() for p in res.parameters()):,}")
    
    t0 = time.time()
    res, _ = train_model(res, train_loader, val_loader, epochs=100, lr=0.001, patience=20)
    test_acc = evaluate(res, test_loader)
    t = time.time() - t0
    
    results.append(('Residual MLP', test_acc, t))
    print(f"   ✅ Test: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, res, 'Residual MLP'
    
    # ===== 4. Wide & Deep =====
    print("\n" + "=" * 60)
    print("🔷 Model 4: Wide & Deep")
    print("=" * 60)
    
    wd = WideDeep(NUM_FEATURES, [256, 128, 64], NUM_CLASSES, dropout=0.35)
    print(f"   Params: {sum(p.numel() for p in wd.parameters()):,}")
    
    t0 = time.time()
    wd, _ = train_model(wd, train_loader, val_loader, epochs=100, lr=0.001, patience=20)
    test_acc = evaluate(wd, test_loader)
    t = time.time() - t0
    
    results.append(('Wide & Deep', test_acc, t))
    print(f"   ✅ Test: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, wd, 'Wide & Deep'
    
    # ===== RESULTS =====
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    results.append(('LightGBM (baseline)', 0.413, 0))
    results.append(('Previous Deep MLP', 0.4144, 0))
    
    print(f"\n{'Model':<20} {'Test Acc':>10}")
    print("-" * 32)
    for name, acc, t in sorted(results, key=lambda x: -x[1]):
        mark = " 🏆" if acc == best_acc else ""
        print(f"{name:<20} {acc:>10.4f}{mark}")
    
    improvement = (best_acc - 0.413) / 0.413 * 100
    print(f"\n📈 vs LightGBM: {improvement:+.2f}%")
    
    # Export
    lookup = create_lookup(best_model, mean, std)
    model_data = {
        'model_type': best_name.lower().replace(' ', '_').replace('&', 'and'),
        'n_classes': NUM_CLASSES,
        'classes': OUTCOME_CLASSES,
        'features': FEATURE_COLS,
        'test_accuracy': best_acc,
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
    print(f"\n✅ Exported {outpath.name}")
    
    if best_acc > 0.4144:
        import shutil
        shutil.copy(outpath, ML_DIR / 'ball_outcome_model.json')
        print(f"🏆 NEW SOTA! {best_name}: {best_acc:.4f}")
    else:
        print(f"📊 Previous best still holds (0.4144 vs {best_acc:.4f})")
    
    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()
