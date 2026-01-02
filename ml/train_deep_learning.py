#!/usr/bin/env python3
"""
Phase 5: Deep Learning - Quick Training Script
Optimized for faster results
"""

import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# Paths
ML_DIR = Path(__file__).parent

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {DEVICE}")

OUTCOME_CLASSES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET']
FEATURE_COLS = [
    'over', 'ball_in_over', 'innings', 'phase_powerplay', 'phase_middle', 'phase_death',
    'runs_scored', 'wickets_lost', 'run_rate', 'required_rate', 'balls_remaining',
    'batsman_sr', 'batsman_avg', 'batsman_experience',
    'bowler_economy', 'bowler_sr', 'bowler_experience',
    'matchup_sr', 'matchup_balls', 'venue_avg_score', 'recent_form'
]

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

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256, 128, 64], out_dim=7, drop=0.3):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train(model, loader, crit, opt):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum/len(loader), correct/total

def evaluate(model, loader, crit):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss_sum += crit(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum/len(loader), correct/total

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
                    probs = torch.softmax(model(feat), 1).cpu().numpy()[0].tolist()
                    lookup[f"{over}_{wkts}_{runs}"] = probs
    return lookup

def main():
    print("=" * 50)
    print("🧠 Phase 5: Deep Learning Training")
    print("=" * 50)
    
    # Load data
    train_ds = CricketDataset(ML_DIR / 'train_data.csv')
    val_ds = CricketDataset(ML_DIR / 'val_data.csv')
    test_ds = CricketDataset(ML_DIR / 'test_data.csv')
    
    mean, std = train_ds.mean, train_ds.std
    train_ds.normalize()
    val_ds.normalize(mean, std)
    test_ds.normalize(mean, std)
    
    print(f"📊 Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    
    train_dl = DataLoader(train_ds, 256, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, 512, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, 512, num_workers=2, pin_memory=True)
    
    results = []
    
    # MLP
    print("\n🔷 Training MLP...")
    mlp = MLP(21, [256, 128, 64]).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    best_acc, best_state = 0, None
    t0 = time.time()
    for ep in range(30):
        tr_loss, tr_acc = train(mlp, train_dl, crit, opt)
        vl_loss, vl_acc = evaluate(mlp, val_dl, crit)
        sched.step(vl_loss)
        if vl_acc > best_acc:
            best_acc, best_state = vl_acc, mlp.state_dict().copy()
        print(f"  Ep {ep+1:2d}: Train {tr_acc:.4f} | Val {vl_acc:.4f}")
    
    mlp.load_state_dict(best_state)
    _, test_acc = evaluate(mlp, test_dl, crit)
    mlp_time = time.time() - t0
    results.append(('MLP', test_acc, mlp_time))
    print(f"  ✅ MLP Test Acc: {test_acc:.4f} ({mlp_time:.1f}s)")
    
    # Deep MLP
    print("\n🔷 Training Deep MLP...")
    deep = MLP(21, [512, 256, 128, 64], drop=0.4).to(DEVICE)
    opt = optim.AdamW(deep.parameters(), lr=0.0005, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    best_acc, best_state = 0, None
    t0 = time.time()
    for ep in range(30):
        tr_loss, tr_acc = train(deep, train_dl, crit, opt)
        vl_loss, vl_acc = evaluate(deep, val_dl, crit)
        sched.step(vl_loss)
        if vl_acc > best_acc:
            best_acc, best_state = vl_acc, deep.state_dict().copy()
        print(f"  Ep {ep+1:2d}: Train {tr_acc:.4f} | Val {vl_acc:.4f}")
    
    deep.load_state_dict(best_state)
    _, test_acc = evaluate(deep, test_dl, crit)
    deep_time = time.time() - t0
    results.append(('Deep MLP', test_acc, deep_time))
    print(f"  ✅ Deep MLP Test Acc: {test_acc:.4f} ({deep_time:.1f}s)")
    
    # Results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    results.append(('LightGBM (baseline)', 0.413, 0))
    
    best = max(results, key=lambda x: x[1])
    for name, acc, t in sorted(results, key=lambda x: -x[1]):
        mark = " 🏆" if name == best[0] else ""
        print(f"  {name:<20} {acc:.4f}{mark}")
    
    # Export best DL model
    best_dl = max([r for r in results if r[0] != 'LightGBM (baseline)'], key=lambda x: x[1])
    if best_dl[0] == 'MLP':
        model = mlp
    else:
        model = deep
    
    lookup = create_lookup(model, mean, std)
    model_data = {
        'model_type': best_dl[0].lower().replace(' ', '_'),
        'n_classes': 7,
        'classes': OUTCOME_CLASSES,
        'features': FEATURE_COLS,
        'test_accuracy': best_dl[1],
        'scenario_lookup': lookup,
        'phase_adjustments': {
            'powerplay': {'DOT': 0.95, 'FOUR': 1.15, 'SIX': 1.10, 'WICKET': 0.90},
            'middle': {'DOT': 1.05, 'FOUR': 0.90, 'SIX': 0.85, 'WICKET': 1.00},
            'death': {'DOT': 0.90, 'FOUR': 1.10, 'SIX': 1.20, 'WICKET': 1.15}
        }
    }
    
    outpath = ML_DIR / f'{best_dl[0].lower().replace(" ", "_")}_model.json'
    with open(outpath, 'w') as f:
        json.dump(model_data, f)
    print(f"\n✅ Exported {outpath.name} ({outpath.stat().st_size/1024:.1f} KB)")
    
    # Update main model if DL beats LightGBM
    if best_dl[1] > 0.413:
        import shutil
        shutil.copy(outpath, ML_DIR / 'ball_outcome_model.json')
        print(f"🏆 DL beats LightGBM! Updated ball_outcome_model.json")
    else:
        print(f"📊 LightGBM still best ({0.413:.4f} vs {best_dl[1]:.4f})")
    
    print("\n✅ Phase 5 Complete!")

if __name__ == "__main__":
    main()
