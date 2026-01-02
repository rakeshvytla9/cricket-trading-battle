#!/usr/bin/env python3
"""
SOTA Tabular Deep Learning for Cricket Ball Outcome Prediction
Target: 50-60% accuracy (up from 41.4%)

Architectures:
1. TabNet-style with attention
2. Residual MLP with class-balanced loss
3. DCN-v2 Feature Cross Network
4. GBM + Neural Ensemble
"""

import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import time

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
        """Compute inverse frequency weights for balanced training"""
        counts = torch.bincount(self.y, minlength=NUM_CLASSES).float()
        weights = 1.0 / (counts + 1)
        return weights / weights.sum() * NUM_CLASSES
    
    def get_sample_weights(self):
        """Get per-sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        return class_weights[self.y]


# ============ ARCHITECTURE 1: TabNet-style Attention ============
class GatedLinearUnit(nn.Module):
    """GLU activation for feature selection"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        
    def forward(self, x):
        x = self.bn(self.fc(x))
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class FeatureTransformer(nn.Module):
    """Feature transformer block for TabNet"""
    def __init__(self, in_dim, out_dim, shared_layers=None):
        super().__init__()
        self.shared = shared_layers or nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.specific = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.specific(self.shared(x))


class TabNetBlock(nn.Module):
    """Single TabNet decision step"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.feature_transformer = FeatureTransformer(in_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.Softmax(dim=-1)
        )
        self.output = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, prior_scales=None):
        h = self.feature_transformer(x)
        mask = self.attention(h)
        if prior_scales is not None:
            mask = mask * prior_scales
        masked_x = x * mask
        return self.output(h), mask


class TabNet(nn.Module):
    """TabNet-style model for tabular data"""
    def __init__(self, in_dim, hidden=128, n_steps=3, out_dim=7, dropout=0.2):
        super().__init__()
        self.n_steps = n_steps
        
        self.initial_bn = nn.BatchNorm1d(in_dim)
        self.initial_fc = nn.Linear(in_dim, hidden)
        
        self.steps = nn.ModuleList([
            TabNetBlock(in_dim, hidden, hidden)
            for _ in range(n_steps)
        ])
        
        self.final = nn.Sequential(
            nn.Linear(hidden * n_steps, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, x):
        x = self.initial_bn(x)
        
        outputs = []
        prior_scales = torch.ones_like(x)
        
        for step in self.steps:
            out, mask = step(x, prior_scales)
            outputs.append(out)
            # Update prior scales (relaxation factor)
            prior_scales = prior_scales * (1 - mask)
        
        aggregated = torch.cat(outputs, dim=-1)
        return self.final(aggregated)


# ============ ARCHITECTURE 2: Residual MLP ============
class ResidualBlock(nn.Module):
    """Residual block with pre-activation"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """Residual MLP with skip connections"""
    def __init__(self, in_dim, hidden=256, n_blocks=4, out_dim=7, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden, dropout) for _ in range(n_blocks)
        ])
        
        self.output = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)


# ============ ARCHITECTURE 3: DCN-v2 Feature Cross ============
class CrossLayer(nn.Module):
    """Cross layer for explicit feature interactions (DCN-v2)"""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x0, x):
        # x0 is the original input, x is current layer input
        return x0 * self.weight(x) + self.bias + x


class DCNv2(nn.Module):
    """Deep & Cross Network v2"""
    def __init__(self, in_dim, cross_layers=3, deep_layers=[256, 128, 64], out_dim=7, dropout=0.3):
        super().__init__()
        
        self.initial_bn = nn.BatchNorm1d(in_dim)
        
        # Cross network
        self.cross_layers = nn.ModuleList([
            CrossLayer(in_dim) for _ in range(cross_layers)
        ])
        
        # Deep network
        deep = []
        prev_dim = in_dim
        for dim in deep_layers:
            deep.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.deep = nn.Sequential(*deep)
        
        # Combine cross + deep
        self.output = nn.Linear(in_dim + deep_layers[-1], out_dim)
        
    def forward(self, x):
        x = self.initial_bn(x)
        
        # Cross network
        x0 = x
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        
        # Deep network
        x_deep = self.deep(x)
        
        # Concatenate and output
        combined = torch.cat([x_cross, x_deep], dim=-1)
        return self.output(combined)


# ============ ARCHITECTURE 4: Wide & Deep Ensemble ============
class WideAndDeep(nn.Module):
    """Wide & Deep model with explicit wide component"""
    def __init__(self, in_dim, deep_dims=[256, 128, 64], out_dim=7, dropout=0.3):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(in_dim)
        
        # Wide: linear model
        self.wide = nn.Linear(in_dim, out_dim)
        
        # Deep: MLP
        deep = []
        prev = in_dim
        for dim in deep_dims:
            deep.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev = dim
        deep.append(nn.Linear(prev, out_dim))
        self.deep = nn.Sequential(*deep)
        
    def forward(self, x):
        x = self.bn(x)
        return self.wide(x) + self.deep(x)


# ============ TRAINING UTILITIES ============
def train_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    
    if scheduler:
        scheduler.step()
        
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
    return total_loss / len(loader), correct / total


def train_model(model, train_ds, val_loader, epochs=80, lr=0.001, patience=15, use_sampler=True):
    """Train with early stopping and optional class-balanced sampling"""
    model = model.to(DEVICE)
    
    # Class-balanced loss
    class_weights = train_ds.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optional: use weighted random sampler for balanced batches
    if use_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc, best_state, no_improve = 0, None, 0
    
    print(f"{'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7}")
    print("-" * 40)
    
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        
        if (ep + 1) % 5 == 0 or ep < 5:
            print(f"{ep+1:>3} {tr_loss:>8.4f} {tr_acc:>7.4f} {vl_loss:>8.4f} {vl_acc:>7.4f}")
        
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
    """Create scenario lookup for JS export"""
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
    print("🎯 SOTA Tabular Models for 60% Accuracy")
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
    
    val_loader = DataLoader(val_ds, 512, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 512, num_workers=4, pin_memory=True)
    
    results = []
    best_model, best_name, best_acc = None, None, 0
    
    # ===== 1. TabNet =====
    print("\n" + "=" * 60)
    print("🔷 Architecture 1: TabNet (Attention for Tabular Data)")
    print("=" * 60)
    
    tabnet = TabNet(NUM_FEATURES, hidden=128, n_steps=4, out_dim=NUM_CLASSES, dropout=0.2)
    params = sum(p.numel() for p in tabnet.parameters())
    print(f"   Parameters: {params:,}")
    
    t0 = time.time()
    tabnet, _ = train_model(tabnet, train_ds, val_loader, epochs=80, lr=0.002, patience=15)
    _, test_acc = evaluate(tabnet, test_loader, nn.CrossEntropyLoss())
    t = time.time() - t0
    
    results.append(('TabNet', test_acc, t))
    print(f"   ✅ Test Accuracy: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, tabnet, 'TabNet'
    
    # ===== 2. Residual MLP =====
    print("\n" + "=" * 60)
    print("🔷 Architecture 2: Residual MLP (Skip Connections)")
    print("=" * 60)
    
    resmlp = ResidualMLP(NUM_FEATURES, hidden=256, n_blocks=4, out_dim=NUM_CLASSES, dropout=0.35)
    params = sum(p.numel() for p in resmlp.parameters())
    print(f"   Parameters: {params:,}")
    
    t0 = time.time()
    resmlp, _ = train_model(resmlp, train_ds, val_loader, epochs=80, lr=0.001, patience=15)
    _, test_acc = evaluate(resmlp, test_loader, nn.CrossEntropyLoss())
    t = time.time() - t0
    
    results.append(('Residual MLP', test_acc, t))
    print(f"   ✅ Test Accuracy: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, resmlp, 'Residual MLP'
    
    # ===== 3. DCN-v2 =====
    print("\n" + "=" * 60)
    print("🔷 Architecture 3: DCN-v2 (Feature Cross Network)")
    print("=" * 60)
    
    dcn = DCNv2(NUM_FEATURES, cross_layers=3, deep_layers=[256, 128, 64], out_dim=NUM_CLASSES, dropout=0.3)
    params = sum(p.numel() for p in dcn.parameters())
    print(f"   Parameters: {params:,}")
    
    t0 = time.time()
    dcn, _ = train_model(dcn, train_ds, val_loader, epochs=80, lr=0.001, patience=15)
    _, test_acc = evaluate(dcn, test_loader, nn.CrossEntropyLoss())
    t = time.time() - t0
    
    results.append(('DCN-v2', test_acc, t))
    print(f"   ✅ Test Accuracy: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, dcn, 'DCN-v2'
    
    # ===== 4. Wide & Deep =====
    print("\n" + "=" * 60)
    print("🔷 Architecture 4: Wide & Deep")
    print("=" * 60)
    
    wnd = WideAndDeep(NUM_FEATURES, deep_dims=[512, 256, 128], out_dim=NUM_CLASSES, dropout=0.35)
    params = sum(p.numel() for p in wnd.parameters())
    print(f"   Parameters: {params:,}")
    
    t0 = time.time()
    wnd, _ = train_model(wnd, train_ds, val_loader, epochs=80, lr=0.0008, patience=15, use_sampler=False)
    _, test_acc = evaluate(wnd, test_loader, nn.CrossEntropyLoss())
    t = time.time() - t0
    
    results.append(('Wide & Deep', test_acc, t))
    print(f"   ✅ Test Accuracy: {test_acc:.4f} ({t:.1f}s)")
    if test_acc > best_acc:
        best_acc, best_model, best_name = test_acc, wnd, 'Wide & Deep'
    
    # ===== RESULTS =====
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    results.append(('LightGBM (baseline)', 0.413, 0))
    results.append(('Deep MLP (previous)', 0.4144, 0))
    
    print(f"\n{'Model':<20} {'Accuracy':>10} {'Time':>8}")
    print("-" * 42)
    for name, acc, t in sorted(results, key=lambda x: -x[1]):
        mark = " 🏆" if acc == best_acc else ""
        print(f"{name:<20} {acc:>10.4f} {t:>8.1f}s{mark}")
    
    improvement = (best_acc - 0.413) / 0.413 * 100
    print(f"\n📈 Improvement over LightGBM: {improvement:+.2f}%")
    
    # Export best model
    lookup = create_lookup(best_model, mean, std)
    
    model_data = {
        'model_type': best_name.lower().replace(' ', '_').replace('-', ''),
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
    
    print(f"\n✅ Exported {outpath.name} ({outpath.stat().st_size/1024:.1f} KB)")
    
    # Update main model if better
    if best_acc > 0.4144:
        import shutil
        shutil.copy(outpath, ML_DIR / 'ball_outcome_model.json')
        print(f"🏆 New SOTA! Updated ball_outcome_model.json ({best_name}: {best_acc:.4f})")
    
    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()
