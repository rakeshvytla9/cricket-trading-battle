#!/usr/bin/env python3
"""
Advanced Model Training - Boosting & Bagging Ensemble
Trains XGBoost, LightGBM, and Random Forest models
Compares performance and exports the best model
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter
import time

# Paths
ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR.parent / "data"

# Outcome classes
OUTCOME_CLASSES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET']

# Features
FEATURE_COLS = [
    'over', 'ball_in_over', 'innings', 'phase_powerplay', 'phase_middle', 'phase_death',
    'runs_scored', 'wickets_lost', 'run_rate', 'required_rate', 'balls_remaining',
    'batsman_sr', 'batsman_avg', 'batsman_experience',
    'bowler_economy', 'bowler_sr', 'bowler_experience',
    'matchup_sr', 'matchup_balls', 'venue_avg_score', 'recent_form'
]

def load_data(filename: str):
    """Load training data from CSV"""
    X = []
    y = []
    
    with open(ML_DIR / filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[col]) for col in FEATURE_COLS]
            X.append(features)
            y.append(int(row['outcome_idx']))
    
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred, y_proba, name):
    """Evaluate model performance"""
    accuracy = np.mean(y_pred == y_true)
    
    # Log loss
    eps = 1e-10
    log_loss = -np.mean(np.log(y_proba[np.arange(len(y_true)), y_true] + eps))
    
    print(f"\n{name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Log Loss: {log_loss:.4f}")
    
    return accuracy, log_loss

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n" + "="*50)
    print("Training XGBoost (Gradient Boosting)")
    print("="*50)
    
    try:
        import xgboost as xgb
        
        start = time.time()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=7,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        train_time = time.time() - start
        print(f"   Training time: {train_time:.2f}s")
        
        return model, 'xgboost'
        
    except ImportError:
        print("   XGBoost not available")
        return None, None

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    print("\n" + "="*50)
    print("Training LightGBM (Gradient Boosting)")
    print("="*50)
    
    try:
        import lightgbm as lgb
        
        start = time.time()
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            num_class=7,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        train_time = time.time() - start
        print(f"   Training time: {train_time:.2f}s")
        
        return model, 'lightgbm'
        
    except ImportError:
        print("   LightGBM not available")
        return None, None

def train_random_forest(X_train, y_train):
    """Train Random Forest model (Bagging)"""
    print("\n" + "="*50)
    print("Training Random Forest (Bagging)")
    print("="*50)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        start = time.time()
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start
        print(f"   Training time: {train_time:.2f}s")
        
        return model, 'random_forest'
        
    except ImportError:
        print("   sklearn not available")
        return None, None

def export_model_to_json(model, model_type, filepath):
    """Export model to JSON for JavaScript"""
    
    if model_type == 'xgboost':
        # XGBoost can dump to JSON
        model.get_booster().save_model(str(filepath).replace('.json', '_xgb.json'))
        
        # Also create simplified version for our JS predictor
        export_sklearn_like_model(model, filepath)
        
    elif model_type == 'lightgbm':
        export_sklearn_like_model(model, filepath)
        
    elif model_type == 'random_forest':
        export_sklearn_like_model(model, filepath)

def export_sklearn_like_model(model, filepath):
    """Export sklearn-compatible model to simplified JSON"""
    
    # Get base probabilities from training
    n_classes = 7
    
    # For tree-based models, we'll export the tree structure
    # But for JS compatibility, we'll use a simplified approach:
    # Store the predicted probabilities for key feature combinations
    
    model_data = {
        'type': 'lookup_with_model',
        'n_classes': n_classes,
        'classes': OUTCOME_CLASSES,
        'features': FEATURE_COLS,
        'base_probs': None,  # Will be filled
        'feature_importance': None
    }
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_.tolist()
        model_data['feature_importance'] = dict(zip(FEATURE_COLS, importance))
    
    # We'll create a lookup table for common scenarios
    # This is more compatible with simple JS than full tree traversal
    
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"   Model metadata exported to {filepath}")

def create_probability_lookup(model, X_train, y_train):
    """Create probability lookup table for common game scenarios"""
    
    # Key scenarios to pre-compute
    scenarios = []
    
    # Different phases and situations
    for over in [0, 3, 6, 10, 15, 18]:  # Sample overs
        for wickets in [0, 2, 4, 6]:
            for runs in [0, 40, 80, 120, 160]:
                # Create feature vector
                ball = 3  # Mid-over
                phase_pp = 1 if over < 6 else 0
                phase_mid = 1 if 6 <= over < 15 else 0
                phase_death = 1 if over >= 15 else 0
                balls_rem = (20 - over) * 6
                rr = runs / (over * 6 + ball) * 6 if over > 0 else 0
                
                features = [
                    over, ball, 1, phase_pp, phase_mid, phase_death,
                    runs, wickets, rr, 8.0, balls_rem,
                    130, 30, 2.0,  # avg batsman
                    8.0, 20, 2.0,  # avg bowler
                    130, 0.5, 165, 10  # matchup, venue, form
                ]
                scenarios.append(features)
    
    X_scenarios = np.array(scenarios)
    probs = model.predict_proba(X_scenarios)
    
    # Create lookup dictionary
    lookup = {}
    for i, (features, prob) in enumerate(zip(scenarios, probs)):
        key = f"{int(features[0])}_{int(features[7])}_{int(features[6])}"
        lookup[key] = prob.tolist()
    
    return lookup

def export_best_model_for_js(model, model_type, X_train, y_train, filepath):
    """Export the best model in a format usable by JavaScript"""
    
    print(f"\n   Exporting {model_type} model for JavaScript...")
    
    # Get base probabilities
    y_proba = model.predict_proba(X_train)
    base_probs = y_proba.mean(axis=0).tolist()
    
    # Create probability lookup for common scenarios
    lookup = create_probability_lookup(model, X_train, y_train)
    
    # Get feature importance
    importance = {}
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    
    # Export format that our JS can use
    model_data = {
        'model_type': model_type,
        'n_classes': 7,
        'classes': OUTCOME_CLASSES,
        'features': FEATURE_COLS,
        'base_probs': base_probs,
        'feature_importance': importance,
        'scenario_lookup': lookup,
        'phase_adjustments': {
            'powerplay': {'DOT': 0.95, 'FOUR': 1.15, 'SIX': 1.10, 'WICKET': 0.90},
            'middle': {'DOT': 1.05, 'FOUR': 0.90, 'SIX': 0.85, 'WICKET': 1.00},
            'death': {'DOT': 0.90, 'FOUR': 1.10, 'SIX': 1.20, 'WICKET': 1.15}
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(model_data, f)
    
    # Get file size
    size_kb = filepath.stat().st_size / 1024
    print(f"   ✅ Exported to {filepath.name} ({size_kb:.1f} KB)")

def main():
    print("="*60)
    print("🚀 Advanced Model Training - Boosting & Bagging")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    X_train, y_train = load_data('train_data.csv')
    X_val, y_val = load_data('val_data.csv')
    X_test, y_test = load_data('test_data.csv')
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val:   {len(X_val):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Train models
    models = []
    
    # 1. XGBoost
    xgb_model, xgb_type = train_xgboost(X_train, y_train, X_val, y_val)
    if xgb_model:
        models.append((xgb_model, xgb_type))
    
    # 2. LightGBM
    lgb_model, lgb_type = train_lightgbm(X_train, y_train, X_val, y_val)
    if lgb_model:
        models.append((lgb_model, lgb_type))
    
    # 3. Random Forest
    rf_model, rf_type = train_random_forest(X_train, y_train)
    if rf_model:
        models.append((rf_model, rf_type))
    
    # Evaluate all models
    print("\n" + "="*60)
    print("📈 Model Comparison on Test Set")
    print("="*60)
    
    best_model = None
    best_type = None
    best_accuracy = 0
    
    results = []
    
    for model, model_type in models:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        acc, loss = evaluate_model(y_test, y_pred, y_proba, model_type.upper())
        results.append((model_type, acc, loss))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_type = model_type
    
    # Summary table
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'Log Loss':>10}")
    print("-"*40)
    for name, acc, loss in sorted(results, key=lambda x: -x[1]):
        marker = " 🏆" if name == best_type else ""
        print(f"{name:<15} {acc:>10.4f} {loss:>10.4f}{marker}")
    
    # Export best model
    print(f"\n🏆 Best Model: {best_type.upper()} (Accuracy: {best_accuracy:.4f})")
    
    export_path = ML_DIR / 'ball_outcome_model_advanced.json'
    export_best_model_for_js(best_model, best_type, X_train, y_train, export_path)
    
    # Also update the main model file
    main_model_path = ML_DIR / 'ball_outcome_model.json'
    export_best_model_for_js(best_model, best_type, X_train, y_train, main_model_path)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
