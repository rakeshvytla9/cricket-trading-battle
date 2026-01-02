#!/usr/bin/env python3
"""
Train Ball Outcome Prediction Model
Uses gradient boosting to predict probability of each ball outcome
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter

# Paths
ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR.parent / "data"

# Outcome classes
OUTCOME_CLASSES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET']

# Features to use for training
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

def softmax(x):
    """Compute softmax probabilities"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleGradientBoostingClassifier:
    """
    A simple gradient boosting classifier that can be exported to JavaScript.
    Uses decision stumps (single-split trees) for interpretability and easy JS export.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []  # List of trees for each class
        self.n_classes = len(OUTCOME_CLASSES)
        self.base_probs = None
    
    def fit(self, X, y):
        """Train the model"""
        n_samples = len(y)
        
        # Initialize with class probabilities
        class_counts = Counter(y)
        self.base_probs = np.array([class_counts.get(i, 1) / n_samples for i in range(self.n_classes)])
        
        # Current predictions (log-odds)
        F = np.log(self.base_probs + 1e-10)[np.newaxis, :].repeat(n_samples, axis=0)
        
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        for iteration in range(self.n_estimators):
            # Compute probabilities
            probs = softmax(F)
            
            # Compute residuals (negative gradient)
            residuals = y_onehot - probs
            
            # Fit a tree for each class
            iteration_trees = []
            for c in range(self.n_classes):
                tree = self._fit_tree(X, residuals[:, c])
                iteration_trees.append(tree)
                
                # Update predictions
                predictions = self._predict_tree(X, tree)
                F[:, c] += self.learning_rate * predictions
            
            self.trees.append(iteration_trees)
            
            if (iteration + 1) % 20 == 0:
                probs = softmax(F)
                preds = np.argmax(probs, axis=1)
                acc = np.mean(preds == y)
                print(f"   Iteration {iteration + 1}/{self.n_estimators}, Accuracy: {acc:.4f}")
        
        return self
    
    def _fit_tree(self, X, residuals):
        """Fit a simple decision tree (stump or shallow tree)"""
        return self._build_tree(X, residuals, depth=0)
    
    def _build_tree(self, X, residuals, depth):
        """Recursively build a decision tree"""
        n_samples, n_features = X.shape
        
        # Base case: max depth or too few samples
        if depth >= self.max_depth or n_samples < 10:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        best_gain = -float('inf')
        best_feature = 0
        best_threshold = 0
        best_left_indices = None
        best_right_indices = None
        
        # Find best split
        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            thresholds = np.percentile(values, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                
                left_mean = np.mean(residuals[left_mask])
                right_mean = np.mean(residuals[right_mask])
                
                # Compute gain (reduction in variance)
                left_var = np.var(residuals[left_mask]) * np.sum(left_mask)
                right_var = np.var(residuals[right_mask]) * np.sum(right_mask)
                total_var = np.var(residuals) * n_samples
                
                gain = total_var - left_var - right_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_mask
                    best_right_indices = right_mask
        
        if best_left_indices is None:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        # Recursively build children
        left_child = self._build_tree(X[best_left_indices], residuals[best_left_indices], depth + 1)
        right_child = self._build_tree(X[best_right_indices], residuals[best_right_indices], depth + 1)
        
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': float(best_threshold),
            'left': left_child,
            'right': right_child
        }
    
    def _predict_tree(self, X, tree):
        """Predict using a single tree"""
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            predictions[i] = self._predict_single(X[i], tree)
        return predictions
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        n_samples = len(X)
        F = np.log(self.base_probs + 1e-10)[np.newaxis, :].repeat(n_samples, axis=0)
        
        for iteration_trees in self.trees:
            for c, tree in enumerate(iteration_trees):
                predictions = self._predict_tree(X, tree)
                F[:, c] += self.learning_rate * predictions
        
        return softmax(F)
    
    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def export_to_json(self, filepath):
        """Export model to JSON for JavaScript"""
        model_data = {
            'n_classes': self.n_classes,
            'classes': OUTCOME_CLASSES,
            'features': FEATURE_COLS,
            'base_probs': self.base_probs.tolist(),
            'learning_rate': self.learning_rate,
            'trees': self.trees
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f)
        
        print(f"   Model exported to {filepath}")

def evaluate_model(model, X, y, name=""):
    """Evaluate model performance"""
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    
    accuracy = np.mean(preds == y)
    
    # Compute log loss
    eps = 1e-10
    log_loss = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
    
    print(f"\n{name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Log Loss: {log_loss:.4f}")
    
    # Per-class accuracy
    print(f"\n   Per-class accuracy:")
    for i, outcome in enumerate(OUTCOME_CLASSES):
        mask = y == i
        if np.sum(mask) > 0:
            class_acc = np.mean(preds[mask] == y[mask])
            class_count = np.sum(mask)
            print(f"   {outcome:8}: {class_acc:.3f} ({class_count} samples)")
    
    return accuracy, log_loss

def main():
    print("=" * 60)
    print("Training Ball Outcome Prediction Model")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading training data...")
    X_train, y_train = load_data('train_data.csv')
    X_val, y_val = load_data('val_data.csv')
    X_test, y_test = load_data('test_data.csv')
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Check class distribution
    print("\n2. Class distribution in training data:")
    for i, outcome in enumerate(OUTCOME_CLASSES):
        count = np.sum(y_train == i)
        pct = count / len(y_train) * 100
        print(f"   {outcome:8}: {count:5} ({pct:5.2f}%)")
    
    # Train model
    print("\n3. Training gradient boosting model...")
    model = SimpleGradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n4. Evaluating model...")
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    test_acc, test_loss = evaluate_model(model, X_test, y_test, "Test")
    
    # Export model
    print("\n5. Exporting model...")
    model.export_to_json(ML_DIR / 'ball_outcome_model.json')
    
    # Test prediction
    print("\n6. Sample predictions:")
    sample_X = X_test[:5]
    sample_probs = model.predict_proba(sample_X)
    
    for i, probs in enumerate(sample_probs):
        actual = OUTCOME_CLASSES[y_test[i]]
        predicted = OUTCOME_CLASSES[np.argmax(probs)]
        print(f"   Sample {i+1}: Actual={actual:8}, Predicted={predicted:8}")
        print(f"            Probs: " + " ".join([f"{OUTCOME_CLASSES[j][:3]}:{p:.2f}" for j, p in enumerate(probs)]))
    
    print("\n" + "=" * 60)
    print(f"✅ Model trained! Test Accuracy: {test_acc:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
