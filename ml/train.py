"""
Placeholder ML training pipeline.

This module demonstrates the structure for:
1. Loading 17k-row job posting dataset
2. Splitting by train/val/test ratios (e.g., 70/15/15)
3. Preprocessing text data
4. Training four models (Logistic Regression, Random Forest, CNN, LSTM)
5. Computing confusion matrix, ROC, AUC, metrics
6. Exporting artifacts (models, scalers, plots)

IMPLEMENTATION STEPS:
1. Load dataset: 
   - Place your CSV at `ml/data/jobs_dataset.csv`
   - Expected columns: title, description, location, label (0 or 1)

2. Preprocess:
   - Text cleaning (lowercase, remove HTML, tokenize)
   - For sklearn: TfidfVectorizer
   - For Keras: Tokenizer + Padding

3. Train models:
   - Logistic Regression with TF-IDF
   - Random Forest with TF-IDF
   - CNN with embeddings
   - LSTM (bidirectional) with embeddings

4. Export:
   - Save models with joblib (sklearn) or .h5 (Keras)
   - Export metrics as JSON
   - Save confusion matrix plots and ROC curves

5. Move artifacts to:
   - ml/models/
   - ml/metrics/
   - ml/plots/
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'ml' / 'data'
MODELS_DIR = BASE_DIR / 'ml' / 'models'
METRICS_DIR = BASE_DIR / 'ml' / 'metrics'
PLOTS_DIR = BASE_DIR / 'ml' / 'plots'

# Create dirs
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: str = None) -> tuple:
    """
    Load job posting dataset.
    Expected CSV columns: title, description, location, label (0=legit, 1=fraud)
    """
    if csv_path is None:
        csv_path = DATA_DIR / 'jobs_dataset.csv'
    
    if not Path(csv_path).exists():
        print(f"Dataset not found at {csv_path}")
        print("Returning dummy dataset for demo purposes.")
        # Return dummy data for testing
        data = {
            'title': ['Software Engineer'] * 100 + ['Data Entry'] * 100,
            'description': ['Build scalable systems'] * 100 + ['Quick money online'] * 100,
            'location': ['Remote'] * 200,
            'label': [0] * 100 + [1] * 100,
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(csv_path)
    
    return df


def preprocess_text(df: pd.DataFrame) -> tuple:
    """
    Combine title, description, location into single text field.
    Return vectorized features and labels.
    """
    df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['location'].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['label'].values
    
    return X, y, vectorizer


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred_val = model.predict(X_val)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)
    
    auc_score = auc(y_test, np.argsort(np.argsort(-y_pred_proba_test)) / len(y_test))
    f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    
    metrics = {
        'model_name': 'Logistic Regression',
        'auc': float(auc_score),
        'f1': float(f1),
        'accuracy': float((y_pred_test == y_test).mean()),
        'confusion_matrix': cm.tolist(),
    }
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Train Random Forest model (placeholder; requires scikit-learn)."""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    auc_score = auc(y_test, np.argsort(np.argsort(-y_pred_proba_test)) / len(y_test))
    f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    
    metrics = {
        'model_name': 'Random Forest',
        'auc': float(auc_score),
        'f1': float(f1),
        'accuracy': float((y_pred_test == y_test).mean()),
        'confusion_matrix': cm.tolist(),
    }
    
    return model, metrics


def main():
    """Main training pipeline."""
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset size: {len(df)} rows")
    
    print("Preprocessing...")
    X, y, vectorizer = preprocess_text(df)
    
    # Split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    joblib.dump(lr_model, MODELS_DIR / 'logistic_regression.pkl')
    joblib.dump(vectorizer, MODELS_DIR / 'vectorizer.pkl')
    with open(METRICS_DIR / 'logistic_regression.json', 'w') as f:
        json.dump(lr_metrics, f)
    print(f"  AUC: {lr_metrics['auc']:.3f}, F1: {lr_metrics['f1']:.3f}")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    joblib.dump(rf_model, MODELS_DIR / 'random_forest.pkl')
    with open(METRICS_DIR / 'random_forest.json', 'w') as f:
        json.dump(rf_metrics, f)
    print(f"  AUC: {rf_metrics['auc']:.3f}, F1: {rf_metrics['f1']:.3f}")
    
    print("\nNote: CNN and LSTM training requires additional Keras/TensorFlow setup.")
    print("Placeholder metrics are hardcoded in the views for now.")
    print("\nAll artifacts saved to ml/models/ and ml/metrics/")


if __name__ == '__main__':
    main()
