import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    classification_report, accuracy_score
)

def train_success_prediction(train_path='data/train_gtd.pkl', test_path='data/test_gtd.pkl'):
    print("\n=== Training Attack Success Prediction Model ===")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    feature_cols = [
        'iyear', 'imonth_clean', 'iday_clean', 'quarter',
        'region', 'country', 'region_txt_code', 'country_txt_code',
        'latitude_clean', 'longitude_clean',
        'attacktype1', 'attacktype1_txt_code',
        'targtype1', 'targtype1_txt_code',
        'weaptype1', 'weaptype1_txt_code',
        'extended', 'vicinity', 'suicide', 'property', 'ishostkid',
        'nperps_missing', 'claimed_missing', 'claimed_clean', 'is_known_group'
    ]

    X_train, y_train = train_df[feature_cols], train_df['success']
    X_test, y_test = test_df[feature_cols], test_df['success']

    # Compute scale_pos_weight (negative count / positive count)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"Training set size: {len(X_train)} (Success: {pos_count}, Failure: {neg_count})")
    print(f"Test set size: {len(X_test)}")

    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"Failure Detection Recall (Class 0): {(y_pred[y_test == 0] == 0).mean():.4f}")

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).to_dict()

    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': clf, 'features': feature_cols}, 'models/success_prediction_lgb.joblib')

    metrics = {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'feature_importances': importances
    }
    
    return clf, metrics

if __name__ == '__main__':
    train_success_prediction()
