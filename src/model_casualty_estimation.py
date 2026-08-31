import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def train_casualty_estimation(train_path='data/train_gtd.pkl', test_path='data/test_gtd.pkl'):
    print("\n=== Training Casualty Severity & Quantile Risk Model ===")
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

    X_train, y_tier_train = train_df[feature_cols], train_df['casualty_tier']
    X_test, y_tier_test = test_df[feature_cols], test_df['casualty_tier']

    y_cas_train = train_df['total_casualty']
    y_cas_test = test_df['total_casualty']

    print(f"Training set size: {len(X_train)} rows")
    print(f"Test set size: {len(X_test)} rows")

    # 1. Multi-Class Ordinal Tier Classifier
    print("Training Multi-Class Casualty Tier Classifier...")
    clf_tier = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf_tier.fit(X_train, y_tier_train)
    
    y_pred_tier = clf_tier.predict(X_test)
    tier_acc = accuracy_score(y_tier_test, y_pred_tier)
    tier_macro_f1 = f1_score(y_tier_test, y_pred_tier, average='macro')
    tier_weighted_f1 = f1_score(y_tier_test, y_pred_tier, average='weighted')

    print(f"Casualty Tier Accuracy: {tier_acc:.4f}")
    print(f"Casualty Tier Macro F1: {tier_macro_f1:.4f}")
    print(f"Casualty Tier Weighted F1: {tier_weighted_f1:.4f}")

    # 2. Quantile Regression (50th percentile - Median, 90th percentile - High-Risk Upper Bound)
    print("Training Quantile Regressors (q=0.50, q=0.90)...")
    reg_q50 = LGBMRegressor(objective='quantile', alpha=0.50, n_estimators=100, learning_rate=0.08, random_state=42, verbose=-1)
    reg_q90 = LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=100, learning_rate=0.08, random_state=42, verbose=-1)

    reg_q50.fit(X_train, y_cas_train)
    reg_q90.fit(X_train, y_cas_train)

    pred_q50 = reg_q50.predict(X_test)
    pred_q90 = reg_q90.predict(X_test)

    def pinball_loss(y_true, y_pred, alpha):
        err = y_true - y_pred
        return np.maximum(alpha * err, (alpha - 1) * err).mean()

    p_loss_q50 = pinball_loss(y_cas_test, pred_q50, 0.50)
    p_loss_q90 = pinball_loss(y_cas_test, pred_q90, 0.90)
    mae_q50 = mean_absolute_error(y_cas_test, pred_q50)

    print(f"q=0.50 Pinball Loss: {p_loss_q50:.4f} (MAE: {mae_q50:.4f})")
    print(f"q=0.90 Pinball Loss: {p_loss_q90:.4f}")

    # Feature importances for tier classifier
    importances = pd.Series(clf_tier.feature_importances_, index=feature_cols).sort_values(ascending=False).to_dict()

    # Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'clf_tier': clf_tier,
        'reg_q50': reg_q50,
        'reg_q90': reg_q90,
        'features': feature_cols
    }, 'models/casualty_estimation_lgb.joblib')

    metrics = {
        'casualty_tier_accuracy': float(tier_acc),
        'casualty_tier_macro_f1': float(tier_macro_f1),
        'casualty_tier_weighted_f1': float(tier_weighted_f1),
        'pinball_loss_q50': float(p_loss_q50),
        'pinball_loss_q90': float(p_loss_q90),
        'mae_q50': float(mae_q50),
        'feature_importances': importances
    }

    return clf_tier, metrics

if __name__ == '__main__':
    train_casualty_estimation()
