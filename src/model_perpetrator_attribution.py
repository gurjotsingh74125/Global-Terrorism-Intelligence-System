import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

def train_perpetrator_attribution(train_path='data/train_gtd.pkl', test_path='data/test_gtd.pkl'):
    print("\n=== Training Perpetrator Attribution Model ===")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    # Filter to known groups only for training and testing evaluation
    known_train = train_df[train_df['is_known_group'] == 1].copy()
    known_test = test_df[test_df['is_known_group'] == 1].copy()
    unknown_test = test_df[test_df['is_known_group'] == 0].copy()

    feature_cols = [
        'iyear', 'imonth_clean', 'iday_clean', 'quarter',
        'region', 'country', 'region_txt_code', 'country_txt_code',
        'latitude_clean', 'longitude_clean',
        'attacktype1', 'attacktype1_txt_code',
        'targtype1', 'targtype1_txt_code',
        'weaptype1', 'weaptype1_txt_code',
        'extended', 'vicinity', 'suicide', 'property', 'ishostkid', 'claimed_clean'
    ]

    le = LabelEncoder()
    y_train = le.fit_transform(known_train['gname_mapped'])
    
    # Handle unseen categories in test set
    known_test['gname_mapped_clean'] = known_test['gname_mapped'].apply(
        lambda x: x if x in le.classes_ else 'Other_Known'
    )
    y_test = le.transform(known_test['gname_mapped_clean'])

    X_train = known_train[feature_cols]
    X_test = known_test[feature_cols]

    print(f"Training on {len(X_train)} known incidents across {len(le.classes_)} group categories...")
    
    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Top-3 Accuracy
    top3_correct = 0
    for i, true_label in enumerate(y_test):
        top3_preds = np.argsort(y_proba[i])[-3:]
        if true_label in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(y_test)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print(f"Test Top-3 Accuracy: {top3_acc:.4f}")

    # Predict for Unknown Attacks in test set
    if len(unknown_test) > 0:
        X_unknown = unknown_test[feature_cols]
        unknown_probs = clf.predict_proba(X_unknown)
        unknown_preds = clf.predict(X_unknown)
        top_predicted_groups = pd.Series(le.inverse_transform(unknown_preds)).value_counts().head(5).to_dict()
        print("Attribution summary for Unknown attacks in test set (Top Predicted Groups):")
        print(top_predicted_groups)
    else:
        top_predicted_groups = {}

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).to_dict()

    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': clf, 'label_encoder': le, 'features': feature_cols}, 'models/perpetrator_attribution_lgb.joblib')

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'top3_accuracy': float(top3_acc),
        'top_unknown_attributed_groups': top_predicted_groups,
        'feature_importances': importances
    }
    
    return clf, metrics

if __name__ == '__main__':
    train_perpetrator_attribution()
