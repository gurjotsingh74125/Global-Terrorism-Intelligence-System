import os
import joblib
import pandas as pd
import numpy as np

class GTDInferenceEngine:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.attribution_pack = joblib.load(os.path.join(models_dir, 'perpetrator_attribution_lgb.joblib'))
        self.success_pack = joblib.load(os.path.join(models_dir, 'success_prediction_lgb.joblib'))
        self.casualty_pack = joblib.load(os.path.join(models_dir, 'casualty_estimation_lgb.joblib'))
        self.spatiotemporal_pack = joblib.load(os.path.join(models_dir, 'spatiotemporal_lgb.joblib'))

    def predict_attribution(self, input_dict):
        """Predict perpetrator group probabilities."""
        df_input = pd.DataFrame([input_dict])
        feats = self.attribution_pack['features']
        # Align features
        for f in feats:
            if f not in df_input.columns:
                df_input[f] = 0
        df_input = df_input[feats]

        model = self.attribution_pack['model']
        le = self.attribution_pack['label_encoder']

        probs = model.predict_proba(df_input)[0]
        top_indices = np.argsort(probs)[::-1][:5]
        
        results = []
        for idx in top_indices:
            group_name = le.inverse_transform([idx])[0]
            probability = probs[idx]
            results.append({'group': group_name, 'probability': float(probability)})
        return results

    def predict_success(self, input_dict):
        """Predict attack success probability."""
        df_input = pd.DataFrame([input_dict])
        feats = self.success_pack['features']
        for f in feats:
            if f not in df_input.columns:
                df_input[f] = 0
        df_input = df_input[feats]

        model = self.success_pack['model']
        success_prob = model.predict_proba(df_input)[0][1]
        is_success = bool(success_prob >= 0.5)
        return {'success_probability': float(success_prob), 'predicted_outcome': 'Success' if is_success else 'Failure'}

    def predict_casualty_risk(self, input_dict):
        """Predict casualty severity tier and quantile bounds."""
        df_input = pd.DataFrame([input_dict])
        feats = self.casualty_pack['features']
        for f in feats:
            if f not in df_input.columns:
                df_input[f] = 0
        df_input = df_input[feats]

        clf_tier = self.casualty_pack['clf_tier']
        reg_q50 = self.casualty_pack['reg_q50']
        reg_q90 = self.casualty_pack['reg_q90']

        tier_idx = int(clf_tier.predict(df_input)[0])
        tier_probs = clf_tier.predict_proba(df_input)[0]

        tier_names = {
            0: 'Zero Casualties (0)',
            1: 'Low Severity (1-4)',
            2: 'Moderate Severity (5-19)',
            3: 'High Severity (20-99)',
            4: 'Mass Casualty (100+)'
        }

        est_median = max(0.0, float(reg_q50.predict(df_input)[0]))
        est_high_risk = max(est_median, float(reg_q90.predict(df_input)[0]))

        return {
            'tier_code': tier_idx,
            'tier_name': tier_names.get(tier_idx, 'Unknown'),
            'tier_probabilities': {tier_names[i]: float(p) for i, p in enumerate(tier_probs)},
            'median_casualty_estimate_q50': round(est_median, 1),
            'high_risk_upper_bound_q90': round(est_high_risk, 1)
        }

    def get_spatial_clusters(self):
        """Return top spatial clusters."""
        return self.spatiotemporal_pack['spatial_clusters']

    def forecast_monthly_incidents(self, region_code, month, lag_1, lag_2, lag_3, lag_6, lag_12):
        """Forecast monthly incidents given lag features."""
        r3 = (lag_1 + lag_2 + lag_3) / 3.0
        r6 = (lag_1 + lag_2 + lag_3 + lag_6) / 4.0
        df_input = pd.DataFrame([{
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'lag_6': lag_6,
            'lag_12': lag_12,
            'rolling_mean_3': r3,
            'rolling_mean_6': r6,
            'region_code': region_code,
            'imonth_clean': month
        }])
        forecaster = self.spatiotemporal_pack['forecaster']
        prediction = forecaster.predict(df_input[self.spatiotemporal_pack['lag_features']])[0]
        return max(0.0, float(prediction))
