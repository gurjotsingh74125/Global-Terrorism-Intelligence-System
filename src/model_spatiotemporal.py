import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_spatiotemporal(train_path='data/train_gtd.pkl', test_path='data/test_gtd.pkl'):
    print("\n=== Training Spatiotemporal Clustering & Forecasting Models ===")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Spatial DBSCAN Clustering (Hotspot Discovery)
    print("Performing Spatial DBSCAN Clustering on valid geolocation coordinates...")
    geo_df = full_df[(full_df['latitude_clean'] != 0) & (full_df['longitude_clean'] != 0)].copy()
    
    # Sample 30,000 recent points for efficient Haversine clustering
    geo_sample = geo_df[geo_df['iyear'] >= 2010].copy()
    coords_rad = np.radians(geo_sample[['latitude_clean', 'longitude_clean']].values)

    kms_per_radian = 6371.0088
    epsilon_km = 50.0  # 50km neighborhood radius
    epsilon_rad = epsilon_km / kms_per_radian

    db = DBSCAN(eps=epsilon_rad, min_samples=30, metric='haversine', algorithm='ball_tree')
    geo_sample['cluster'] = db.fit_predict(coords_rad)

    n_clusters = len(set(geo_sample['cluster'])) - (1 if -1 in geo_sample['cluster'] else 0)
    noise_count = (geo_sample['cluster'] == -1).sum()

    print(f"Identified {n_clusters} spatial hotspot clusters (Noise incidents: {noise_count})")

    top_clusters = []
    cluster_counts = geo_sample['cluster'].value_counts()
    for cid, count in cluster_counts.items():
        if cid == -1:
            continue
        c_df = geo_sample[geo_sample['cluster'] == cid]
        lat_cen = c_df['latitude_clean'].mean()
        lon_cen = c_df['longitude_clean'].mean()
        top_region = c_df['region_txt'].mode()[0]
        top_country = c_df['country_txt'].mode()[0]
        top_clusters.append({
            'cluster_id': int(cid),
            'incident_count': int(count),
            'latitude_centroid': float(lat_cen),
            'longitude_centroid': float(lon_cen),
            'primary_region': top_region,
            'primary_country': top_country
        })
    
    top_clusters = sorted(top_clusters, key=lambda x: x['incident_count'], reverse=True)[:10]
    print(f"Top spatial cluster: {top_clusters[0]['primary_country']} ({top_clusters[0]['incident_count']} incidents)")

    # 2. Time-Series Incident Volume Forecasting (Monthly by Region)
    print("Building Regional Monthly Incident Time-Series Lag Forecasting Model...")
    ts_df = full_df.groupby(['region_txt', 'iyear', 'imonth_clean']).size().reset_index(name='incident_count')
    ts_df['year_month'] = pd.to_datetime(ts_df['iyear'].astype(str) + '-' + ts_df['imonth_clean'].astype(str).str.zfill(2) + '-01')
    ts_df = ts_df.sort_values(['region_txt', 'year_month']).reset_index(drop=True)

    # Create lag features per region
    for lag in [1, 2, 3, 6, 12]:
        ts_df[f'lag_{lag}'] = ts_df.groupby('region_txt')['incident_count'].shift(lag)

    ts_df['rolling_mean_3'] = ts_df.groupby('region_txt')['lag_1'].transform(lambda x: x.rolling(3).mean())
    ts_df['rolling_mean_6'] = ts_df.groupby('region_txt')['lag_1'].transform(lambda x: x.rolling(6).mean())

    ts_clean = ts_df.dropna().copy()
    ts_clean['region_code'] = ts_clean['region_txt'].astype('category').cat.codes

    ts_train = ts_clean[ts_clean['iyear'] <= 2012]
    ts_test = ts_clean[ts_clean['iyear'] > 2012]

    lag_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_6', 'region_code', 'imonth_clean']
    
    forecaster = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    forecaster.fit(ts_train[lag_cols], ts_train['incident_count'])

    y_ts_pred = forecaster.predict(ts_test[lag_cols])
    y_ts_true = ts_test['incident_count']

    ts_mae = mean_absolute_error(y_ts_true, y_ts_pred)
    ts_rmse = np.sqrt(mean_squared_error(y_ts_true, y_ts_pred))
    ts_r2 = r2_score(y_ts_true, y_ts_pred)

    print(f"Monthly Forecast Test MAE: {ts_mae:.4f}")
    print(f"Monthly Forecast Test RMSE: {ts_rmse:.4f}")
    print(f"Monthly Forecast Test R2: {ts_r2:.4f}")

    # Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'spatial_clusters': top_clusters,
        'forecaster': forecaster,
        'lag_features': lag_cols
    }, 'models/spatiotemporal_lgb.joblib')

    metrics = {
        'spatial_cluster_count': n_clusters,
        'top_10_spatial_hotspots': top_clusters,
        'forecast_mae': float(ts_mae),
        'forecast_rmse': float(ts_rmse),
        'forecast_r2': float(ts_r2)
    }

    return metrics

if __name__ == '__main__':
    train_spatiotemporal()
