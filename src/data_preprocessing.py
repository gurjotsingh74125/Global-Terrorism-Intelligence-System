import os
import pandas as pd
import numpy as np

def load_and_preprocess_gtd(raw_csv_path='globalterrorismdb_0718dist.csv'):
    print(f"Loading dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, encoding='ISO-8859-1', low_memory=False)
    print(f"Raw shape: {df.shape}")

    # 1. Temporal features
    df['imonth_clean'] = df['imonth'].apply(lambda x: x if x in range(1, 13) else 1)
    df['iday_clean'] = df['iday'].apply(lambda x: x if x in range(1, 32) else 1)
    df['date'] = pd.to_datetime(
        df['iyear'].astype(str) + '-' + 
        df['imonth_clean'].astype(str).str.zfill(2) + '-' + 
        df['iday_clean'].astype(str).str.zfill(2),
        errors='coerce'
    )
    df['quarter'] = df['imonth_clean'].apply(lambda m: (m - 1) // 3 + 1)
    df['decade'] = (df['iyear'] // 10) * 10

    # 2. Casualty Targets & Severity Tiers
    df['nkill_clean'] = df['nkill'].fillna(0)
    df['nwound_clean'] = df['nwound'].fillna(0)
    df['total_casualty'] = df['nkill_clean'] + df['nwound_clean']

    def assign_casualty_tier(val):
        if val == 0:
            return 0  # Zero
        elif val <= 4:
            return 1  # Low (1-4)
        elif val <= 19:
            return 2  # Moderate (5-19)
        elif val <= 99:
            return 3  # High (20-99)
        else:
            return 4  # Mass Casualty (100+)

    df['casualty_tier'] = df['total_casualty'].apply(assign_casualty_tier)

    # 3. Perpetrator Group Targets & Mapping
    top_groups = df[df['gname'] != 'Unknown']['gname'].value_counts().head(15).index.tolist()
    
    def map_group(g):
        if g == 'Unknown':
            return 'Unknown'
        elif g in top_groups:
            return g
        else:
            return 'Other_Known'

    df['gname_mapped'] = df['gname'].apply(map_group)
    df['is_known_group'] = (df['gname'] != 'Unknown').astype(int)

    # 4. Impute Spatial Coordinates by Country Centroids
    country_lat_mean = df.groupby('country')['latitude'].transform('mean')
    country_lon_mean = df.groupby('country')['longitude'].transform('mean')
    
    df['latitude_clean'] = df['latitude'].fillna(country_lat_mean).fillna(0)
    df['longitude_clean'] = df['longitude'].fillna(country_lon_mean).fillna(0)

    # 5. Tactical Features & Missingness Indicators
    df['nperps_missing'] = df['nperps'].isnull().astype(int)
    df['claimed_missing'] = df['claimed'].isnull().astype(int)
    df['claimed_clean'] = df['claimed'].fillna(-1).astype(int)
    
    # Categorical string features
    cat_cols = [
        'region_txt', 'country_txt', 'provstate', 'city',
        'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt',
        'targsubtype1_txt', 'weapsubtype1_txt'
    ]

    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
        # Create encoded numerical feature
        df[f'{col}_code'] = df[col].astype('category').cat.codes

    # 6. Feature columns for ML modeling
    feature_cols = [
        'iyear', 'imonth_clean', 'iday_clean', 'quarter', 'decade',
        'region', 'country', 'region_txt_code', 'country_txt_code',
        'latitude_clean', 'longitude_clean',
        'attacktype1', 'attacktype1_txt_code',
        'targtype1', 'targtype1_txt_code',
        'weaptype1', 'weaptype1_txt_code',
        'extended', 'vicinity', 'crit1', 'crit2', 'crit3',
        'doubtterr', 'multiple', 'suicide', 'property', 'ishostkid',
        'nperps_missing', 'claimed_missing', 'claimed_clean'
    ]
    
    # Ensure binary indicators have no NaNs
    for c in ['extended', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr', 'multiple', 'suicide', 'property', 'ishostkid']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    print(f"Preprocessing completed. Feature count: {len(feature_cols)}")
    
    # 7. Time-based train/test split (Train: 1970-2012, Test: 2013-2017)
    train_mask = df['iyear'] <= 2012
    test_mask = df['iyear'] > 2012

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"Train split (1970-2012): {len(train_df)} rows")
    print(f"Test split (2013-2017): {len(test_df)} rows")

    os.makedirs('data', exist_ok=True)
    df.to_pickle('data/processed_gtd.pkl')
    train_df.to_pickle('data/train_gtd.pkl')
    test_df.to_pickle('data/test_gtd.pkl')
    
    print("Saved processed data to data/ processed_gtd.pkl, train_gtd.pkl, test_gtd.pkl")
    return df, train_df, test_df, feature_cols

if __name__ == '__main__':
    load_and_preprocess_gtd()
