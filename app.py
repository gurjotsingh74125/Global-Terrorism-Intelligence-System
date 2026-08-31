import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.model_inference import GTDInferenceEngine

# Set Page Config
st.set_page_config(
    page_title="Global Terrorism Intelligence System (GTIS)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism UI & Styling
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-subtitle {
        color: #3B82F6;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    div[data-testid="stColumn"] {
        border-right: 2px solid rgba(59, 130, 246, 0.3) !important;
        padding-right: 15px;
        padding-left: 15px;
    }
    div[data-testid="stColumn"]:last-child {
        border-right: none !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metrics():
    with open('outputs/metrics.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_sample_df():
    if os.path.exists('data/processed_gtd.pkl'):
        df = pd.read_pickle('data/processed_gtd.pkl')
        return df[['iyear', 'imonth_clean', 'region_txt', 'country_txt', 'latitude_clean', 'longitude_clean',
                   'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt', 'gname_mapped', 'success', 'total_casualty', 'casualty_tier']]
    else:
        return None

@st.cache_resource
def load_engine():
    return GTDInferenceEngine()

metrics_data = load_metrics()
sample_df = load_sample_df()
engine = load_engine()

# Header Banner
st.title("🛡️ Global Terrorism Intelligence System (GTIS)")
st.markdown("*A Multi-Modal Machine Learning & Spatiotemporal Risk Analytics Suite*")

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Total Incidents</div>
        <div class="metric-value">181,691</div>
        <div class="metric-subtitle">47-Year Span (1970-2017)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Attribution Model</div>
        <div class="metric-value">92.7%</div>
        <div class="metric-subtitle">Top-3 Acc: 99.9% (LightGBM)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Success Predictor</div>
        <div class="metric-value">0.906</div>
        <div class="metric-subtitle">ROC-AUC (PR-AUC: 0.981)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Spatial Hotspots</div>
        <div class="metric-value">118</div>
        <div class="metric-subtitle">DBSCAN Clusters Identified</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Attribution Gap</div>
        <div class="metric-value">45.6%</div>
        <div class="metric-subtitle">Unclaimed Incidents Solved</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary & GTD Overview",
    "🗺️ 3D Geospatial Threat Hotspots",
    "🎯 Live ML Prediction Studio",
    "📈 Forecasting & Model Diagnostics"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY & OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.header("Global Terrorism Intelligence Overview")
    if sample_df is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Incidents by Region")
            region_counts = sample_df['region_txt'].value_counts().reset_index()
            region_counts.columns = ['Region', 'Count']
            fig_reg = px.bar(region_counts, x='Count', y='Region', orientation='h',
                             color='Count', color_continuous_scale='Reds',
                             title="Incident Count per Region")
            fig_reg.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_reg, use_container_width=True)

        with c2:
            st.subheader("Top Attack Types")
            attack_counts = sample_df['attacktype1_txt'].value_counts().head(8).reset_index()
            attack_counts.columns = ['Attack Type', 'Count']
            fig_atk = px.pie(attack_counts, values='Count', names='Attack Type', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             title="Attack Method Distribution")
            fig_atk.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_atk, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Target Type Distribution")
            targ_counts = sample_df['targtype1_txt'].value_counts().head(8).reset_index()
            targ_counts.columns = ['Target Type', 'Count']
            fig_targ = px.bar(targ_counts, x='Target Type', y='Count', color='Count',
                              color_continuous_scale='Viridis', title="Top Target Categories")
            fig_targ.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_targ, use_container_width=True)

        with c4:
            st.subheader("Temporal Trends (1970–2017)")
            yearly_counts = sample_df.groupby('iyear').size().reset_index(name='Count')
            fig_year = px.line(yearly_counts, x='iyear', y='Count', title="Annual Global Incident Volume",
                               line_shape='spline')
            fig_year.update_traces(line_color='#3B82F6', line_width=3)
            fig_year.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_year, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: 3D GEOSPATIAL THREAT MAP
# ---------------------------------------------------------
with tab2:
    st.header("3D Geospatial Threat & Spatial Cluster Map")
    
    col_map1, col_map2 = st.columns([1, 3])

    with col_map1:
        st.subheader("Filter Spatial Data")
        selected_region = st.selectbox(
            "Select Region",
            options=["All Regions"] + list(sample_df['region_txt'].unique())
        )
        year_range = st.slider("Select Year Range", 1970, 2017, (2000, 2017))
        
        st.markdown("---")
        st.markdown("### Top Spatial Hotspot Clusters")
        clusters = metrics_data['spatiotemporal_forecasting']['top_10_spatial_hotspots']
        cluster_df = pd.DataFrame(clusters)
        st.dataframe(cluster_df[['primary_country', 'incident_count', 'primary_region']], use_container_width=True)

    with col_map2:
        # Filter map dataframe
        map_df = sample_df[(sample_df['iyear'] >= year_range[0]) & (sample_df['iyear'] <= year_range[1])].copy()
        if selected_region != "All Regions":
            map_df = map_df[map_df['region_txt'] == selected_region]
        
        # Sample for rendering performance if large
        if len(map_df) > 10000:
            map_df = map_df.sample(10000, random_state=42)

        st.subheader(f"Interactive Geospatial Scatter Map ({len(map_df)} Displayed Incidents)")
        
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude_clean",
            lon="longitude_clean",
            color="region_txt",
            size="total_casualty",
            size_max=20,
            zoom=1,
            hover_name="country_txt",
            hover_data=["iyear", "attacktype1_txt", "gname_mapped", "total_casualty"],
            mapbox_style="carto-darkmatter",
            title="Global Terrorism Incident Locations & Casualty Severity"
        )
        fig_map.update_layout(template='plotly_dark', margin={"r":0,"t":40,"l":0,"b":0}, height=650)
        st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: LIVE ML PREDICTION STUDIO
# ---------------------------------------------------------
with tab3:
    st.header("🎯 Live ML Model Prediction Studio")
    st.markdown("Configure attack tactical parameters below to receive instant real-time predictions from all 4 trained ML models.")

    c_in1, c_in2, c_in3 = st.columns(3)

    with c_in1:
        st.subheader("1. Location & Timing")
        pred_year = st.number_input("Incident Year", min_value=1970, max_value=2025, value=2017)
        pred_month = st.selectbox("Incident Month", options=list(range(1, 13)), index=5)
        pred_region = st.selectbox("Region Code", options=[
            (1, "North America"), (2, "Central America & Caribbean"), (3, "South America"),
            (4, "Western Europe"), (5, "Eastern Europe"), (6, "Middle East & North Africa"),
            (7, "Sub-Saharan Africa"), (8, "South Asia"), (9, "Southeast Asia"),
            (10, "East Asia"), (11, "Central Asia"), (12, "Australasia & Oceania")
        ], format_func=lambda x: x[1])
        pred_country = st.number_input("Country Code", min_value=1, max_value=250, value=95)

    with c_in2:
        st.subheader("2. Tactical Vectors")
        pred_attack = st.selectbox("Attack Type Code", options=[
            (1, "Assassination"), (2, "Armed Assault"), (3, "Bombing/Explosion"),
            (4, "Hijacking"), (5, "Hostage Taking (Barricade)"), (6, "Hostage Taking (Kidnapping)"),
            (7, "Facility/Infrastructure"), (8, "Unarmed Assault"), (9, "Unknown")
        ], index=2, format_func=lambda x: x[1])

        pred_target = st.selectbox("Target Type Code", options=[
            (1, "Business"), (2, "Government"), (3, "Police"), (4, "Military"),
            (6, "Airports/Aircraft"), (7, "Government (Diplomatic)"), (14, "Private Citizens & Property")
        ], index=6, format_func=lambda x: x[1])

        pred_weapon = st.selectbox("Weapon Type Code", options=[
            (5, "Firearms"), (6, "Explosives"), (8, "Incendiary"), (9, "Melee"), (13, "Unknown")
        ], index=1, format_func=lambda x: x[1])

    with c_in3:
        st.subheader("3. Modus Operandi Features")
        pred_suicide = st.selectbox("Suicide Attack?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
        pred_extended = st.selectbox("Extended (>24hrs)?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
        pred_property = st.selectbox("Property Damage?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], index=1)
        pred_claimed = st.selectbox("Claimed Responsibility?", options=[(1, "Yes"), (0, "No"), (-1, "Unknown")], format_func=lambda x: x[1])

    if st.button("🚀 Run Live ML Inference Engine", type="primary", use_container_width=True):
        input_dict = {
            'iyear': pred_year,
            'imonth_clean': pred_month,
            'iday_clean': 15,
            'quarter': (pred_month - 1) // 3 + 1,
            'region': pred_region[0],
            'country': pred_country,
            'region_txt_code': pred_region[0],
            'country_txt_code': pred_country,
            'latitude_clean': 33.3,
            'longitude_clean': 44.4,
            'attacktype1': pred_attack[0],
            'attacktype1_txt_code': pred_attack[0],
            'targtype1': pred_target[0],
            'targtype1_txt_code': pred_target[0],
            'weaptype1': pred_weapon[0],
            'weaptype1_txt_code': pred_weapon[0],
            'extended': pred_extended[0],
            'vicinity': 0,
            'suicide': pred_suicide[0],
            'property': pred_property[0],
            'ishostkid': 0,
            'claimed_clean': pred_claimed[0],
            'is_known_group': 1 if pred_claimed[0] == 1 else 0
        }

        with st.spinner("Computing predictions across all 4 models..."):
            attribution_res = engine.predict_attribution(input_dict)
            success_res = engine.predict_success(input_dict)
            casualty_res = engine.predict_casualty_risk(input_dict)

        st.markdown("---")
        st.subheader("Model Inference Results")

        res_col1, res_col2, res_col3 = st.columns(3, border=True)

        with res_col1:
            st.markdown("### 🔍 Perpetrator Attribution")
            st.markdown(f"**Top Predicted Group:** `{attribution_res[0]['group']}`")
            st.markdown(f"**Confidence:** `{attribution_res[0]['probability']*100:.1f}%`")
            
            # Probability Breakdown Chart
            attr_df = pd.DataFrame(attribution_res)
            fig_attr = px.bar(attr_df, x='probability', y='group', orientation='h',
                              title="Top 5 Group Probabilities", color='probability', color_continuous_scale='Blues')
            fig_attr.update_layout(template='plotly_dark', height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_attr, use_container_width=True)

        with res_col2:
            st.markdown("### 🎯 Attack Success Predictor")
            prob = success_res['success_probability']
            outcome = success_res['predicted_outcome']

            st.markdown(f"**Predicted Outcome:** `{outcome}`")
            st.markdown(f"**Success Probability:** `{prob*100:.1f}%`")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if outcome == "Failure" else "#10B981"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(30, 41, 59, 0.7)"},
                        {'range': [50, 100], 'color': "rgba(15, 23, 42, 0.9)"}
                    ]
                }
            ))
            fig_gauge.update_layout(template='plotly_dark', height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col3:
            st.markdown("### ⚠️ Casualty Severity Risk")
            st.markdown(f"**Severity Tier:** `{casualty_res['tier_name']}`")
            st.markdown(f"**Median Casualty Estimate (q=0.50):** `{casualty_res['median_casualty_estimate_q50']}` lives")
            st.markdown(f"**90th-Percentile Risk Upper Bound:** `{casualty_res['high_risk_upper_bound_q90']}` lives")

            # Tier Probabilities
            tier_df = pd.DataFrame(list(casualty_res['tier_probabilities'].items()), columns=['Tier', 'Probability'])
            fig_tier = px.bar(tier_df, x='Probability', y='Tier', orientation='h',
                              color='Probability', color_continuous_scale='Oranges',
                              title="Casualty Severity Tier Probabilities")
            fig_tier.update_layout(template='plotly_dark', height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_tier, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: FORECASTING & MODEL DIAGNOSTICS
# ---------------------------------------------------------
with tab4:
    st.header("📈 Model Diagnostics & Feature Importances")
    
    st.subheader("Model Performance Benchmark Matrix")
    benchmark_data = [
        {"Model Domain": "Perpetrator Attribution", "Primary Model": "Multi-Class LightGBM", "Primary Metric": "Accuracy: 92.71%", "Secondary Metric": "Top-3 Acc: 99.96%"},
        {"Model Domain": "Attack Success Prediction", "Primary Model": "Imbalanced LightGBM", "Primary Metric": "ROC-AUC: 0.9059", "Secondary Metric": "PR-AUC: 0.9810"},
        {"Model Domain": "Casualty Risk Estimation", "Primary Model": "Quantile LightGBM", "Primary Metric": "q=0.50 Loss: 2.21", "Secondary Metric": "q=0.90 Loss: 2.20"},
        {"Model Domain": "Spatiotemporal Forecasting", "Primary Model": "Regional Lag LightGBM", "Primary Metric": "Test R2: 0.7891", "Secondary Metric": "Test MAE: 33.81"}
    ]
    st.table(pd.DataFrame(benchmark_data))

    st.markdown("---")
    st.subheader("Feature Importance Rankings")
    
    diag_model = st.selectbox("Select Model to Inspect Feature Importances", options=[
        "Perpetrator Attribution", "Attack Success Prediction", "Casualty Risk Estimation"
    ])

    if diag_model == "Perpetrator Attribution":
        fi_dict = metrics_data['perpetrator_attribution']['feature_importances']
    elif diag_model == "Attack Success Prediction":
        fi_dict = metrics_data['attack_success_prediction']['feature_importances']
    else:
        fi_dict = metrics_data['casualty_risk_estimation']['feature_importances']

    fi_df = pd.DataFrame(list(fi_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True).tail(15)
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance',
                    color_continuous_scale='Tealgrn', title=f"Top 15 Feature Importances ({diag_model})")
    fig_fi.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig_fi, use_container_width=True)
