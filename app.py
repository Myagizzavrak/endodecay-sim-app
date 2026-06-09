# ==============================================================================
# PROJECT: EndoDecay-Sim v13.1 UI Dashboard (Dynamic Simulation Edition)
# PARADIGM: Live Predictive Filtering & Dynamic Kaplan-Meier Recalculation
# EXECUTION COMMAND: streamlit run app.py
# AUTHOR: Muhammet Yagiz Zavrak
# COPYRIGHT: (c) 2026 Muhammet Yagiz Zavrak. All rights reserved.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="EndoDecay-Sim v13.1 Platform")
st.title("🫀 EndoDecay-Sim v13.1: Clinical Decision Support Platform")
st.markdown("---")

cohort_path = 'EndoDecay_Sim_v13_Final_Data.csv'

def calculate_live_km_curve(filtered_df, total_months=120):
    """Filtrelenmiş alt popülasyon için canlı Kaplan-Meier eğrisi hesaplar."""
    timeline = np.arange(0, total_months + 1, 1)
    survival_probabilities = []
    active_at_risk = len(filtered_df)
    
    for month in timeline:
        events = filtered_df[(filtered_df['Event_Observed'] == 1) & (filtered_df['Time_To_Event'].astype(int) == month)].shape[0]
        if active_at_risk > 0:
            step = 1.0 - (events / active_at_risk)
            if len(survival_probabilities) == 0: 
                survival_probabilities.append(step)
            else: 
                survival_probabilities.append(survival_probabilities[-1] * step)
        else:
            survival_probabilities.append(0.0)
        active_at_risk -= events
        
    return pd.DataFrame({'Timeline_Month': timeline, 'Survival_Probability': survival_probabilities})

if os.path.exists(cohort_path):
    @st.cache_data
    def load_base_data(path):
        return pd.read_csv(path)
    
    df_raw = load_base_data(cohort_path)
    
    st.sidebar.header("📋 Clinical Scenario Simulation")
    st.sidebar.markdown("Adjust patient criteria to observe multi-systemic cardiorenal outcomes in real-time.")
    
    min_age_val = int(np.floor(df_raw['Age'].min()))
    max_age_val = int(np.ceil(df_raw['Age'].max()))
    age_filter = st.sidebar.slider("Patient Age Range", min_age_val, max_age_val, (min_age_val, max_age_val))
    
    max_hba1c_val = float(df_raw['HbA1c'].max())
    min_hba1c_val = float(df_raw['HbA1c'].min())
    hba1c_filter = st.sidebar.slider("Maximum Permissible HbA1c", min_hba1c_val, max_hba1c_val, max_hba1c_val)
    
    min_egfr_val = float(df_raw['Baseline_eGFR'].min())
    max_egfr_val = float(df_raw['Baseline_eGFR'].max())
    egfr_filter = st.sidebar.slider("Minimum Baseline eGFR Clearance", min_egfr_val, max_egfr_val, min_egfr_val)

    toxic_filter = st.sidebar.slider("Minimum Toxic Insult Load", 0.0, 1.0, 0.0)
    
    df_filtered = df_raw[
        (df_raw['Age'] >= age_filter[0]) & (df_raw['Age'] <= age_filter[1]) &
        (df_raw['HbA1c'] <= hba1c_filter) &
        (df_raw['Baseline_eGFR'] >= egfr_filter) &
        (df_raw['Toxic_Insult'] >= toxic_filter)
    ]
    
    st.subheader("📊 Selected Sub-Cohort Telemetry")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Sample Size (Active Virtual Patients)", f"{len(df_filtered):,}")
    with m_col2:
        if len(df_filtered) > 0:
            incidence_rate = (df_filtered['Event_Observed'].sum() / len(df_filtered)) * 100
            st.metric("Cardiorenal Collapse Incidence Rate", f"{incidence_rate:.2f}%")
        else:
            st.metric("Cardiorenal Collapse Incidence Rate", "0.00%")
    with m_col3:
        if len(df_filtered) > 0:
            st.metric("Mean Terminal eGFR Clearance", f"{df_filtered['eGFR_Final'].mean():.2f} mL/min")
        else:
            st.metric("Mean Terminal eGFR Clearance", "0.00 mL/min")
            
    st.markdown("---")
    
    if len(df_filtered) > 0:
        df_dynamic_km = calculate_live_km_curve(df_filtered)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dynamic Kaplan-Meier Survival Estimation")
            fig_km = px.line(
                df_dynamic_km, 
                x='Timeline_Month', 
                y='Survival_Probability',
                labels={'Timeline_Month': 'Timeline (Months)', 'Survival_Probability': 'Survival Probability'},
                template='plotly_dark'
            )
            fig_km.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig_km, use_container_width=True)
            
        with col2:
            st.subheader("Terminal eGFR Clearance vs. Myocardial Wall Stress (NT-proBNP)")
            fig_scatter = px.scatter(
                df_filtered,
                x='eGFR_Final',
                y='NT_proBNP_Final',
                color='Event_Observed',
                labels={'eGFR_Final': 'Terminal eGFR (mL/min)', 'NT_proBNP_Final': 'Terminal NT-proBNP (pg/mL)'},
                template='plotly_dark',
                trendline="ols" if len(df_filtered) > 10 else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.success("STATUS: Clinical simulation matrices synchronized and recalculated successfully.")
    else:
        st.warning("⚠️ No virtual patients match the selected clinical criteria. Please broaden your sidebar filters.")
        
else:
    st.error("CRITICAL: Downstream simulation datasets missing. Please execute 'endodecay_sim_v13.1_core.py' first to register core matrices.")