"""
TicketIQ - Page 6: Comprehensive Model Evaluation & Benchmarks
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

from src.utils import load_json
from src.evaluation import create_confusion_matrix_fig, create_model_comparison_bar_chart

st.set_page_config(page_title="Model Performance | TicketIQ", page_icon="🏆", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">🏆 Machine Learning Model Evaluation & Benchmarks</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">Rigorous empirical comparison of 5 classification algorithms across Category routing and Priority triage.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load Evaluation Reports
eval_summary = load_json("reports/evaluation_summary.json")
comp_path = "reports/model_comparison.csv"
comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()

# Metrics Explanation Card
with st.expander("📘 Guide: Understanding Machine Learning Evaluation Metrics", expanded=False):
    st.markdown(
        """
        - **Accuracy:** The overall proportion of correctly classified tickets out of all evaluated test samples.
        - **Precision (Macro / Weighted):** Out of all tickets predicted as a specific category/priority, how many were actually that class. High precision avoids misrouting.
        - **Recall (Macro / Weighted):** Out of all true tickets belonging to a class, how many were correctly detected. High recall ensures critical emergencies are not missed.
        - **F1-Score (Macro / Weighted):** The harmonic mean of precision and recall. Ideal for multi-class NLP where class balance is essential.
        - **Confusion Matrix:** A square matrix mapping true ground truth classes against model predictions to pinpoint confusion clusters.
        """
    )

if not comp_df.empty:
    m_tab1, m_tab2, m_tab3 = st.tabs([
        "🏷️ Category Models Benchmark",
        "⚡ Priority Models Benchmark",
        "📥 Download Evaluation Reports"
    ])
    
    # Category Models Tab
    with m_tab1:
        cat_comp = comp_df[comp_df["task"] == "Category Classification"].copy()
        st.markdown("### 📊 **Algorithm Comparison Table (Test Split)**")
        st.dataframe(cat_comp, use_container_width=True)
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.plotly_chart(create_model_comparison_bar_chart(cat_comp, "f1_weighted"), use_container_width=True)
        with c_chart2:
            st.plotly_chart(create_model_comparison_bar_chart(cat_comp, "training_time"), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧩 **Best Category Model Confusion Matrix**")
        cat_best_dict = eval_summary.get("category_model", {})
        best_cat_name = cat_best_dict.get("best_model", "Multinomial Naive Bayes")
        best_cat_data = cat_best_dict.get("all_models", {}).get(best_cat_name, {})
        
        if "confusion_matrix" in best_cat_data:
            cm_cat_fig = create_confusion_matrix_fig(
                best_cat_data["confusion_matrix"],
                best_cat_data["classes"],
                f"Confusion Matrix: {best_cat_name}"
            )
            st.plotly_chart(cm_cat_fig, use_container_width=True)
            
        if "classification_report_text" in best_cat_data:
            with st.expander(f"Full Classification Report: {best_cat_name}"):
                st.code(best_cat_data["classification_report_text"])

    # Priority Models Tab
    with m_tab2:
        prio_comp = comp_df[comp_df["task"] == "Priority Prediction"].copy()
        st.markdown("### 📊 **Algorithm Comparison Table (Test Split)**")
        st.dataframe(prio_comp, use_container_width=True)
        
        p_chart1, p_chart2 = st.columns(2)
        with p_chart1:
            st.plotly_chart(create_model_comparison_bar_chart(prio_comp, "f1_weighted"), use_container_width=True)
        with p_chart2:
            st.plotly_chart(create_model_comparison_bar_chart(prio_comp, "training_time"), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧩 **Best Priority Model Confusion Matrix**")
        prio_best_dict = eval_summary.get("priority_model", {})
        best_prio_name = prio_best_dict.get("best_model", "Random Forest")
        best_prio_data = prio_best_dict.get("all_models", {}).get(best_prio_name, {})
        
        if "confusion_matrix" in best_prio_data:
            cm_prio_fig = create_confusion_matrix_fig(
                best_prio_data["confusion_matrix"],
                best_prio_data["classes"],
                f"Confusion Matrix: {best_prio_name}"
            )
            st.plotly_chart(cm_prio_fig, use_container_width=True)
            
        if "classification_report_text" in best_prio_data:
            with st.expander(f"Full Classification Report: {best_prio_name}"):
                st.code(best_prio_data["classification_report_text"])

    # Download Tab
    with m_tab3:
        st.markdown("### 📥 **Export Benchmarking & Evaluation Artifacts**")
        st.markdown("Download official evaluation reports for documentation, compliance, and internship submission.")
        
        d_col1, d_col2, d_col3 = st.columns(3)
        
        with d_col1:
            # CSV Download
            csv_data = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Model Comparison (CSV)",
                data=csv_data,
                file_name="model_comparison_report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with d_col2:
            # TXT Report Download
            txt_path = "reports/classification_report.txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_data = f.read()
                st.download_button(
                    label="📑 Download Full Text Report (TXT)",
                    data=txt_data.encode("utf-8"),
                    file_name="classification_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        with d_col3:
            # JSON Download
            import json
            json_str = json.dumps(eval_summary, indent=2)
            st.download_button(
                label="⚙️ Download Complete JSON Summary",
                data=json_str.encode("utf-8"),
                file_name="evaluation_summary.json",
                mime="application/json",
                use_container_width=True
            )
else:
    st.info("Evaluation reports not found. Run model training from Page 4 or execute `python train_all.py`.")

