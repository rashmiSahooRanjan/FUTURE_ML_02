"""
TicketIQ - Page 3: Bulk Ticket Classification & Batch Processing
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import io

from src.data_loader import detect_columns
from src.prediction import predict_batch_tickets
from src.analytics import plot_category_distribution, plot_priority_distribution

st.set_page_config(page_title="Bulk Classification | TicketIQ", page_icon="📁", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">📁 Bulk Ticket Classification & Batch Triage</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">Process entire CSV ticket dumps automatically. Classify thousands of customer issues and download enriched triage spreadsheets.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File Uploader or Sample Loader
upload_tab, sample_tab = st.tabs(["📤 Upload Custom CSV", "🧪 Load Built-in Sample Dataset"])

df_to_process = None

with upload_tab:
    uploaded_file = st.file_uploader("Upload customer tickets file (CSV format)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file successfully loaded: **{len(df_to_process):,} records**.")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

with sample_tab:
    sample_path = "data/sample/sample_tickets.csv"
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        st.markdown(f"The built-in sample contains **{len(sample_df)} representative support tickets** across all business units.")
        if st.button("🚀 Load Sample Dataset for Testing", key="load_sample_btn"):
            df_to_process = sample_df.copy()
            st.session_state["active_bulk_df"] = df_to_process
    else:
        st.info("Sample dataset will be generated during initial model training.")

if "active_bulk_df" in st.session_state and df_to_process is None:
    df_to_process = st.session_state["active_bulk_df"]

if df_to_process is not None:
    st.markdown("---")
    st.markdown("### 🔍 **Column Mapping & Configuration**")
    
    detected = detect_columns(df_to_process.columns.tolist())
    cols = df_to_process.columns.tolist()
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        default_desc_idx = cols.index(detected["description"]) if detected["description"] in cols else 0
        desc_col = st.selectbox("Ticket Description Column *", cols, index=default_desc_idx)
    with col_sel2:
        subject_options = ["None"] + cols
        default_subj_idx = (subject_options.index(detected["subject"])) if (detected["subject"] in cols) else 0
        subj_col_sel = st.selectbox("Ticket Subject Column (Optional)", subject_options, index=default_subj_idx)
        subj_col = None if subj_col_sel == "None" else subj_col_sel

    st.markdown("#### **Raw Data Preview (First 5 Rows)**")
    st.dataframe(df_to_process.head(5), use_container_width=True)

    # Process Button
    if st.button("⚡ Run Bulk AI Classification", use_container_width=True):
        with st.spinner(f"Classifying {len(df_to_process):,} tickets using trained NLP models..."):
            try:
                classified_df = predict_batch_tickets(
                    df=df_to_process,
                    description_col=desc_col,
                    subject_col=subj_col
                )
                st.session_state["classified_bulk_df"] = classified_df
                st.success(f" Successfully classified {len(classified_df):,} support tickets!")
            except Exception as e:
                st.error(f"Error during bulk classification: {str(e)}")

# Display Results if Available
if "classified_bulk_df" in st.session_state:
    classified_df = st.session_state["classified_bulk_df"]
    
    st.markdown("---")
    st.markdown("### 📊 **Batch Classification Summary**")
    
    # KPI Row
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Processed Tickets</div><div class="kpi-value">{len(classified_df):,}</div><div class="kpi-subtext">Completed records</div></div>""", unsafe_allow_html=True)
    with b2:
        crit_n = (classified_df["Predicted_Priority"] == "Critical").sum()
        st.markdown(f"""<div class="kpi-card critical"><div class="kpi-label">Critical Predicted</div><div class="kpi-value">{crit_n:,}</div><div class="kpi-subtext">Require immediate SLA</div></div>""", unsafe_allow_html=True)
    with b3:
        high_n = (classified_df["Predicted_Priority"] == "High").sum()
        st.markdown(f"""<div class="kpi-card high"><div class="kpi-label">High Predicted</div><div class="kpi-value">{high_n:,}</div><div class="kpi-subtext">Expedited queue</div></div>""", unsafe_allow_html=True)
    with b4:
        avg_conf = classified_df["Category_Confidence"].mean() * 100
        st.markdown(f"""<div class="kpi-card purple"><div class="kpi-label">Avg Category Confidence</div><div class="kpi-value">{avg_conf:.1f}%</div><div class="kpi-subtext">Across batch corpus</div></div>""", unsafe_allow_html=True)

    # Visualizations of batch results
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.plotly_chart(plot_category_distribution(classified_df, "Predicted_Category"), use_container_width=True)
    with chart_c2:
        st.plotly_chart(plot_priority_distribution(classified_df, "Predicted_Priority"), use_container_width=True)

    # Interactive Table with Filters
    st.markdown("### 📑 **Enriched Ticket Results**")
    
    tf1, tf2, tf3 = st.columns([2, 1, 1])
    with tf1:
        tbl_search = st.text_input("Filter results by keyword:", "")
    with tf2:
        cat_choices = ["All"] + sorted(classified_df["Predicted_Category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Filter Predicted Category:", cat_choices)
    with tf3:
        prio_choices = ["All"] + sorted(classified_df["Predicted_Priority"].dropna().unique().tolist())
        sel_prio = st.selectbox("Filter Predicted Priority:", prio_choices)

    view_df = classified_df.copy()
    if sel_cat != "All":
        view_df = view_df[view_df["Predicted_Category"] == sel_cat]
    if sel_prio != "All":
        view_df = view_df[view_df["Predicted_Priority"] == sel_prio]
    if tbl_search.strip():
        s = tbl_search.strip().lower()
        view_df = view_df[view_df[desc_col].fillna("").str.lower().str.contains(s)]

    st.markdown(f"**Displaying {len(view_df):,} of {len(classified_df):,} records**")
    st.dataframe(view_df, use_container_width=True, height=380)

    # Download Enriched CSV
    csv_buffer = io.StringIO()
    classified_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Classified Tickets (CSV)",
        data=csv_bytes,
        file_name="classified_support_tickets_ticketiq.csv",
        mime="text/csv",
        use_container_width=True
    )

