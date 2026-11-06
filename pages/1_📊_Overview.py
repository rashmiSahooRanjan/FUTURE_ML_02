"""
TicketIQ - Page 1: Overview Dashboard
"""
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from src.utils import load_json, get_priority_badge
from src.analytics import (
    plot_category_distribution,
    plot_priority_distribution,
    plot_category_priority_heatmap
)

st.set_page_config(page_title="Overview | TicketIQ", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Data
@st.cache_data
def get_data():
    raw_path = "data/raw/customer_support_tickets.csv"
    if not os.path.exists(raw_path):
        raw_path = "Dataset/customer_support_tickets.csv"
    return pd.read_csv(raw_path)

df = get_data()
metadata = load_json("models/model_metadata.json")

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">📊 Support Operations Overview</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">Real-time insights across customer ticket volume, classification distribution, and model operations.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# High Level KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
total_t = len(df)
crit_t = (df["Ticket Priority"] == "Critical").sum() if "Ticket Priority" in df.columns else 0
high_t = (df["Ticket Priority"] == "High").sum() if "Ticket Priority" in df.columns else 0
med_t = (df["Ticket Priority"] == "Medium").sum() if "Ticket Priority" in df.columns else 0
low_t = (df["Ticket Priority"] == "Low").sum() if "Ticket Priority" in df.columns else 0
num_cats = df["Ticket Type"].nunique() if "Ticket Type" in df.columns else 5

with c1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Tickets</div><div class="kpi-value">{total_t:,}</div><div class="kpi-subtext">Active corpus records</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card critical"><div class="kpi-label">Critical / Urgent</div><div class="kpi-value">{crit_t:,}</div><div class="kpi-subtext">Immediate escalation</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card high"><div class="kpi-label">High Priority</div><div class="kpi-value">{high_t:,}</div><div class="kpi-subtext">Expedited queue</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card medium"><div class="kpi-label">Medium Priority</div><div class="kpi-value">{med_t:,}</div><div class="kpi-subtext">Standard workflow</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="kpi-card low"><div class="kpi-label">Low Priority</div><div class="kpi-value">{low_t:,}</div><div class="kpi-subtext">General inquiries</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visual Distribution Row
col_left, col_right = st.columns([1, 1])

with col_left:
    fig_cat = plot_category_distribution(df, "Ticket Type")
    st.plotly_chart(fig_cat, use_container_width=True)
    st.caption("ℹ️ **Category Insight:** Tickets are balanced across 5 primary functional domains (Refund requests, Technical issues, Cancellation requests, Product inquiries, and Billing inquiries).")

with col_right:
    fig_prio = plot_priority_distribution(df, "Ticket Priority")
    st.plotly_chart(fig_prio, use_container_width=True)
    st.caption("ℹ️ **Priority Insight:** Distribution of operational severity ensures consistent training data across critical emergencies, high disruptions, and routine inquiries.")

st.markdown("<br>", unsafe_allow_html=True)

# Cross Tabulation Heatmap
st.markdown("### 🔀 **Category vs Priority Cross-Tabulation**")
fig_heat = plot_category_priority_heatmap(df, "Ticket Type", "Ticket Priority")
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("ℹ️ **Cross-Tabulation Analysis:** Depicts volume distribution of each priority level across support ticket categories. Enables managers to identify which categories generate the highest concentration of high-severity incidents.")

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Recent Records Table with Search & Filter
st.markdown("### 📋 **Support Ticket Registry Explorer**")
f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

with f_col1:
    search_q = st.text_input("Search tickets by subject, description, or customer:", "")
with f_col2:
    all_cats = ["All Categories"] + sorted(df["Ticket Type"].dropna().unique().tolist())
    selected_cat = st.selectbox("Filter Category:", all_cats)
with f_col3:
    all_prios = ["All Priorities"] + sorted(df["Ticket Priority"].dropna().unique().tolist())
    selected_prio = st.selectbox("Filter Priority:", all_prios)

# Filter Logic
filtered_df = df.copy()
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df["Ticket Type"] == selected_cat]
if selected_prio != "All Priorities":
    filtered_df = filtered_df[filtered_df["Ticket Priority"] == selected_prio]
if search_q.strip():
    q = search_q.strip().lower()
    mask = (
        filtered_df["Ticket Subject"].fillna("").str.lower().str.contains(q) |
        filtered_df["Ticket Description"].fillna("").str.lower().str.contains(q) |
        filtered_df["Customer Name"].fillna("").str.lower().str.contains(q)
    )
    filtered_df = filtered_df[mask]

st.markdown(f"**Showing {len(filtered_df):,} matching tickets**")

# Display columns
display_cols = ["Ticket ID", "Customer Name", "Ticket Subject", "Ticket Type", "Ticket Priority", "Ticket Status"]
if "Date of Purchase" in df.columns:
    display_cols.append("Date of Purchase")

st.dataframe(
    filtered_df[display_cols].head(100),
    use_container_width=True,
    height=380
)

