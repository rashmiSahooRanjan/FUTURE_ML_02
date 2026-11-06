"""
TicketIQ – AI-Powered Support Ticket Classification & Prioritization System
Main Application Entry Point & Executive Dashboard Hub
"""
import streamlit as st
import pandas as pd
import os
import json

from src.utils import load_json, get_priority_badge
from src.prediction import predict_single_ticket

# Configure Page
st.set_page_config(
    page_title="TicketIQ | AI Support Ticket Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Load Model Metadata
metadata = load_json("models/model_metadata.json")
best_cat_model = metadata.get("best_category_model", "Multinomial Naive Bayes")
best_prio_model = metadata.get("best_priority_model", "Random Forest")
cat_acc = metadata.get("category_accuracy", 0.0)
prio_acc = metadata.get("priority_accuracy", 0.0)

# Sidebar
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=220)
    else:
        st.markdown("## ⚡ **TicketIQ**")
    
    st.markdown("### **System Status**")
    st.markdown(
        """
        <div style="background:#151F32; padding:12px; border-radius:10px; border:1px solid #1E293B; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#94A3B8;">Inference Engine</span>
                <span style="background:#10B98122; color:#10B981; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">● ONLINE</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#94A3B8;">NLP Pipeline</span>
                <span style="color:#60A5FA; font-size:0.8rem; font-weight:600;">TF-IDF + NLTK</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.8rem; color:#94A3B8;">External API Dependency</span>
                <span style="color:#10B981; font-size:0.8rem; font-weight:600;">None (0$ Free)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### **Active ML Pipelines**")
    st.markdown(f"**Category Classifier:** `{best_cat_model}`")
    st.markdown(f"**Priority Classifier:** `{best_prio_model}`")
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#64748B; line-height:1.5;">
            <b>TicketIQ v1.0.0</b><br>
            Production-grade Machine Learning & Natural Language Processing Platform.
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Hero Header
st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
            <span style="background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Enterprise ML Suite</span>
            <span style="background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Production Trained</span>
        </div>
        <div class="hero-title">TicketIQ – Support Ticket Intelligence Platform</div>
        <div class="hero-subtitle">Automated natural language ticket routing, multi-class issue classification, and algorithmic SLA prioritization.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Load dataset summary for KPI cards
@st.cache_data
def load_summary_stats():
    raw_path = "data/raw/customer_support_tickets.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        total = len(df)
        cats = df["Ticket Type"].nunique() if "Ticket Type" in df.columns else 5
        crit = (df["Ticket Priority"] == "Critical").sum() if "Ticket Priority" in df.columns else 0
        high = (df["Ticket Priority"] == "High").sum() if "Ticket Priority" in df.columns else 0
        med = (df["Ticket Priority"] == "Medium").sum() if "Ticket Priority" in df.columns else 0
        low = (df["Ticket Priority"] == "Low").sum() if "Ticket Priority" in df.columns else 0
        return total, cats, crit, high, med, low
    return 8469, 5, 2129, 2085, 2192, 2063

total_tickets, total_cats, crit_count, high_count, med_count, low_count = load_summary_stats()

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Corpus Tickets</div>
            <div class="kpi-value">{total_tickets:,}</div>
            <div class="kpi-subtext">Enterprise training corpus</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="kpi-card critical">
            <div class="kpi-label">Critical / Urgent</div>
            <div class="kpi-value">{crit_count:,}</div>
            <div class="kpi-subtext">&lt; 2 Hr Urgent SLA Target</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class="kpi-card high">
            <div class="kpi-label">High Priority</div>
            <div class="kpi-value">{high_count:,}</div>
            <div class="kpi-subtext">&lt; 6 Hr Expedited SLA</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"""
        <div class="kpi-card purple">
            <div class="kpi-label">Issue Categories</div>
            <div class="kpi-value">{total_cats}</div>
            <div class="kpi-subtext">Multi-class classifications</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col5:
    st.markdown(
        f"""
        <div class="kpi-card low">
            <div class="kpi-label">Active Classifier</div>
            <div class="kpi-value" style="font-size:1.25rem; padding-top:0.35rem;">{best_cat_model}</div>
            <div class="kpi-subtext">Optimized with TF-IDF N-grams</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Quick Triage Playground
st.markdown("### ⚡ **Instant Ticket Triage Playground**")
st.markdown("Test the production NLP classification pipelines directly by entering or selecting sample customer issues.")

sample_presets = {
    "Select a preset scenario...": ("", ""),
    "🚨 Emergency Service Outage": (
        "Complete Service Outage - Server 500 Error",
        "Our entire company cannot access the platform. We are getting a 500 internal server error. Immediate escalation needed, business operations completely blocked!"
    ),
    "💳 Double Billing Dispute": (
        "Incorrect charge on subscription invoice",
        "I noticed on my bank statement that I was charged twice for this month's renewal. Please review my account and process a refund for the duplicate transaction."
    ),
    "🔧 Product Setup & Sync Issue": (
        "Hardware pairing failure with bluetooth",
        "Having trouble connecting the device to my laptop. The pairing light blinks for 30 seconds then turns red. Followed the manual troubleshooting guide without success."
    ),
    "❓ Feature Inquiry": (
        "Inquiry regarding upcoming API webhooks",
        "Could you please share documentation on the new webhook events and rate limits? Planning an integration next quarter."
    )
}

selected_preset = st.selectbox("Choose a real-world scenario preset:", list(sample_presets.keys()))
default_subject, default_desc = sample_presets[selected_preset]

with st.form("quick_triage_form"):
    q_subject = st.text_input("Ticket Subject", value=default_subject, placeholder="e.g. Cannot log in to administration dashboard")
    q_desc = st.text_area("Ticket Description", value=default_desc, placeholder="e.g. Getting authentication failure error whenever attempting to login...", height=110)
    submitted = st.form_submit_button("⚡ Run AI Triage & Predict", use_container_width=True)

if submitted:
    if not q_desc.strip():
        st.warning("Please provide a ticket description to predict.")
    else:
        with st.spinner("Analyzing text using trained NLP Pipeline..."):
            try:
                res = predict_single_ticket(q_subject, q_desc)
                
                cat_name = res["category"]["prediction"]
                cat_conf = res["category"]["confidence"]
                prio_name = res["priority"]["prediction"]
                prio_conf = res["priority"]["confidence"]
                sla = res["sla"]
                
                prio_badge = get_priority_badge(prio_name)
                
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #1E293B; padding-bottom:12px; margin-bottom:14px;">
                            <div>
                                <span style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; font-weight:700;">Predicted Category</span>
                                <div style="font-size:1.6rem; font-weight:800; color:#60A5FA;">{cat_name}</div>
                                <span style="font-size:0.8rem; color:#94A3B8;">Model Confidence: <b>{cat_conf*100:.1f}%</b></span>
                            </div>
                            <div style="text-align:right;">
                                <span style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; font-weight:700;">Assigned Priority</span>
                                <div style="margin-top:4px;">{prio_badge}</div>
                                <div style="font-size:0.8rem; color:#94A3B8; margin-top:6px;">Model Confidence: <b>{prio_conf*100:.1f}%</b></div>
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:12px;">
                            <div style="background:#0F172A; padding:12px; border-radius:8px; border:1px solid #1E293B;">
                                <div style="font-size:0.75rem; color:#94A3B8; font-weight:600; text-transform:uppercase;">Recommended SLA Target</div>
                                <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-top:2px;">{sla.get('target', 'Standard SLA')}</div>
                                <div style="font-size:0.8rem; color:#CBD5E1; margin-top:4px;">Routing: <b>{sla.get('tier', 'Tier 1 Support')}</b></div>
                            </div>
                            <div style="background:#0F172A; padding:12px; border-radius:8px; border:1px solid #1E293B;">
                                <div style="font-size:0.75rem; color:#94A3B8; font-weight:600; text-transform:uppercase;">Key Influential Terms</div>
                                <div style="font-size:0.88rem; color:#F8FAFC; margin-top:4px;">
                                    {", ".join([f"<code>{w}</code>" for w, _ in (res['category']['top_keywords'] + res['priority']['top_keywords'])[:5]]) or 'Domain vocabulary matching'}
                                </div>
                            </div>
                        </div>
                        <div style="font-size:0.85rem; color:#94A3B8; font-style:italic;">
                            {res['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

st.markdown("---")

# Feature Tiles
st.markdown("### 🧭 **Platform Navigation & Capabilities**")
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📊 Overview & EDA</h4>
            <p style="color:#94A3B8; font-size:0.88rem;">Real-time metrics, priority breakdown, category distribution, and cross-tabulation heatmaps of customer tickets.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with n2:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🎯 Single & Bulk Triage</h4>
            <p style="color:#94A3B8; font-size:0.88rem;">Score individual tickets or upload entire CSV spreadsheets to classify thousands of tickets with confidence ratings.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with n3:
    st.markdown(
        """
        <div class="feature-card">
            <h4>⚙️ Model Training & Benchmark</h4>
            <p style="color:#94A3B8; font-size:0.88rem;">Train, compare, and benchmark 5 ML algorithms with TF-IDF / Bag of Words, confusion matrices, and 1-click model deployment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

