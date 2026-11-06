"""
TicketIQ - Page 2: Single Ticket AI Prediction & Explainability
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.prediction import predict_single_ticket
from src.utils import get_priority_badge

st.set_page_config(page_title="Ticket Prediction | TicketIQ", page_icon="🎯", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">🎯 Real-Time Support Ticket Prediction</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">Process individual support issues through our trained multi-class NLP classifiers for instant classification and SLA triage.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Presets for fast testing
preset_cases = {
    "Custom Ticket (Enter manually below)": {
        "subject": "",
        "desc": "",
        "product": "SaaS Platform",
        "impact": "Single User"
    },
    "🚨 Critical Server Outage": {
        "subject": "System down - 502 Bad Gateway across all endpoints",
        "desc": "Our team is completely unable to access the production database. All API requests are failing with 502 Bad Gateway. Business is completely halted, please escalate urgently!",
        "product": "Cloud API Gateway",
        "impact": "Organization Wide"
    },
    "💰 Billing & Overcharge Dispute": {
        "subject": "Charged twice on latest monthly billing renewal",
        "desc": "I was charged $120 twice on November 1st for the same subscription renewal. Please reverse the duplicate debit and email an updated receipt.",
        "product": "Enterprise Subscription",
        "impact": "Single User"
    },
    "🛠️ Hardware / Setup Malfunction": {
        "subject": "Device not powering on after firmware update",
        "desc": "I initiated the firmware installation following the manual instructions. The device rebooted and has since remained un-responsive with a blinking orange indicator. Need technical assistance.",
        "product": "Smart Hardware Hub",
        "impact": "Single User"
    },
    "❌ Subscription Cancellation": {
        "subject": "Request to cancel annual contract immediately",
        "desc": "Please cancel our company subscription effective at the end of this billing cycle. We are migrating to a different vendor and do not want auto-renewal.",
        "product": "Enterprise Plan",
        "impact": "Departmental"
    }
}

selected_preset = st.selectbox("⚡ Quick Test Presets (Select to auto-populate form):", list(preset_cases.keys()))
active_case = preset_cases[selected_preset]

# Input Form
with st.form("single_ticket_form"):
    st.markdown("### 📝 **Ticket Information**")
    c1, c2 = st.columns(2)
    with c1:
        subject_in = st.text_input("Ticket Subject / Title", value=active_case["subject"], placeholder="Brief issue headline (e.g., Cannot login to account)")
    with c2:
        product_in = st.selectbox("Product / Service Purchased", ["SaaS Platform", "Cloud API Gateway", "Enterprise Plan", "Smart Hardware Hub", "GoPro Hero", "LG Smart TV", "PlayStation 5", "Other"], index=0)
        
    desc_in = st.text_area("Ticket Description *", value=active_case["desc"], placeholder="Detailed description of the customer issue, symptoms, and urgency indicators...", height=140)
    
    with st.expander("Additional Metadata (Optional Context)", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            cust_id = st.text_input("Customer ID / Email", "CUST-98214")
        with ec2:
            impact_level = st.selectbox("Customer Reported Impact", ["Single User", "Departmental", "Organization Wide"], index=0)
        with ec3:
            affected_users = st.number_input("Estimated Affected Users", min_value=1, max_value=10000, value=1)
            
    submit_btn = st.form_submit_button("🚀 Run AI Triage & Predict", use_container_width=True)

# Process Prediction
if submit_btn:
    if not desc_in or not desc_in.strip():
        st.error("⚠️ Ticket Description is mandatory for prediction. Please provide text.")
    else:
        with st.spinner("Executing NLP Preprocessing and Model Inference..."):
            try:
                res = predict_single_ticket(subject=subject_in, description=desc_in, product=product_in)
                
                cat = res["category"]
                prio = res["priority"]
                sla = res["sla"]
                
                st.success("✅ Prediction executed successfully via trained machine learning models!")
                
                # Primary Prediction Cards
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <span style="font-size:0.8rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Predicted Issue Category</span>
                            <div style="font-size:2.0rem; font-weight:800; color:#60A5FA; margin-top:4px;">{cat['prediction']}</div>
                            <div style="margin-top:6px; font-size:0.9rem; color:#CBD5E1;">
                                Model Confidence: <b style="color:#FFFFFF;">{cat['confidence']*100:.1f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Category Probabilities Chart
                    if cat["probabilities"]:
                        prob_df = pd.DataFrame(list(cat["probabilities"].items()), columns=["Category", "Probability"])
                        prob_df["Percentage"] = prob_df["Probability"] * 100
                        prob_df = prob_df.sort_values(by="Probability", ascending=True)
                        
                        fig_cat_p = px.bar(
                            prob_df,
                            x="Percentage",
                            y="Category",
                            orientation="h",
                            text="Percentage",
                            title="Category Probability Distribution (%)"
                        )
                        fig_cat_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_color="#3B82F6")
                        fig_cat_p.update_layout(
                            paper_bgcolor="#151F32",
                            plot_bgcolor="#0B1120",
                            font=dict(family="Inter, sans-serif", color="#F8FAFC"),
                            xaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#1E293B"),
                            yaxis=dict(title="", gridcolor="#1E293B"),
                            height=250,
                            margin=dict(l=20, r=40, t=40, b=20)
                        )
                        st.plotly_chart(fig_cat_p, use_container_width=True)

                with res_col2:
                    prio_badge = get_priority_badge(prio['prediction'])
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <span style="font-size:0.8rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Predicted Priority Level</span>
                            <div style="margin-top:8px;">{prio_badge}</div>
                            <div style="margin-top:14px; font-size:0.9rem; color:#CBD5E1;">
                                Model Confidence: <b style="color:#FFFFFF;">{prio['confidence']*100:.1f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Priority Probabilities Chart
                    if prio["probabilities"]:
                        prio_prob_df = pd.DataFrame(list(prio["probabilities"].items()), columns=["Priority", "Probability"])
                        prio_prob_df["Percentage"] = prio_prob_df["Probability"] * 100
                        prio_prob_df = prio_prob_df.sort_values(by="Probability", ascending=True)
                        
                        fig_prio_p = px.bar(
                            prio_prob_df,
                            x="Percentage",
                            y="Priority",
                            orientation="h",
                            text="Percentage",
                            title="Priority Probability Distribution (%)"
                        )
                        fig_prio_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_color="#F59E0B")
                        fig_prio_p.update_layout(
                            paper_bgcolor="#151F32",
                            plot_bgcolor="#0B1120",
                            font=dict(family="Inter, sans-serif", color="#F8FAFC"),
                            xaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#1E293B"),
                            yaxis=dict(title="", gridcolor="#1E293B"),
                            height=250,
                            margin=dict(l=20, r=40, t=40, b=20)
                        )
                        st.plotly_chart(fig_prio_p, use_container_width=True)

                st.markdown("---")
                
                # SLA and Actionable Resolution Card
                st.markdown("### ⏱️ **Recommended SLA & Dispatch Strategy**")
                sla_c1, sla_c2, sla_c3 = st.columns(3)
                with sla_c1:
                    st.markdown(
                        f"""
                        <div style="background:#151F32; padding:15px; border-radius:12px; border:1px solid #1E293B;">
                            <div style="font-size:0.75rem; color:#94A3B8; font-weight:700; text-transform:uppercase;">Resolution Target</div>
                            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin-top:4px;">{sla.get('target', 'Standard SLA')}</div>
                            <div style="font-size:0.8rem; color:#60A5FA; margin-top:4px;">Window: &lt; {sla.get('sla_hours', 24)} Hours</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with sla_c2:
                    st.markdown(
                        f"""
                        <div style="background:#151F32; padding:15px; border-radius:12px; border:1px solid #1E293B;">
                            <div style="font-size:0.75rem; color:#94A3B8; font-weight:700; text-transform:uppercase;">Assigned Support Tier</div>
                            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin-top:4px;">{sla.get('tier', 'Tier 1 Support')}</div>
                            <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">Automated Skill Routing</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with sla_c3:
                    st.markdown(
                        f"""
                        <div style="background:#151F32; padding:15px; border-radius:12px; border:1px solid #1E293B;">
                            <div style="font-size:0.75rem; color:#94A3B8; font-weight:700; text-transform:uppercase;">Standard Operating Procedure</div>
                            <div style="font-size:0.85rem; color:#CBD5E1; margin-top:4px;">{sla.get('action', 'Standard triage workflow.')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Model Explanation Section
                st.markdown("### 🔍 **Model Prediction Explainability**")
                st.markdown(res["explanation"])
                
                # Preprocessed tokens preview
                with st.expander("NLP Preprocessed Tokens (Cleaned Text)", expanded=False):
                    st.code(res["cleaned_text"], language="text")

            except Exception as e:
                st.error(f"Inference error occurred: {str(e)}")

