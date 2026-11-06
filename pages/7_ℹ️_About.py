"""
TicketIQ - Page 7: About Project, Methodology & Technology Stack
"""
import streamlit as st
import os

st.set_page_config(page_title="About | TicketIQ", page_icon="ℹ️", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">ℹ️ About TicketIQ</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">AI-Powered Support Ticket Classification & Prioritization System</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="prediction-box" style="margin-bottom:1.5rem;">
        <h3 style="color:#60A5FA; margin-bottom:8px;">Executive Project Summary</h3>
        <p style="color:#CBD5E1; font-size:0.95rem; line-height:1.6;">
            <b>TicketIQ</b> is a production-grade, end-to-end Machine Learning and Natural Language Processing (NLP) web platform designed for SaaS organizations, IT departments, and customer operations teams. It automatically understands the content of inbound customer tickets, categorizes the functional issue, and predicts the operational priority level to drive automated triage and prevent SLA breaches.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🎯 **Core Project Objectives**")
    st.markdown(
        """
        - **Automated Routing:** Eliminate slow, error-prone manual triage by routing tickets directly to appropriate departments (Billing, Tech, Cancellation, Product).
        - **Urgency Detection:** Rapidly identify mission-critical outages, security concerns, and financial disputes for immediate escalation.
        - **Zero External AI Costs:** 100% self-contained open-source Scikit-Learn and NLTK machine learning pipeline without dependence on paid proprietary APIs.
        - **Explainable Triage:** Surface the salient tokens, n-grams, and TF-IDF feature weights that influenced the algorithm's predictions.
        - **Operational Business Intelligence:** Track volume trends, CSAT correlation, and cross-tabulation patterns to improve resolution efficiency.
        """
    )

with c2:
    st.markdown("### 🛠️ **Technology Stack**")
    st.markdown(
        """
        - **Programming Language:** Python 3.10+
        - **Interactive UI Framework:** Streamlit (Multi-page Enterprise Layout)
        - **Machine Learning Core:** Scikit-Learn
          - Multinomial Naive Bayes
          - Regularized Logistic Regression
          - Linear Support Vector Machine (Calibrated with `CalibratedClassifierCV`)
          - Random Forest Classifier
          - Decision Tree Classifier
        - **Natural Language Processing:** NLTK, TF-IDF Vectorizer, Bag-of-Words, Lemmatization, Stopword Filtering with Urgency Preservation
        - **Data Manipulation:** Pandas, NumPy
        - **Visualizations:** Plotly Interactive Heatmaps, Donut Charts, Bar Charts, WordCloud
        - **Model Persistence:** Joblib Pipelines
        """
    )

st.markdown("---")

st.markdown("### 🔄 **System Architecture & ML Pipeline**")
st.markdown(
    """
    ```mermaid
    flowchart LR
        A[Inbound Support Ticket] --> B[NLP Preprocessing Pipeline]
        B --> C[Stopword Filtering with Urgency Whitelist]
        C --> D[WordNet Lemmatization]
        D --> E[TF-IDF / N-gram Feature Extraction]
        E --> F[Category Classification Pipeline]
        E --> G[Priority Prediction Pipeline]
        F --> H[Class & Confidence Scores]
        G --> I[SLA Target & Automated Dispatch]
    ```
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("### ⚖️ **Priority Triage Business Rules**")
pr1, pr2, pr3, pr4 = st.columns(4)
with pr1:
    st.markdown(
        """
        <div style="background:#151F32; padding:14px; border-radius:10px; border-top:3px solid #EF4444;">
            <h4 style="color:#EF4444; margin-bottom:4px;">🚨 Critical (&lt; 2h SLA)</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">Complete platform outages, security breaches, data loss, payment system down, all users blocked.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with pr2:
    st.markdown(
        """
        <div style="background:#151F32; padding:14px; border-radius:10px; border-top:3px solid #F97316;">
            <h4 style="color:#F97316; margin-bottom:4px;">🔥 High (&lt; 6h SLA)</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">Account lockout, duplicate charges, broken core features, urgent business disruption with no workaround.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with pr3:
    st.markdown(
        """
        <div style="background:#151F32; padding:14px; border-radius:10px; border-top:3px solid #F59E0B;">
            <h4 style="color:#F59E0B; margin-bottom:4px;">⚡ Medium (&lt; 24h SLA)</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">Intermittent glitches, configuration issues, delayed operations, minor usability defects with workarounds.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with pr4:
    st.markdown(
        """
        <div style="background:#151F32; padding:14px; border-radius:10px; border-top:3px solid #10B981;">
            <h4 style="color:#10B981; margin-bottom:4px;">🌱 Low (&lt; 48h SLA)</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">General inquiries, feature requests, documentation requests, non-urgent feedback.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("### 🎓 **Internship & Project Metadata**")
st.markdown(
    """
    - **Internship Organization:** Future Interns
    - **Project ID:** Project - 2
    - **Deliverable:** TicketIQ – AI-Powered Support Ticket Classification & Prioritization System
    - **License:** Open Source (MIT)
    """
)

