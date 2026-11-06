"""
TicketIQ - Page 4: Model Training Studio & Hyperparameter Lab
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from src.data_loader import detect_columns, load_and_validate_dataset
from src.train_category import train_and_compare_category_models
from src.train_priority import train_and_compare_priority_models
from src.evaluation import create_confusion_matrix_fig, create_model_comparison_bar_chart
from src.utils import save_pickle, save_json

st.set_page_config(page_title="Model Training | TicketIQ", page_icon="⚙️", layout="wide")

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin-bottom:4px;">⚙️ Machine Learning Training Studio</h1>
        <p style="color:#94A3B8; font-size:1.05rem;">Train, cross-evaluate, and benchmark classification algorithms with configurable vectorization and hyperparameter controls.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Dataset Selection
st.markdown("### 1. 📂 **Select Training Corpus**")
data_source = st.radio("Choose dataset source:", ["Use Default Corpus (data/raw/customer_support_tickets.csv)", "Upload New Dataset (CSV)"], horizontal=True)

df_train = None
if data_source.startswith("Use Default"):
    default_p = "data/raw/customer_support_tickets.csv"
    if not os.path.exists(default_p):
        default_p = "Dataset/customer_support_tickets.csv"
    if os.path.exists(default_p):
        df_train = pd.read_csv(default_p)
        st.success(f"Loaded default training dataset with **{len(df_train):,} tickets**.")
    else:
        st.error("Default dataset not found.")
else:
    uploaded = st.file_uploader("Upload custom support tickets CSV", type=["csv"])
    if uploaded:
        df_train = pd.read_csv(uploaded)
        st.success(f"Uploaded dataset loaded with **{len(df_train):,} tickets**.")

if df_train is not None:
    st.markdown("---")
    st.markdown("### 2. 🎛️ **Schema Mapping & Target Selection**")
    
    detected = detect_columns(df_train.columns.tolist())
    all_cols = df_train.columns.tolist()
    
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        d_idx = all_cols.index(detected["description"]) if detected["description"] in all_cols else 0
        desc_col = st.selectbox("Description Column *", all_cols, index=d_idx)
    with mc2:
        c_idx = all_cols.index(detected["category"]) if detected["category"] in all_cols else 0
        cat_col = st.selectbox("Category Target Column *", all_cols, index=c_idx)
    with mc3:
        p_idx = all_cols.index(detected["priority"]) if detected["priority"] in all_cols else 0
        prio_col = st.selectbox("Priority Target Column *", all_cols, index=p_idx)
        
    st.markdown("---")
    st.markdown("### 3. 🔬 **NLP Feature Extraction & Algorithms Configuration**")
    
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        vec_type = st.selectbox("Vectorization Technique", ["tfidf", "bag_of_words"], format_func=lambda x: "TF-IDF (Term Frequency - Inverse Doc Frequency)" if x == "tfidf" else "Bag of Words (CountVectorizer)")
    with hc2:
        max_feats = st.slider("Max Vocabulary Features", min_value=1000, max_value=20000, value=10000, step=1000)
    with hc3:
        test_split = st.slider("Test Evaluation Split", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
        
    st.markdown("#### **Select Algorithms to Benchmark:**")
    all_algo_names = [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Linear SVM",
        "Random Forest",
        "Decision Tree"
    ]
    
    algo_cols = st.columns(5)
    selected_algos = []
    for i, name in enumerate(all_algo_names):
        with algo_cols[i]:
            if st.checkbox(name, value=True, key=f"algo_{i}"):
                selected_algos.append(name)

    if not selected_algos:
        st.warning("Please select at least one algorithm.")

    # Train Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Training & Benchmark Pipeline", use_container_width=True, disabled=(len(selected_algos) == 0)):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("1/3: Preprocessing text and splitting data...")
        progress_bar.progress(15)
        
        try:
            # Category Training
            status_text.text(f"2/3: Training Category Models ({', '.join(selected_algos)})...")
            cat_pipe, cat_comparison, cat_best_eval, cat_all = train_and_compare_category_models(
                df=df_train,
                text_column=desc_col,
                category_column=cat_col,
                vectorizer_type=vec_type,
                max_features=max_feats,
                test_size=test_split,
                models_to_train=selected_algos
            )
            progress_bar.progress(60)
            
            # Priority Training
            status_text.text(f"3/3: Training Priority Models ({', '.join(selected_algos)})...")
            prio_pipe, prio_comparison, prio_best_eval, prio_all = train_and_compare_priority_models(
                df=df_train,
                text_column=desc_col,
                priority_column=prio_col,
                vectorizer_type=vec_type,
                max_features=max_feats,
                test_size=test_split,
                models_to_train=selected_algos
            )
            progress_bar.progress(100)
            status_text.success(" Training and benchmarking completed successfully!")
            
            st.session_state["training_cat_comparison"] = cat_comparison
            st.session_state["training_prio_comparison"] = prio_comparison
            st.session_state["training_cat_best"] = cat_best_eval
            st.session_state["training_prio_best"] = prio_best_eval
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

# Display Benchmark Results
if "training_cat_comparison" in st.session_state:
    cat_comp = st.session_state["training_cat_comparison"]
    prio_comp = st.session_state["training_prio_comparison"]
    cat_best = st.session_state["training_cat_best"]
    prio_best = st.session_state["training_prio_best"]
    
    st.markdown("---")
    st.markdown("### 🏆 **Training Studio Benchmark Results**")
    
    t_tab1, t_tab2 = st.tabs(["🏷️ Category Classification", "⚡ Priority Classification"])
    
    with t_tab1:
        st.markdown(f"**Best Performing Category Model:** `{cat_best['model_name']}` (Weighted F1: `{cat_best['f1_weighted']:.4f}` | Accuracy: `{cat_best['accuracy']*100:.2f}%`)")
        st.dataframe(cat_comp, use_container_width=True)
        
        st.plotly_chart(create_model_comparison_bar_chart(cat_comp, "f1_weighted"), use_container_width=True)
        
        if "confusion_matrix" in cat_best:
            cm_fig = create_confusion_matrix_fig(
                cat_best["confusion_matrix"],
                cat_best["classes"],
                f"Confusion Matrix - {cat_best['model_name']}"
            )
            st.plotly_chart(cm_fig, use_container_width=True)

    with t_tab2:
        st.markdown(f"**Best Performing Priority Model:** `{prio_best['model_name']}` (Weighted F1: `{prio_best['f1_weighted']:.4f}` | Accuracy: `{prio_best['accuracy']*100:.2f}%`)")
        st.dataframe(prio_comp, use_container_width=True)
        
        st.plotly_chart(create_model_comparison_bar_chart(prio_comp, "f1_weighted"), use_container_width=True)
        
        if "confusion_matrix" in prio_best:
            cm_fig_prio = create_confusion_matrix_fig(
                prio_best["confusion_matrix"],
                prio_best["classes"],
                f"Confusion Matrix - {prio_best['model_name']}"
            )
            st.plotly_chart(cm_fig_prio, use_container_width=True)

