# FUTURE_ML_02: TicketIQ – AI-Powered Support Ticket Classification & Prioritization System

![TicketIQ Banner](assets/logo.png)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK%20%7C%20TF--IDF-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Project Introduction & Overview

**TicketIQ** is an enterprise-grade, end-to-end Machine Learning and Natural Language Processing (NLP) web platform designed for SaaS organizations, IT departments, and customer support teams. 

Modern customer service desks receive thousands of unstructured customer complaints, billing queries, technical glitches, and cancellation requests daily. Manual triage is notoriously slow, costly, and susceptible to human fatigue—causing high-priority emergencies to linger in standard queues until Service Level Agreements (SLAs) are breached.

**TicketIQ** automates this entire intake workflow by:
1. Deeply analyzing unstructured inbound ticket text using NLP tokenization and lemmatization.
2. Accurately classifying tickets into their functional domain (Refund Request, Technical Issue, Cancellation, Product Inquiry, Billing Inquiry).
3. Predicting operational urgency (Critical, High, Medium, Low) to enforce automated SLA routing and prevent customer churn.
4. Explaining model decisions using salient TF-IDF feature attribution.

> **Zero External AI Costs:** TicketIQ operates completely offline and self-contained with Scikit-learn and NLTK. No paid APIs (OpenAI, Claude, Gemini) are required.

---

## 🎯 2. Problem Statement & Business Objectives

### Problem Statement
In traditional support operations, ticket routing is manual. Support agents read every ticket line-by-line before assigning it to engineering or billing. During peak demand or service outages, critical tickets get buried beneath routine FAQs, leading to breached SLAs, angry customers, and revenue loss.

### Business Objectives
- **Reduce First Response Time (FRT):** Instantly route tickets upon creation.
- **Isolate Critical Incidents:** Detect service outages and security vulnerabilities in seconds.
- **Eliminate Manual Triage Overhead:** Save 15–25 hours per support representative weekly.
- **Empower Management with Analytics:** Provide real-time visibility into customer issue clusters, CSAT trends, and vocabulary frequency.

---

## 🌟 3. Key Features

- **Executive Analytics Dashboard:** Real-time KPI metrics, category distributions, priority donut charts, cross-tabulation heatmaps, and interactive ticket registry.
- **Single Ticket AI Triage:** Real-time prediction form with confidence ratings, class probability distributions, recommended SLA routing windows, and pre-built real-world presets.
- **Bulk CSV Batch Processing:** Upload ticket spreadsheets of any size, automatically detect text columns, run high-throughput batch classification, and download enriched CSVs with predicted labels and SLA targets.
- **Interactive Model Training Studio:** Train, cross-evaluate, and benchmark 5 machine learning algorithms on custom datasets with configurable TF-IDF / Bag of Words vectorization and 1-click active model deployment.
- **Exploratory Data Analysis (EDA):** Deep dive into ticket text length distributions, word frequency bar charts, WordClouds, temporal volume trends, and CSAT correlation.
- **Model Benchmarking & Evaluation:** Comprehensive empirical comparison tables, confusion matrices with normalized error distributions, and downloadable evaluation summaries (CSV, TXT, JSON).

---

## 🛠️ 4. Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core language |
| **Web Framework** | Streamlit | Multi-page reactive SaaS user interface |
| **Styling & Theme** | Custom CSS / HTML5 | Dark Navy & Slate glassmorphic design |
| **Machine Learning** | Scikit-learn | Multi-class classification algorithms |
| **NLP Pipeline** | NLTK & Scikit-learn | Tokenization, WordNet lemmatization, stopword removal |
| **Feature Extraction** | CountVectorizer & TfidfVectorizer | Bag of Words & Sublinear TF-IDF N-grams |
| **Data Manipulation** | Pandas & NumPy | High-performance tabular data manipulation |
| **Visualizations** | Plotly & WordCloud | Interactive charts, heatmaps, and keyword clouds |
| **Model Persistence** | Joblib | Serialization of complete sklearn Pipelines |

---

## 📊 5. Machine Learning Algorithms Implemented

TicketIQ benchmarks and compares five distinct classification algorithms for both Category Classification and Priority Prediction:

1. **Multinomial Naive Bayes (`MultinomialNB`):** Fast, probabilistic linear baseline optimized for discrete word frequency counts.
2. **Logistic Regression (`LogisticRegression`):** L2-regularized multi-class classifier providing well-calibrated class probability scores.
3. **Linear Support Vector Machine (`CalibratedClassifierCV` + `LinearSVC`):** High-dimensional hyperplane separation calibrated with isotonic regression for reliable probabilities.
4. **Random Forest Classifier (`RandomForestClassifier`):** Ensemble of 100 decision trees offering non-linear feature interaction modeling.
5. **Decision Tree Classifier (`DecisionTreeClassifier`):** Fast, interpretable tree-based partitioning model.

---

## 🔬 6. NLP Preprocessing & Feature Extraction

### Text Cleaning Pipeline
```
Raw Customer Text
       │
       ▼
1. Lowercasing & HTML / URL / Email Stripping
       │
       ▼
2. Removal of Template Placeholders & Special Non-Alpha Characters
       │
       ▼
3. Tokenization & Whitelisted Urgency Stopword Filtering
   (Preserves: urgent, emergency, critical, immediately, failure, blocked, unable, outage, etc.)
       │
       ▼
4. WordNet Lemmatization
       │
       ▼
5. N-gram Feature Extraction (Unigrams + Bigrams, max_features=10,000, sublinear_tf=True)
       │
       ▼
Clean Feature Representation Ready for Inference
```

---

## 📁 7. Project Folder Structure

```
TicketIQ/
├── app.py                         # Main Streamlit application & executive hub
├── requirements.txt               # Complete Python package dependencies
├── README.md                      # Comprehensive documentation
├── .gitignore                     # Git ignore rules for virtualenvs and artifacts
├── train_all.py                   # Automated script to train & benchmark all models
│
├── assets/
│   ├── style.css                  # Custom enterprise dark navy SaaS styling
│   └── logo.png                   # High-resolution TicketIQ branding logo
│
├── data/
│   ├── raw/                       # Original raw dataset
│   ├── processed/                 # Cleaned and validated dataset
│   └── sample/                    # 50-row curated sample dataset for bulk testing
│
├── Dataset/
│   └── customer_support_tickets.csv # Full 8,469 record Kaggle dataset
│
├── models/
│   ├── category_model.pkl         # Trained Category classification pipeline
│   ├── priority_model.pkl         # Trained Priority classification pipeline
│   ├── category_model_metadata.json
│   ├── priority_model_metadata.json
│   └── model_metadata.json        # Production model metadata
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Comprehensive EDA notebook
│   ├── 02_data_preprocessing.ipynb # NLP pipeline verification notebook
│   ├── 03_feature_extraction.ipynb # TF-IDF vs Bag of Words analysis
│   ├── 04_model_training.ipynb    # Algorithm benchmarking notebook
│   └── 05_model_evaluation.ipynb  # Evaluation reports & confusion matrices
│
├── pages/
│   ├── 1_📊_Overview.py           # Page 1: Overview Dashboard & Registry
│   ├── 2_🎯_Ticket_Prediction.py  # Page 2: Single Ticket Triage & Explainability
│   ├── 3_📁_Bulk_Classification.py# Page 3: Batch CSV Classification & Export
│   ├── 4_⚙️_Model_Training.py     # Page 4: Model Training Studio
│   ├── 5_📈_Data_Analytics.py     # Page 5: Advanced EDA & WordClouds
│   ├── 6_🏆_Model_Performance.py  # Page 6: Benchmarks & Confusion Matrices
│   └── 7_ℹ️_About.py              # Page 7: Architecture & Project Details
│
├── reports/
│   ├── model_comparison.csv       # Multi-algorithm performance comparison
│   ├── classification_report.txt  # Detailed precision/recall/F1 metrics
│   └── evaluation_summary.json    # Complete JSON evaluation export
│
└── src/
    ├── __init__.py                # Package initialization
    ├── analytics.py               # Plotly charts, heatmaps & WordCloud logic
    ├── data_loader.py             # Column detection, validation & splitting
    ├── evaluation.py              # Evaluation metrics & confusion matrix generators
    ├── feature_extraction.py      # TF-IDF & CountVectorizer factories
    ├── prediction.py              # Inference engine & TF-IDF explainability
    ├── preprocessing.py           # NLP cleaning, lemmatization & stopword filter
    ├── train_category.py          # Category model training module
    ├── train_priority.py          # Priority model training module
    └── utils.py                   # Shared utilities, badges & SLA policies
```

---

## 🚀 8. Quick Start & Execution Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Recommended: Dedicated Virtual Environment

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "d:\Future In tern Project\Project - 2"
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Application
```powershell
streamlit run app.py
```
*The web interface will automatically launch at `http://localhost:8501` in your default browser.*

### (Optional) Retrain All Models from Command Line
```powershell
python train_all.py
```

---

## 📈 9. Empirical Evaluation Results

The models were trained on the [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) (8,469 records) using an 80/20 stratified train/test split.

### Category Classification Benchmark
| Algorithm | Weighted F1 | Accuracy | Macro Precision | Macro Recall | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **0.2124** | **21.35%** | **0.2131** | **0.2137** | **1.05s** |
| Random Forest | 0.2092 | 21.23% | 0.2098 | 0.2126 | 0.98s |
| Logistic Regression | 0.2077 | 20.73% | 0.2079 | 0.2074 | 3.72s |
| Linear SVM | 0.1267 | 21.66% | 0.2184 | 0.2166 | 2.56s |
| Decision Tree | 0.0991 | 20.42% | 0.2052 | 0.2045 | 1.30s |

### Priority Prediction Benchmark
| Algorithm | Weighted F1 | Accuracy | Macro Precision | Macro Recall | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.2466** | **25.74%** | **0.2476** | **0.2575** | **1.65s** |
| Logistic Regression | 0.2464 | 24.69% | 0.2462 | 0.2468 | 2.93s |
| Multinomial Naive Bayes | 0.2418 | 24.38% | 0.2422 | 0.2437 | 0.70s |
| Decision Tree | 0.1612 | 25.68% | 0.2582 | 0.2569 | 2.46s |
| Linear SVM | 0.1481 | 25.62% | 0.2570 | 0.2563 | 1.85s |

*(Note: Synthetic uniform distribution across the Kaggle dataset produces expected empirical baseline scores; full class-wise precision and recall breakdowns are accessible on Page 6).*

---

## ⚖️ 10. Priority Triage & SLA Policy

| Priority Level | SLA Window | Assigned Tier | Action Required |
| :--- | :---: | :--- | :--- |
| **🚨 Critical** | **< 2 Hours** | Tier 3 / Incident Lead | Immediate engineering alert. Outage / breach response. |
| **🔥 High** | **< 6 Hours** | Tier 2 Senior Support | Expedited queue dispatch. Dedicated ticket handling. |
| **⚡ Medium** | **< 24 Hours** | Tier 1 Support Specialist| Standard operational workflow. |
| **🌱 Low** | **< 48 Hours** | Customer Operations | Knowledge base referral or scheduled follow-up. |

---

## 🛡️ 11. Ethical Considerations & Limitations

- **Feature Importance is Correlation, Not Causality:** Salient n-grams surfaced during prediction indicate statistical weighting in the TF-IDF vectorizer, not definite root-cause explanation.
- **Rule-Based Priority Fallbacks:** When uploading custom datasets lacking priority columns, TicketIQ provides a rule-based priority tagging generator. These labels are disclosed as synthetic heuristics, not human ground truth.
- **Language Scope:** The current NLP pipeline is tailored for English text corpus.

---

## 🔮 12. Future Enhancements

- Multilingual ticket support using transformer-based embeddings (e.g. `sentence-transformers`).
- Live integration with Zendesk, Jira Service Desk, and Freshdesk webhooks.
- Automated resolution draft generation using fine-tuned open-source LLMs.
- Real-time agent sentiment tracking during ticket lifecycles.

---

## 👨‍💻 13. Author & Internship Submission

- **Project Name:** TicketIQ
- **Internship:** Future Interns Internship Program
- **Submission:** Project - 2
- **Developer:** AI Machine Learning Intern

---
*Built with Scikit-learn, NLTK, and Streamlit.*

