"""
TicketIQ - Prediction & Inference Engine with Feature Explainability
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from src.preprocessing import clean_text
from src.utils import load_pickle, load_json, get_sla_recommendation

# Global cached pipelines
_CATEGORY_PIPELINE = None
_PRIORITY_PIPELINE = None

def get_pipelines():
    """Load and cache category and priority pipelines."""
    global _CATEGORY_PIPELINE, _PRIORITY_PIPELINE
    if _CATEGORY_PIPELINE is None:
        _CATEGORY_PIPELINE = load_pickle("models/category_model.pkl")
    if _PRIORITY_PIPELINE is None:
        _PRIORITY_PIPELINE = load_pickle("models/priority_model.pkl")
    return _CATEGORY_PIPELINE, _PRIORITY_PIPELINE

def extract_influential_words(pipeline, cleaned_text: str, predicted_class: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Extract top words in the ticket that contributed most to the predicted class.
    Uses TF-IDF features and classifier weights.
    """
    try:
        vec = pipeline.named_steps.get("vectorizer")
        clf = pipeline.named_steps.get("classifier")
        
        if vec is None or clf is None:
            return []
            
        feature_names = vec.get_feature_names_out()
        words_in_text = set(cleaned_text.split())
        
        # Transform the single document
        tfidf_vec = vec.transform([cleaned_text]).toarray()[0]
        active_indices = np.where(tfidf_vec > 0)[0]
        
        if len(active_indices) == 0:
            return []
            
        # Get classifier weights for the predicted class
        sub_clf = clf
        if hasattr(sub_clf, "calibrated_classifiers_"):
            sub_clf = sub_clf.calibrated_classifiers_[0].estimator
            
        weights = None
        if hasattr(sub_clf, "coef_"):
            classes = list(clf.classes_)
            if predicted_class in classes:
                cls_idx = classes.index(predicted_class)
                weights = sub_clf.coef_[cls_idx]
        elif hasattr(sub_clf, "feature_importances_"):
            weights = sub_clf.feature_importances_
            
        scored_terms = []
        for idx in active_indices:
            term = feature_names[idx]
            tfidf_score = tfidf_vec[idx]
            weight = weights[idx] if weights is not None and idx < len(weights) else 1.0
            impact = tfidf_score * weight
            scored_terms.append((term, float(impact)))
            
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms[:top_k]
    except Exception:
        return []

def predict_single_ticket(
    subject: str,
    description: str,
    customer_id: Optional[str] = None,
    product: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full prediction for a single ticket.
    Returns predicted category, priority, confidence scores, and explanations.
    """
    cat_pipe, prio_pipe = get_pipelines()
    
    if not description or not str(description).strip():
        raise ValueError("Ticket description cannot be empty.")
        
    combined_raw = f"{str(subject).strip()} - {str(description).strip()}" if subject else str(description).strip()
    cleaned = clean_text(combined_raw)
    
    if not cleaned:
        cleaned = "inquiry assistance"
        
    response = {
        "raw_text": combined_raw,
        "cleaned_text": cleaned,
        "category": {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
            "top_keywords": []
        },
        "priority": {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
            "top_keywords": []
        },
        "sla": {},
        "explanation": ""
    }
    
    # 1. Category Prediction
    if cat_pipe is not None:
        cat_pred = cat_pipe.predict([cleaned])[0]
        response["category"]["prediction"] = str(cat_pred)
        
        if hasattr(cat_pipe, "predict_proba"):
            probs = cat_pipe.predict_proba([cleaned])[0]
            classes = list(cat_pipe.classes_)
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            response["category"]["probabilities"] = prob_dict
            response["category"]["confidence"] = round(float(np.max(probs)), 4)
            
        response["category"]["top_keywords"] = extract_influential_words(cat_pipe, cleaned, str(cat_pred))
        
    # 2. Priority Prediction
    if prio_pipe is not None:
        prio_pred = prio_pipe.predict([cleaned])[0]
        response["priority"]["prediction"] = str(prio_pred)
        
        if hasattr(prio_pipe, "predict_proba"):
            probs = prio_pipe.predict_proba([cleaned])[0]
            classes = list(prio_pipe.classes_)
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            response["priority"]["probabilities"] = prob_dict
            response["priority"]["confidence"] = round(float(np.max(probs)), 4)
            
        response["priority"]["top_keywords"] = extract_influential_words(prio_pipe, cleaned, str(prio_pred))
        response["sla"] = get_sla_recommendation(str(prio_pred))
        
    # Build human explanation
    cat_kw = [w for w, _ in response["category"]["top_keywords"]]
    prio_kw = [w for w, _ in response["priority"]["top_keywords"]]
    
    kw_str = ", ".join([f"'{k}'" for k in (cat_kw + prio_kw)[:5]])
    explanation = f"The trained model classified this ticket as **{response['category']['prediction']}** with priority **{response['priority']['prediction']}**."
    if kw_str:
        explanation += f" Salient keywords that influenced this determination include {kw_str}."
    explanation += " *(Note: Influential keywords represent model feature weights, not proof of real-world causality.)*"
    
    response["explanation"] = explanation
    return response

def predict_batch_tickets(
    df: pd.DataFrame,
    description_col: str,
    subject_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Run batch classification on a DataFrame of tickets.
    Returns a copy of DataFrame with prediction columns appended.
    """
    cat_pipe, prio_pipe = get_pipelines()
    out_df = df.copy()
    
    if subject_col and subject_col in df.columns:
        texts = out_df[subject_col].fillna("").astype(str) + " - " + out_df[description_col].fillna("").astype(str)
    else:
        texts = out_df[description_col].fillna("").astype(str)
        
    cleaned_texts = [clean_text(t) or "support inquiry" for t in texts]
    
    # Category predictions
    if cat_pipe is not None:
        out_df["Predicted_Category"] = cat_pipe.predict(cleaned_texts)
        if hasattr(cat_pipe, "predict_proba"):
            probs = cat_pipe.predict_proba(cleaned_texts)
            out_df["Category_Confidence"] = np.round(np.max(probs, axis=1), 4)
        else:
            out_df["Category_Confidence"] = 1.0
    else:
        out_df["Predicted_Category"] = "N/A"
        out_df["Category_Confidence"] = 0.0
        
    # Priority predictions
    if prio_pipe is not None:
        out_df["Predicted_Priority"] = prio_pipe.predict(cleaned_texts)
        if hasattr(prio_pipe, "predict_proba"):
            probs = prio_pipe.predict_proba(cleaned_texts)
            out_df["Priority_Confidence"] = np.round(np.max(probs, axis=1), 4)
        else:
            out_df["Priority_Confidence"] = 1.0
            
        out_df["Recommended_SLA"] = out_df["Predicted_Priority"].apply(
            lambda p: get_sla_recommendation(str(p)).get("target", "Standard SLA")
        )
    else:
        out_df["Predicted_Priority"] = "N/A"
        out_df["Priority_Confidence"] = 0.0
        out_df["Recommended_SLA"] = "Standard SLA"
        
    return out_df

