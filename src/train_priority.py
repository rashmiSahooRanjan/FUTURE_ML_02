"""
TicketIQ - Ticket Priority Model Training & Comparison Pipeline
"""
import time
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import prepare_train_test_data
from src.feature_extraction import get_vectorizer
from src.evaluation import evaluate_predictions
from src.utils import save_json, save_pickle

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns candidate models for priority prediction.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual="auto", C=1.0, random_state=random_state), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=random_state, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=random_state)
    }

def train_and_compare_priority_models(
    df: pd.DataFrame,
    text_column: str = "Ticket Description",
    priority_column: str = "Ticket Priority",
    subject_column: Optional[str] = "Ticket Subject",
    vectorizer_type: str = "tfidf",
    max_features: int = 10000,
    test_size: float = 0.20,
    models_to_train: Optional[list] = None
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Train and compare multiple models for Ticket Priority prediction.
    Returns:
        best_pipeline, comparison_df, best_eval_dict, all_evaluations
    """
    print(f"[Priority Training] Preprocessing and splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        df=df,
        text_column=text_column,
        target_column=priority_column,
        subject_column=subject_column,
        test_size=test_size
    )
    
    candidates = get_candidate_models()
    if models_to_train:
        candidates = {k: v for k, v in candidates.items() if k in models_to_train}
        
    results_list = []
    all_evaluations = {}
    fitted_pipelines = {}
    
    for name, clf in candidates.items():
        print(f"[Priority Training] Training {name} with {vectorizer_type}...")
        vec = get_vectorizer(method=vectorizer_type, max_features=max_features)
        pipe = Pipeline([
            ("vectorizer", vec),
            ("classifier", clf)
        ])
        
        # Measure training time
        t_start = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t_start
        
        # Measure prediction time
        t_start_pred = time.time()
        y_pred = pipe.predict(X_test)
        pred_time = time.time() - t_start_pred
        
        eval_metrics = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            training_time=train_time,
            prediction_time=pred_time,
            model_name=name
        )
        
        fitted_pipelines[name] = pipe
        all_evaluations[name] = eval_metrics
        results_list.append({
            "model_name": name,
            "accuracy": eval_metrics["accuracy"],
            "precision_macro": eval_metrics["precision_macro"],
            "recall_macro": eval_metrics["recall_macro"],
            "f1_macro": eval_metrics["f1_macro"],
            "f1_weighted": eval_metrics["f1_weighted"],
            "training_time": eval_metrics["training_time"],
            "prediction_time": eval_metrics["prediction_time"]
        })
        print(f" -> {name} | Weighted F1: {eval_metrics['f1_weighted']:.4f} | Acc: {eval_metrics['accuracy']:.4f} | Time: {train_time:.2f}s")
        
    comparison_df = pd.DataFrame(results_list).sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_eval = all_evaluations[best_model_name]
    
    print(f"[Priority Training] Best priority model selected: {best_model_name} (F1: {best_eval['f1_weighted']:.4f})")
    
    # Save best model pipeline and artifacts
    os.makedirs("models", exist_ok=True)
    save_pickle(best_pipeline, "models/priority_model.pkl")
    
    # Save training metadata
    metadata = {
        "best_priority_model": best_model_name,
        "vectorizer": vectorizer_type,
        "max_features": max_features,
        "accuracy": best_eval["accuracy"],
        "f1_weighted": best_eval["f1_weighted"],
        "classes": best_eval["classes"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(metadata, "models/priority_model_metadata.json")
    
    return best_pipeline, comparison_df, best_eval, all_evaluations

