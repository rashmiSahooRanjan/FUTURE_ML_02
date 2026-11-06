"""
Script to train, evaluate, and benchmark all ML models on customer_support_tickets.csv
and generate production model artifacts and evaluation reports.
"""
import os
import json
import pandas as pd
import numpy as np

from src.data_loader import load_and_validate_dataset
from src.train_category import train_and_compare_category_models
from src.train_priority import train_and_compare_priority_models
from src.utils import save_json

def main():
    dataset_path = "data/raw/customer_support_tickets.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "Dataset/customer_support_tickets.csv"
        
    print(f"Loading and validating dataset from: {dataset_path}")
    df, mapping, stats = load_and_validate_dataset(dataset_path)
    print("Detected column mapping:", mapping)
    print("Data health stats:", stats)
    
    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_tickets.csv", index=False)
    print("Saved processed dataset to data/processed/cleaned_tickets.csv")
    
    # Save a small sample dataset (50 rows) for quick bulk classification testing
    os.makedirs("data/sample", exist_ok=True)
    sample_df = df.sample(min(50, len(df)), random_state=42)
    sample_df.to_csv("data/sample/sample_tickets.csv", index=False)
    print("Saved sample dataset to data/sample/sample_tickets.csv")
    
    # Train Category Classification Models
    print("\n" + "="*50)
    print("TRAINING TICKET CATEGORY CLASSIFICATION MODELS")
    print("="*50)
    cat_pipe, cat_comparison, cat_best_eval, cat_all_evals = train_and_compare_category_models(
        df=df,
        text_column=mapping["description"],
        category_column=mapping["category"],
        subject_column=mapping.get("subject"),
        vectorizer_type="tfidf",
        max_features=10000,
        test_size=0.20
    )
    
    # Train Priority Classification Models
    print("\n" + "="*50)
    print("TRAINING TICKET PRIORITY PREDICTION MODELS")
    print("="*50)
    prio_pipe, prio_comparison, prio_best_eval, prio_all_evals = train_and_compare_priority_models(
        df=df,
        text_column=mapping["description"],
        priority_column=mapping["priority"],
        subject_column=mapping.get("subject"),
        vectorizer_type="tfidf",
        max_features=10000,
        test_size=0.20
    )
    
    # Combine Model Comparisons
    os.makedirs("reports", exist_ok=True)
    cat_comp_copy = cat_comparison.copy()
    cat_comp_copy["task"] = "Category Classification"
    
    prio_comp_copy = prio_comparison.copy()
    prio_comp_copy["task"] = "Priority Prediction"
    
    combined_comparison = pd.concat([cat_comp_copy, prio_comp_copy], ignore_index=True)
    combined_comparison.to_csv("reports/model_comparison.csv", index=False)
    print("\nSaved reports/model_comparison.csv")
    
    # Save Text Classification Reports
    report_text = f"""======================================================================
TICKETIQ - MACHINE LEARNING MODEL EVALUATION REPORT
======================================================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Customer Support Ticket Dataset
Total Samples: {len(df)}

----------------------------------------------------------------------
1. BEST CATEGORY MODEL: {cat_best_eval['model_name']}
----------------------------------------------------------------------
Accuracy:           {cat_best_eval['accuracy']:.4f}
Weighted F1-Score:  {cat_best_eval['f1_weighted']:.4f}
Macro F1-Score:     {cat_best_eval['f1_macro']:.4f}
Macro Precision:    {cat_best_eval['precision_macro']:.4f}
Macro Recall:       {cat_best_eval['recall_macro']:.4f}
Training Time:      {cat_best_eval['training_time']:.2f}s
Prediction Time:    {cat_best_eval['prediction_time']:.2f}s

Category Classification Report:
{cat_best_eval['classification_report_text']}

----------------------------------------------------------------------
2. BEST PRIORITY MODEL: {prio_best_eval['model_name']}
----------------------------------------------------------------------
Accuracy:           {prio_best_eval['accuracy']:.4f}
Weighted F1-Score:  {prio_best_eval['f1_weighted']:.4f}
Macro F1-Score:     {prio_best_eval['f1_macro']:.4f}
Macro Precision:    {prio_best_eval['precision_macro']:.4f}
Macro Recall:       {prio_best_eval['recall_macro']:.4f}
Training Time:      {prio_best_eval['training_time']:.2f}s
Prediction Time:    {prio_best_eval['prediction_time']:.2f}s

Priority Classification Report:
{prio_best_eval['classification_report_text']}

======================================================================
"""
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved reports/classification_report.txt")
    
    # Save Comprehensive Evaluation Summary JSON
    summary_json = {
        "dataset_summary": {
            "total_samples": len(df),
            "columns": list(df.columns),
            "mapping": mapping,
            "stats": stats
        },
        "category_model": {
            "best_model": cat_best_eval["model_name"],
            "accuracy": cat_best_eval["accuracy"],
            "f1_weighted": cat_best_eval["f1_weighted"],
            "all_models": cat_all_evals
        },
        "priority_model": {
            "best_model": prio_best_eval["model_name"],
            "accuracy": prio_best_eval["accuracy"],
            "f1_weighted": prio_best_eval["f1_weighted"],
            "all_models": prio_all_evals
        }
    }
    save_json(summary_json, "reports/evaluation_summary.json")
    print("Saved reports/evaluation_summary.json")
    
    # Master metadata
    master_metadata = {
        "best_category_model": cat_best_eval["model_name"],
        "best_priority_model": prio_best_eval["model_name"],
        "category_accuracy": cat_best_eval["accuracy"],
        "category_f1": cat_best_eval["f1_weighted"],
        "priority_accuracy": prio_best_eval["accuracy"],
        "priority_f1": prio_best_eval["f1_weighted"],
        "category_classes": cat_best_eval["classes"],
        "priority_classes": prio_best_eval["classes"],
        "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(master_metadata, "models/model_metadata.json")
    print("Saved models/model_metadata.json")
    
    print("\n ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

