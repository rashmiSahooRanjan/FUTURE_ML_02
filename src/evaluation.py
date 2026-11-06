"""
TicketIQ - Model Evaluation and Metrics Visualization Module
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_predictions(
    y_true,
    y_pred,
    training_time: float = 0.0,
    prediction_time: float = 0.0,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for classification results.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    classes = sorted(list(set(y_true).union(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time": round(training_time, 4),
        "prediction_time": round(prediction_time, 4),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text
    }

def create_confusion_matrix_fig(cm: List[List[int]], classes: List[str], title: str = "Confusion Matrix"):
    """
    Generate an interactive Plotly heatmap for confusion matrix.
    """
    cm_arr = np.array(cm)
    # Calculate row-normalized percentage
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm_arr / row_sums) * 100
    
    # Custom hover annotations
    text_annotations = []
    for i in range(len(classes)):
        row_annot = []
        for j in range(len(classes)):
            count = cm_arr[i][j]
            pct = cm_norm[i][j]
            row_annot.append(f"<b>{count}</b><br>({pct:.1f}%)")
        text_annotations.append(row_annot)
        
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale=[[0, "#0B1120"], [0.5, "#1E40AF"], [1.0, "#3B82F6"]],
        colorbar=dict(title="Ticket Count", thickness=15),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Predicted Class", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B"),
        yaxis=dict(title="Actual Class", tickfont=dict(color="#CBD5E1"), autorange="reversed", gridcolor="#1E293B"),
        paper_bgcolor="#151F32",
        plot_bgcolor="#0B1120",
        margin=dict(l=60, r=40, t=50, b=60),
        font=dict(family="Inter, sans-serif", color="#F8FAFC")
    )
    return fig

def create_model_comparison_bar_chart(results_df: pd.DataFrame, metric: str = "f1_weighted"):
    """
    Bar chart comparing models on a specific metric.
    """
    metric_labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (Macro)",
        "recall_macro": "Recall (Macro)",
        "f1_macro": "F1-Score (Macro)",
        "f1_weighted": "F1-Score (Weighted)",
        "training_time": "Training Time (s)"
    }
    y_col = metric if metric in results_df.columns else "f1_weighted"
    
    fig = px.bar(
        results_df,
        x="model_name",
        y=y_col,
        text=results_df[y_col].apply(lambda x: f"{x:.3f}" if y_col != "training_time" else f"{x:.2f}s"),
        color=y_col,
        color_continuous_scale="Viridis",
        title=f"Model Comparison - {metric_labels.get(y_col, y_col)}"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="#151F32",
        plot_bgcolor="#0B1120",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        xaxis=dict(title="Algorithm", tickfont=dict(color="#CBD5E1")),
        yaxis=dict(title=metric_labels.get(y_col, y_col), tickfont=dict(color="#CBD5E1")),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

