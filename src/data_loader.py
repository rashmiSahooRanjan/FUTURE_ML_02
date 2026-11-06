"""
TicketIQ - Dataset Loading, Column Detection, and Validation Module
"""
import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text, preprocess_series

# Keyword lists for automatic column mapping
COLUMN_PATTERNS = {
    "description": [
        "ticket description", "description", "issue description", "ticket_description",
        "issue_description", "body", "ticket_body", "message", "details", "content", "text"
    ],
    "subject": [
        "ticket subject", "subject", "title", "ticket_subject", "issue_subject",
        "summary", "headline", "topic"
    ],
    "category": [
        "ticket type", "ticket_type", "category", "issue_type", "type",
        "ticket_category", "department", "classification", "issue_category"
    ],
    "priority": [
        "ticket priority", "ticket_priority", "priority", "urgency", "severity",
        "priority_level", "severity_level", "sla_level"
    ],
    "id": [
        "ticket id", "ticket_id", "id", "ticket#", "incident_id", "case_id"
    ],
    "status": [
        "ticket status", "ticket_status", "status", "state", "ticket_state"
    ],
    "date": [
        "date of purchase", "date", "created_at", "created_date", "timestamp",
        "ticket_date", "purchase_date"
    ],
    "customer": [
        "customer name", "customer_name", "customer id", "customer_id",
        "client", "user", "customer email", "email"
    ]
}

# Rule-based priority generator keywords for datasets lacking priority
RULE_PRIORITY_KEYWORDS = {
    "Critical": [
        "outage", "emergency", "breach", "security breach", "compromised", "hacked",
        "complete failure", "system crash", "data loss", "payment gateway down",
        "cant process payment", "all users down", "severe outage"
    ],
    "High": [
        "urgent", "immediately", "critical", "blocked", "blocking", "failing",
        "cant login", "cannot login", "unable to login", "password reset failed",
        "charged twice", "wrong amount charged", "unauthorized charge", "broken feature"
    ],
    "Medium": [
        "error", "issue", "bug", "intermittent", "delayed", "slow", "glitch",
        "setup problem", "configuration", "troubleshoot", "workaround", "not working"
    ],
    "Low": [
        "how to", "inquiry", "question", "feature request", "feedback",
        "information", "guide", "documentation", "suggestion", "upgrade question"
    ]
}

def detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently map dataframe column names to canonical schema keys.
    """
    col_map = {
        "description": None,
        "subject": None,
        "category": None,
        "priority": None,
        "id": None,
        "status": None,
        "date": None,
        "customer": None
    }
    
    clean_cols = {c: re.sub(r"[_\s\-]+", " ", str(c).strip().lower()) for c in columns}
    
    for key, patterns in COLUMN_PATTERNS.items():
        # First try exact match
        for orig_col, clean_col in clean_cols.items():
            if clean_col in patterns:
                col_map[key] = orig_col
                break
        
        # Second try substring match if not found
        if col_map[key] is None:
            for orig_col, clean_col in clean_cols.items():
                for pat in patterns:
                    if pat in clean_col and not any(col_map[k] == orig_col for k in col_map if col_map[k] is not None):
                        col_map[key] = orig_col
                        break
                if col_map[key] is not None:
                    break
                    
    return col_map

def generate_rule_based_priority(text: str) -> str:
    """
    Fallback rule-based priority generator for datasets lacking priority column.
    Produces labels: Critical, High, Medium, Low.
    """
    lower_text = str(text).lower()
    
    for priority, keywords in RULE_PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower_text:
                return priority
                
    # Default fallback based on length or medium
    return "Medium"

def load_and_validate_dataset(
    filepath_or_buffer,
    manual_mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int]]:
    """
    Load dataset from CSV or Excel, validate structure, detect columns,
    and return cleaned dataframe, column mapping, and data health stats.
    """
    if isinstance(filepath_or_buffer, str):
        if filepath_or_buffer.endswith(".xlsx") or filepath_or_buffer.endswith(".xls"):
            df = pd.read_excel(filepath_or_buffer)
        else:
            df = pd.read_csv(filepath_or_buffer)
    else:
        # Buffer (e.g. from Streamlit file_uploader)
        try:
            df = pd.read_csv(filepath_or_buffer)
        except Exception:
            df = pd.read_excel(filepath_or_buffer)
            
    total_raw_rows = len(df)
    
    # Column detection
    mapping = detect_columns(df.columns.tolist())
    if manual_mapping:
        mapping.update({k: v for k, v in manual_mapping.items() if v})
        
    stats = {
        "raw_rows": total_raw_rows,
        "missing_descriptions": 0,
        "duplicates_removed": 0,
        "final_rows": 0,
        "is_priority_rule_generated": False
    }
    
    desc_col = mapping.get("description")
    if desc_col is None or desc_col not in df.columns:
        # Fallback to first text column
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            desc_col = text_cols[0]
            mapping["description"] = desc_col
        else:
            raise ValueError("Could not identify a ticket description column in the dataset.")
            
    # Clean missing descriptions
    initial_desc_missing = df[desc_col].isna().sum()
    stats["missing_descriptions"] = int(initial_desc_missing)
    df = df.dropna(subset=[desc_col]).copy()
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=[desc_col]).copy()
    stats["duplicates_removed"] = int(initial_len - len(df))
    
    # Handle missing priority if not present
    prio_col = mapping.get("priority")
    if prio_col is None or prio_col not in df.columns:
        df["Generated_Ticket_Priority"] = df[desc_col].apply(generate_rule_based_priority)
        mapping["priority"] = "Generated_Ticket_Priority"
        stats["is_priority_rule_generated"] = True
        
    # Ensure category column
    cat_col = mapping.get("category")
    if cat_col is None or cat_col not in df.columns:
        raise ValueError("Could not identify a ticket category / type column in the dataset.")
        
    df = df.dropna(subset=[cat_col]).copy()
    
    stats["final_rows"] = len(df)
    return df, mapping, stats

def prepare_train_test_data(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    subject_column: Optional[str] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare X and y with text preprocessing and stratified train/test split.
    Combines subject + description if subject is available for richer signal.
    """
    # Create combined input text
    if subject_column and subject_column in df.columns:
        combined_text = df[subject_column].fillna("").astype(str) + " - " + df[text_column].fillna("").astype(str)
    else:
        combined_text = df[text_column].fillna("").astype(str)
        
    X_cleaned = preprocess_series(combined_text)
    y = df[target_column].astype(str)
    
    # Filter empty cleaned texts
    valid_mask = X_cleaned.str.strip().str.len() > 0
    X_cleaned = X_cleaned[valid_mask]
    y = y[valid_mask]
    
    # Stratified split if minimum class count allows
    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test

