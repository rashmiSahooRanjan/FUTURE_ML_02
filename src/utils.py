"""
TicketIQ - Utility functions and helpers
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Color Palette for SaaS Dark Theme
THEME = {
    "bg_dark": "#0B1120",
    "card_dark": "#151F32",
    "card_border": "#1E293B",
    "primary": "#3B82F6",
    "primary_light": "#60A5FA",
    "accent_purple": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8"
}

# Priority Color Badges
PRIORITY_COLORS = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#10B981"
}

def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to JSON file with indentation."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

def load_json(filepath: str) -> dict:
    """Load JSON file if exists, else return empty dict."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_pickle(obj, filepath: str) -> None:
    """Save Python object with joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)

def load_pickle(filepath: str):
    """Load Python object with joblib."""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None

def format_percentage(val: float) -> str:
    """Format float as percentage string."""
    return f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"

def get_priority_badge(priority: str) -> str:
    """Return HTML badge for priority level."""
    color = PRIORITY_COLORS.get(priority, "#64748B")
    return f"""<span style="background-color: {color}22; color: {color}; border: 1px solid {color}55; padding: 3px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;">{priority}</span>"""

def get_sla_recommendation(priority: str) -> dict:
    """Get recommended SLA and team assignment based on predicted priority."""
    p = str(priority).capitalize()
    if p == "Critical":
        return {
            "sla_hours": 2,
            "target": "Urgent Response (< 2 Hours)",
            "tier": "Tier 3 / Incident Response Team",
            "action": "Immediate paging of on-call engineering lead. Trigger Incident channel."
        }
    elif p == "High":
        return {
            "sla_hours": 6,
            "target": "Expedited Response (< 6 Hours)",
            "tier": "Tier 2 Senior Technical Support",
            "action": "Assign to active queue with elevated priority tag."
        }
    elif p == "Medium":
        return {
            "sla_hours": 24,
            "target": "Standard Business SLA (< 24 Hours)",
            "tier": "Tier 1 Customer Support Specialist",
            "action": "Route to standard ticket resolution workflow."
        }
    else:
        return {
            "sla_hours": 48,
            "target": "Routine SLA (< 48 Hours)",
            "tier": "Customer Operations / Helpdesk",
            "action": "Provide self-service guide or schedule regular follow-up."
        }

