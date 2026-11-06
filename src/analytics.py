"""
TicketIQ - Exploratory Data Analysis & Business Intelligence Analytics
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import Dict, Any, Optional
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

from src.preprocessing import clean_text

THEME_BG = "#0B1120"
THEME_CARD = "#151F32"
THEME_FONT = dict(family="Inter, sans-serif", color="#F8FAFC")

def get_dataset_summary(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
    """Calculate core high-level KPI metrics from dataset."""
    desc_col = mapping.get("description")
    cat_col = mapping.get("category")
    prio_col = mapping.get("priority")
    date_col = mapping.get("date")
    
    total_tickets = len(df)
    unique_tickets = df[desc_col].nunique() if desc_col in df.columns else total_tickets
    total_cats = df[cat_col].nunique() if cat_col in df.columns else 0
    total_prios = df[prio_col].nunique() if prio_col in df.columns else 0
    
    prio_counts = df[prio_col].value_counts().to_dict() if prio_col in df.columns else {}
    
    return {
        "total_tickets": total_tickets,
        "unique_tickets": unique_tickets,
        "total_categories": total_cats,
        "total_priorities": total_prios,
        "critical_tickets": prio_counts.get("Critical", 0),
        "high_tickets": prio_counts.get("High", 0),
        "medium_tickets": prio_counts.get("Medium", 0),
        "low_tickets": prio_counts.get("Low", 0),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(total_tickets - unique_tickets)
    }

def plot_category_distribution(df: pd.DataFrame, cat_col: str):
    """Interactive bar chart of category distribution."""
    counts = df[cat_col].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    
    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        text="Count",
        color="Count",
        color_continuous_scale="Blues",
        title="Customer Support Tickets by Category"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor=THEME_CARD,
        plot_bgcolor=THEME_BG,
        font=THEME_FONT,
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

def plot_priority_distribution(df: pd.DataFrame, prio_col: str):
    """Interactive donut chart for priority breakdown."""
    counts = df[prio_col].value_counts().reset_index()
    counts.columns = ["Priority", "Count"]
    
    color_map = {
        "Critical": "#EF4444",
        "High": "#F97316",
        "Medium": "#F59E0B",
        "Low": "#10B981"
    }
    
    fig = px.pie(
        counts,
        names="Priority",
        values="Count",
        hole=0.45,
        color="Priority",
        color_discrete_map=color_map,
        title="Ticket Priority Distribution"
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor=THEME_CARD,
        plot_bgcolor=THEME_BG,
        font=THEME_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

def plot_category_priority_heatmap(df: pd.DataFrame, cat_col: str, prio_col: str):
    """Cross-tabulation heatmap of Category vs Priority."""
    cross_tab = pd.crosstab(df[cat_col], df[prio_col])
    
    fig = px.imshow(
        cross_tab,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Cross-Tabulation: Ticket Category vs Priority"
    )
    fig.update_layout(
        paper_bgcolor=THEME_CARD,
        plot_bgcolor=THEME_BG,
        font=THEME_FONT,
        xaxis=dict(title="Priority Level", gridcolor="#1E293B"),
        yaxis=dict(title="Ticket Category", gridcolor="#1E293B"),
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

def plot_ticket_length_distribution(df: pd.DataFrame, desc_col: str):
    """Histogram of ticket character length."""
    lengths = df[desc_col].fillna("").astype(str).str.len()
    fig = px.histogram(
        lengths,
        nbins=40,
        title="Ticket Description Length (Characters)",
        labels={"value": "Character Length", "count": "Frequency"},
        color_discrete_sequence=["#3B82F6"]
    )
    fig.update_layout(
        paper_bgcolor=THEME_CARD,
        plot_bgcolor=THEME_BG,
        font=THEME_FONT,
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

def plot_top_words(df: pd.DataFrame, desc_col: str, top_n: int = 15):
    """Bar chart of top 15 most frequent cleaned words."""
    all_words = []
    # Sample if large for high speed
    sample_texts = df[desc_col].dropna().sample(min(len(df), 2000), random_state=42)
    for text in sample_texts:
        cleaned = clean_text(text)
        all_words.extend(cleaned.split())
        
    counts = Counter(all_words).most_common(top_n)
    if not counts:
        return None
        
    words_df = pd.DataFrame(counts, columns=["Word", "Frequency"])
    fig = px.bar(
        words_df,
        x="Frequency",
        y="Word",
        orientation="h",
        text="Frequency",
        title=f"Top {top_n} Most Frequent Ticket Keywords",
        color="Frequency",
        color_continuous_scale="Purples"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor=THEME_CARD,
        plot_bgcolor=THEME_BG,
        font=THEME_FONT,
        yaxis=dict(autorange="reversed", gridcolor="#1E293B"),
        xaxis=dict(gridcolor="#1E293B"),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig

def generate_wordcloud_image(df: pd.DataFrame, desc_col: str) -> Optional[str]:
    """Generate WordCloud and return as base64 PNG data URI string."""
    try:
        sample_texts = df[desc_col].dropna().sample(min(len(df), 1500), random_state=42)
        full_text = " ".join([clean_text(t) for t in sample_texts])
        
        wc = WordCloud(
            width=800,
            height=400,
            background_color="#0B1120",
            colormap="Blues",
            max_words=120,
            random_state=42
        ).generate(full_text)
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0B1120")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0B1120")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return None

def plot_timeline_trends(df: pd.DataFrame, date_col: str):
    """Monthly ticket volume trends over time."""
    try:
        dt_df = df.copy()
        dt_df["Parsed_Date"] = pd.to_datetime(dt_df[date_col], errors="coerce")
        dt_df = dt_df.dropna(subset=["Parsed_Date"])
        monthly = dt_df.set_index("Parsed_Date").resample("ME").size().reset_index()
        monthly.columns = ["Month", "Ticket Volume"]
        
        fig = px.line(
            monthly,
            x="Month",
            y="Ticket Volume",
            markers=True,
            title="Ticket Volume Trends Over Time",
            line_shape="spline"
        )
        fig.update_traces(line_color="#3B82F6", marker=dict(size=7, color="#60A5FA"))
        fig.update_layout(
            paper_bgcolor=THEME_CARD,
            plot_bgcolor=THEME_BG,
            font=THEME_FONT,
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=40, r=40, t=50, b=50)
        )
        return fig
    except Exception:
        return None

