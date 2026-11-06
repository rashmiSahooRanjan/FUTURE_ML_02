"""
TicketIQ - NLP Text Preprocessing Pipeline
"""
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Download stopwords and wordnet if not available
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

# Crucial domain and urgency words that must NEVER be removed as stopwords
URGENCY_AND_DOMAIN_PRESERVE = {
    "not", "no", "nor", "cannot", "cant", "wont", "dont", "never",
    "urgent", "emergency", "critical", "immediately", "immediate", "asap",
    "failed", "failing", "failure", "blocked", "blocking", "unable",
    "outage", "crash", "crashed", "down", "error", "errors", "broken",
    "breach", "security", "lost", "freeze", "freezing", "halt",
    "unresponsive", "risk", "refund", "cancel", "cancellation",
    "billing", "charge", "charged", "overcharge", "fraud", "scam",
    "hardware", "software", "bug", "payment", "setup", "network", "offline"
}

# Adjusted stopword set (excluding our preserved words)
FILTERED_STOPWORDS = STOP_WORDS - URGENCY_AND_DOMAIN_PRESERVE

# Regex pre-compiled patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PLACEHOLDER_PATTERN = re.compile(r"\{[a-zA-Z0-9_]+\}")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Standardize, clean, tokenize, remove stopwords, and lemmatize ticket text.
    Preserves meaningful urgency and domain keywords.
    """
    if text is None or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = HTML_PATTERN.sub(" ", text)
    
    # Remove URLs
    text = URL_PATTERN.sub(" ", text)
    
    # Remove Email addresses
    text = EMAIL_PATTERN.sub(" ", text)
    
    # Remove dataset placeholder tags e.g. {product_purchased}
    text = PLACEHOLDER_PATTERN.sub(" ", text)
    
    # Replace punctuation and special non-alpha chars with space
    text = NON_ALPHA_PATTERN.sub(" ", text)
    
    # Tokenize (fast whitespace or word tokenizer)
    tokens = text.split()
    
    # Filter stopwords and lemmatize
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if len(token) > 1:
            if token not in FILTERED_STOPWORDS or token in URGENCY_AND_DOMAIN_PRESERVE:
                try:
                    lemmatized = LEMMATIZER.lemmatize(token)
                except Exception:
                    lemmatized = token
                cleaned_tokens.append(lemmatized)
                
    result = " ".join(cleaned_tokens)
    return WHITESPACE_PATTERN.sub(" ", result).strip()

def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply clean_text to a pandas Series of text."""
    return series.fillna("").astype(str).apply(clean_text)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for seamless integration
    into sklearn Pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return preprocess_series(X).values
        elif isinstance(X, (list, tuple)):
            return [clean_text(t) for t in X]
        elif isinstance(X, np.ndarray):
            return np.array([clean_text(t) for t in X.flatten()])
        else:
            return np.array([clean_text(str(X))])

