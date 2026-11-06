"""
TicketIQ - Feature Extraction Module (Bag of Words & TF-IDF)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectorizer(
    method: str = "tfidf",
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
):
    """
    Factory function to instantiate CountVectorizer (BoW) or TfidfVectorizer.
    Returns sparse matrix representations.
    """
    method = method.lower().strip()
    if method in ["bow", "bag_of_words", "count"]:
        return CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
    else:
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf
        )

def get_top_features_by_class(
    vectorizer,
    model,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10
) -> dict:
    """
    Extract top influential terms per class from linear models or tree models.
    """
    if feature_names is None:
        try:
            feature_names = vectorizer.get_feature_names_out()
        except Exception:
            return {}
            
    top_terms = {}
    
    # Handle Logistic Regression, CalibratedClassifierCV, or LinearSVC
    clf = model
    if hasattr(clf, "calibrated_classifiers_"):
        # For CalibratedClassifierCV
        clf = clf.calibrated_classifiers_[0].estimator
        
    if hasattr(clf, "coef_"):
        classes = getattr(model, "classes_", [f"Class {i}" for i in range(len(clf.coef_))])
        for idx, cls_name in enumerate(classes):
            if idx < len(clf.coef_):
                top_indices = np.argsort(clf.coef_[idx])[-top_n:][::-1]
                top_terms[str(cls_name)] = [
                    (feature_names[i], round(float(clf.coef_[idx][i]), 4))
                    for i in top_indices if i < len(feature_names)
                ]
    elif hasattr(clf, "feature_importances_"):
        top_indices = np.argsort(clf.feature_importances_)[-top_n:][::-1]
        top_terms["Overall Top Features"] = [
            (feature_names[i], round(float(clf.feature_importances_[i]), 4))
            for i in top_indices if i < len(feature_names)
        ]
        
    return top_terms

