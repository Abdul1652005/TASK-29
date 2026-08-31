import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ticket_model.joblib")
if not os.path.exists(model_path):
    model_path = os.path.join(BASE_DIR, "ticket-classifier", "backend", "ticket_model.joblib")

vec_path = os.path.join(BASE_DIR, "ticket_vectorizer.joblib")
if not os.path.exists(vec_path):
    vec_path = os.path.join(BASE_DIR, "ticket-classifier", "backend", "ticket_vectorizer.joblib")

model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)
CATEGORIES = list(model.classes_)


def predict_category(text: str):
    vec = vectorizer.transform([text.strip().lower()])
    prediction = model.predict(vec)[0]
    scores = model.decision_function(vec)[0]

    # softmax-style relative confidence across all categories
    exp_s = np.exp(scores - np.max(scores))
    probs = exp_s / exp_s.sum()
    confidence_pct = float(probs.max()) * 100

    distribution = {cat: round(float(p) * 100, 2) for cat, p in zip(CATEGORIES, probs)}

    return prediction, confidence_pct, distribution
