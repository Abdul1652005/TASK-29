import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "ticket_model.joblib"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "ticket_vectorizer.joblib"))


def predict_category(text: str):
    vec = vectorizer.transform([text.strip().lower()])
    prediction = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    confidence = float(max(score) - min(score))  # relative margin, LinearSVC has no predict_proba
    return prediction, confidence
