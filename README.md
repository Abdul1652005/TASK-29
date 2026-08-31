# 🎫 Customer Support Ticket Classifier

**Task 29 — Final Capstone Project**
PKCERT AI & Software Development Internship

An AI-powered tool that automatically reads a customer support ticket and predicts which category it belongs to (e.g. Hardware, Access, HR Support), helping route tickets faster and reduce manual triage.

---

## 🔗 Live Links

| Resource | Link |
|---|---|
| **Live Frontend (Streamlit)** | https://project-findit-qdnpmcct2jftkub8sy2ygt.streamlit.app/ |
| **Live Backend API (Railway)** | https://task-29-production.up.railway.app |
| **API Docs (Swagger)** | https://task-29-production.up.railway.app/docs |
| **GitHub Repository** | https://github.com/Abdul1652005/TASK-29 |

---

## 📌 Problem Statement

Support teams receive many customer tickets every day and need to sort them into categories before assigning them to the right department. This is normally done manually, which is slow and error-prone. This tool automates that first step.

**Target user:** Support teams handling a high volume of incoming tickets.

**Core functionality:**
- Enter a ticket description
- Get a predicted category with a confidence score
- View a session history of past classifications
- Export history as CSV

---

## 🧠 Model

| Stage | Detail |
|---|---|
| Features | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Baseline model | Logistic Regression — 84.9% F1 |
| Final model | Linear SVM (LinearSVC) — 85.5% F1 |
| Dataset | 47,837 labeled support tickets, 8 categories |
| Split | 70% train / 15% validation / 15% test (stratified) |

Categories: `Access`, `Administrative rights`, `HR Support`, `Hardware`, `Internal Project`, `Miscellaneous`, `Purchase`, `Storage`

A classical ML approach (TF-IDF + Linear SVM) was chosen over a Transformer model to keep training and inference lightweight enough to run on a standard laptop with no GPU, while still achieving strong accuracy.

---

## 🏗️ Architecture

```
Jupyter Notebook  →  FastAPI Backend  →  Streamlit Frontend
(train + evaluate)   (POST /predict)     (UI, calls the API)
```

- **Jupyter Notebook** — data cleaning, TF-IDF + model training, evaluation, saves `ticket_model.joblib` and `ticket_vectorizer.joblib`
- **FastAPI Backend** — loads the saved model, exposes `POST /predict`, validates input, returns category + confidence + full score distribution
- **Streamlit Frontend** — text input UI, calls the backend API, displays the result with a confidence chart, keeps session history

### API Contract

**POST** `/predict`

Request:
```json
{ "text": "My laptop screen is black and won't turn on" }
```

Response:
```json
{
  "category": "Hardware",
  "confidence": 66.51,
  "distribution": { "Hardware": 66.51, "Access": 4.46, "...": "..." }
}
```

---

## 📁 Project Structure

```
ticket-classifier/
├── notebook/
│   └── ticket_classifier.ipynb      # data cleaning, training, evaluation
├── backend/
│   ├── main.py                      # FastAPI app (/predict endpoint)
│   ├── model.py                     # loads model + vectorizer, prediction logic
│   ├── ticket_model.joblib
│   ├── ticket_vectorizer.joblib
│   └── requirements.txt
├── frontend/
│   ├── app.py                       # Streamlit UI
│   ├── ticket_model.joblib
│   ├── ticket_vectorizer.joblib
│   └── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run Locally

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` to test the API interactively.

### Frontend (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Visit `http://localhost:8501`.

---

## ☁️ Deployment

- **Backend** — deployed on [Railway](https://railway.com), connected to this GitHub repo (Root Directory: `backend`, Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`). Free tier, no credit card.
- **Frontend** — deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected to this GitHub repo (`frontend/app.py`). Free tier, no credit card.

**Note:** An early deployment issue occurred when Streamlit Cloud defaulted to Python 3.14, which lacked prebuilt wheels for some dependencies. Fixed by explicitly setting the Python version to **3.11** in the app's Advanced Settings.

---

## 🧪 Testing

| Test Case | Input | Result |
|---|---|---|
| Valid ticket text | `"My laptop is not turning on"` | ✅ Passed — returned `Hardware` |
| Empty text | `""` | ✅ Passed — 422 validation error |
| Missing field | `{}` | ✅ Passed — 422 validation error |
| API health check | `GET /` | ✅ Passed — `{"status":"ok"}` |
| Live frontend → API call | Any valid ticket | ✅ Passed — verified live |

---

## ⚠️ Challenges Faced

- **Hardware limits** — chose a lightweight TF-IDF + classical ML pipeline instead of a Transformer to run comfortably on a standard laptop (Core i5, 16GB RAM, no GPU)
- **Free hosting constraints** — Railway's memory limits ruled out heavier models; Streamlit Cloud needed a Python version fix to build successfully
- **Dataset selection** — an initial dataset gave near-random results and was replaced with a better-labeled one
- **Frontend–backend integration** — connected Streamlit to the live FastAPI service with safe fallback handling in case the API is temporarily unreachable

---

## 👤 Author

**Abdul Rehman**  
PKCERT AI & Software Development Internship — Task 29  
GitHub: [github.com/Abdul1652005](https://github.com/Abdul1652005)
