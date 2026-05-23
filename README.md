# ✈️ Airline Tweet Sentiment Analyzer

An end-to-end ML system that classifies airline tweets as **positive**, **neutral**, or **negative** — with a full MLOps pipeline.

## Stack
- **ML**: Scikit-learn (TF-IDF + Logistic Regression)
- **App**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Orchestration**: Kubernetes

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Run the Streamlit app
streamlit run app.py
```

App opens at → http://localhost:8501

---

## 🐳 Docker

```bash
# Build
docker build -t airline-sentiment .

# Run
docker run -p 8501:8501 airline-sentiment
```

---

## ⚙️ GitHub Actions Setup

Add these secrets to your GitHub repo (Settings → Secrets → Actions):

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

**Pipeline flow on push to `main`:**
1. Train model
2. Run pytest
3. Build Docker image
4. Push to Docker Hub

---

## ☸️ Kubernetes Deployment

```bash
# 1. Edit k8s-deployment.yaml — replace YOUR_DOCKERHUB_USERNAME

# 2. Apply
kubectl apply -f k8s-deployment.yaml

# 3. Check status
kubectl get pods
kubectl get service airline-sentiment-service
```

---

## 📁 Project Structure

```
airline-sentiment/
├── data/               # Dataset
│   └── Tweets.csv
├── model/              # Saved model (generated after training)
├── tests/              # Pytest tests
│   └── test_model.py
├── .github/workflows/  # GitHub Actions
│   └── ci-cd.yml
├── train.py            # Model training
├── app.py              # Streamlit app
├── Dockerfile
├── requirements.txt
└── k8s-deployment.yaml
```
