# ✈️ Airline Tweet Sentiment Analyzer

An end-to-end MLOps project that classifies airline tweets as **positive**, **neutral**, or **negative** using Machine Learning — with a full production pipeline.

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 76% |
| Dataset | US Airline Twitter Sentiment (14,872 tweets) |
| Algorithm | TF-IDF + Logistic Regression |
| Classes | Positive, Neutral, Negative |

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn |
| Web App | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Orchestration | Kubernetes |
| Registry | Docker Hub |

---

## 🏗️ Architecture

```
Code Push → GitHub → GitHub Actions → Train & Test → Build Docker Image
                                                            ↓
                                                       Docker Hub
                                                            ↓
                                                       Kubernetes
                                                            ↓
                                                    Streamlit Web App
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install scikit-learn pandas streamlit plotly

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

## ⚙️ GitHub Actions CI/CD

Automatically triggered on every push to `main`:

1. ✅ Install dependencies
2. ✅ Train the model
3. ✅ Run 8 pytest tests
4. ✅ Build Docker image
5. ✅ Push to Docker Hub

**Required GitHub Secrets:**
| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

---

## ☸️ Kubernetes Deployment

```bash
# Deploy
kubectl apply -f k8s-deployment.yaml

# Check pods (2 replicas running)
kubectl get pods

# Access the app
kubectl port-forward service/airline-sentiment-service 8501:80
```
Then open → http://localhost:8501

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```
8 tests covering data loading, label validation, model predictions and probability scores.

---

## 📁 Project Structure

```
airline-sentiment/
├── .github/workflows/
│   └── ci-cd.yml         # GitHub Actions pipeline
├── data/
│   └── Tweets.csv        # Dataset (14,872 tweets)
├── model/                # Saved model (auto-generated)
├── tests/
│   └── test_model.py     # 8 pytest tests
├── train.py              # Model training script
├── app.py                # Streamlit web app
├── Dockerfile            # Container definition
├── k8s-deployment.yaml   # Kubernetes deployment
└── requirements.txt      # Python dependencies
```

---

## 👩‍💻 Author
GitHub: [@Roopa0812](https://github.com/Roopa0812)