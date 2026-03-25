# 🎬 CinePredict — Box Office Intelligence Dashboard

A cinematic dark-theme Streamlit dashboard for exploring, analyzing, and predicting movie box office revenue using a trained neural network.

---

## 📁 Required Files

Place all these files in the **same folder** as `app.py`:

```
app.py
requirements.txt
movies_dataset.csv
box_office_model.keras
preprocessors.pkl
evaluation.png
training_curves.png
feature_importance.png
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📄 Pages

| Page | Description |
|------|-------------|
| 📊 **Overview** | KPIs, revenue by genre, box office trends, studio breakdown |
| 🔍 **Explore Data** | Interactive filters, scatter plots, box plots, correlation heatmap |
| 🤖 **Model Performance** | Evaluation plots, feature importance, training curves + interpretation |
| 🎯 **Predict Revenue** | Input a new film's details and get a revenue forecast + genre benchmarking |

---

## 🧠 Model Notes

- Architecture: Keras Neural Network
- Target: `log(worldwide_gross)`
- R² = 0.635 on test set (log scale)
- Top features: `log_budget`, `log_marketing_budget`, `is_sequel`

If TensorFlow is unavailable, the predictor falls back to a calibrated analytical formula derived from the feature importances.

---

## 🎨 Design

- Dark cinematic theme with **Syne** + **DM Sans** typography
- Orange accent (`#ff6b35`) inspired by marquee lights
- All charts built with **Plotly** for interactivity
