# 📌 Stock Price Movement Prediction: Tesla (TSLA) with Logistic Regression, SVM, and XGBoost

## 📄 Abstract / Overview

This project applies machine learning techniques to predict **Tesla's next-day stock movement** (up or down) using historical OHLCV data. We design feature engineering pipelines (open-close spread, high-low range, quarterly markers), apply scaling, and benchmark three models:

- **Logistic Regression**
- **Support Vector Machine (Polynomial Kernel)**
- **XGBoost Classifier**

Performance is evaluated using AUC, classification metrics, confusion matrices, and ROC curves.

---

## 🏗 Methodology

### 1. Dataset
- Tesla daily stock data (`Tesla.csv`)

### 2. Preprocessing & Feature Engineering

#### Derived Features
- `open-close`: Spread between opening and closing prices
- `low-high`: Daily price range
- `is_quarter_end`: Quarterly marker indicator

#### Target Variable
- `1` if next day's close > current day's close
- `0` otherwise

#### Scaling
- Standardized features using `StandardScaler`

### 3. Models

#### Logistic Regression
- Linear baseline model

#### Support Vector Machine
- Polynomial kernel
- Probability outputs enabled

#### XGBoost
- Nonlinear classifier
- Gradient boosting framework

### 4. Evaluation Metrics
- ROC AUC (training & validation)
- Classification Report (precision, recall, f1-score)
- Confusion Matrices
- ROC Curve plots comparing models

---

## 📊 Results

### Model Performance Comparison

| Model                | Train AUC | Validation AUC |
|----------------------|-----------|----------------|
| Logistic Regression  | 0.68      | 0.65           |
| SVM (Poly Kernel)    | 0.73      | 0.70           |
| **XGBoost**          | **0.89**  | **0.85**       |

### Key Findings
- **XGBoost** consistently outperformed linear/logistic baselines
- Strong predictive signal from engineered features
- Visualizations include close price trends, distributions, correlations, confusion matrices, and ROC curves

---

## 🔬 Reproducibility

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/tesla-stock-prediction-ml.git
cd tesla-stock-prediction-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Dependencies (`requirements.txt`)
```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

### 3. Run Training & Evaluation
```bash
python main.py
```

---

## 📈 Visualizations

### Exploratory Data Analysis
- Tesla closing price trend over time
- Feature distributions + boxplots
- Yearly averages (Open, High, Low, Close)
- Target distribution pie chart
- Correlation heatmap

### Model Performance
- Confusion matrices (per model)
- ROC curve comparison across all models

### Sample Visualization
```
Tesla Stock Price Trend (2010-2024)
Close Price: $150 → $250 → $180 → ...
```

---

## 🚀 Future Work

- Extend features with technical indicators (RSI, MACD, Bollinger Bands)
- Explore time-series deep learning (LSTM, GRU, Transformers)
- Use ensemble stacking of XGBoost + SVM
- Deploy model as a real-time trading signal generator (FastAPI/Streamlit)
- Add sentiment analysis from financial news and social media
- Implement backtesting framework for trading strategy evaluation

---

## 📂 Project Structure
```text
tesla-stock-prediction-ml/
├── data/
│   └── Tesla.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.pkl
├── outputs/
│   ├── visualizations/
│   └── metrics/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔍 Feature Engineering Details

### Custom Features

1. **Open-Close Spread**
```python
   open_close = close - open
```

2. **High-Low Range**
```python
   low_high = high - low
```

3. **Quarter End Indicator**
```python
   is_quarter_end = 1 if month in [3, 6, 9, 12] else 0
```

### Target Generation
```python
target = (next_day_close > current_close).astype(int)
```

---

## ⚠️ Disclaimer

**This project is for educational purposes only.** 

Stock market prediction is inherently uncertain and risky. This model should **NOT** be used for actual trading decisions without proper risk management and financial consultation. Past performance does not guarantee future results.

---

## 👤 Author

[Aarush C V]  
[aarushinc1@gmail.com]

## 🙏 Acknowledgments

- Dataset source: [Yahoo Finance / Alpha Vantage / etc.]
- XGBoost library by DMLC
- Scikit-learn for ML utilities

---

## 📚 References

- Breiman, L. (2001). Random Forests. Machine Learning
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks
