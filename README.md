
# 📈 Stock Price Prediction App

A machine learning-powered web application that predicts future stock prices using historical data and displays results through an interactive Streamlit UI.

## 🚀 Features

- Predict stock prices for companies like Apple (`AAPL`), Google (`GOOG`), etc.
- Interactive Streamlit interface with date selection and chart visualization
- Powered by a trained machine learning model
- Real-time data fetching using `yfinance`
- Lightweight, fast, and easy to use

---

## 🛠️ Tech Stack

- Python
- scikit-learn (ML model)
- yfinance (Stock data)
- Streamlit (UI & Deployment)
- joblib (Model saving/loading)

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run stockpredict.py
