import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="FIN41660 Forecasting App", layout="wide")

st.title("FIN41660 Time Series Forecasting App")
st.caption("ARIMA baseline. OLS and GARCH to be added later.")

# Sidebar: file upload
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file with a date column to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head())

# Sidebar: column selection
st.sidebar.header("2. Columns")

date_col = st.sidebar.selectbox("Date column", df.columns)
value_col = st.sidebar.selectbox(
    "Price or index column to model",
    [c for c in df.columns if c != date_col]
)

# Prepare time series
data = df[[date_col, value_col]].copy()
data[date_col] = pd.to_datetime(data[date_col])
data = data.sort_values(date_col).set_index(date_col)
data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
data = data.dropna()

st.subheader("Level series")
st.line_chart(data[value_col])

# Compute log returns
returns = np.log(data[value_col]).diff().dropna()
returns.name = "log_return"

st.subheader("Log return series")
st.line_chart(returns)

# Sidebar: train-test split
st.sidebar.header("3. Train / test split")
test_size_pct = st.sidebar.slider("Test size percent", 10, 50, 20)

n = len(returns)
n_test = max(1, int(n * test_size_pct / 100))
n_train = n - n_test

train = returns.iloc[:n_train]
test = returns.iloc[n_train:]

st.write(f"Sample size: {n}, train: {n_train}, test: {n_test}")

# Sidebar: ARIMA settings
st.sidebar.header("4. ARIMA settings")
p = st.sidebar.number_input("AR order p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Difference order d", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("MA order q", min_value=0, max_value=5, value=0, step=1)

if st.sidebar.button("Run ARIMA"):
    with st.spinner("Fitting ARIMA model..."):
        model = ARIMA(train, order=(p, d, q)).fit()

        # Forecast over the test period
        forecast = model.forecast(steps=len(test))
        forecast.index = test.index

        # Metrics
        rmse = mean_squared_error(test, forecast, squared=False)
        mae = mean_absolute_error(test, forecast)

    st.subheader("ARIMA model summary")
    st.text(model.summary())

    st.subheader("Forecast vs actual (test sample)")
    chart_df = pd.concat(
        {"actual": test, "forecast": forecast},
        axis=1
    )
    st.line_chart(chart_df)

    st.subheader("Forecast accuracy on test sample")
    st.write(
        pd.DataFrame(
            {"RMSE": [rmse], "MAE": [mae]},
            index=["ARIMA"]
        )
    )
else:
    st.info("Set ARIMA parameters in the sidebar and click 'Run ARIMA'.")
