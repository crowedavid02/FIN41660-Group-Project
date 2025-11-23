import streamlit as st
import pandas as pd
import numpy as np

from models import prepare_series, fit_arima, fit_ols_ar, fit_garch

st.set_page_config(page_title="FIN41660 Forecasting App", layout="wide")

st.title("FIN41660 Time Series Forecasting App")
st.caption("ARIMA, OLS AR(p), and GARCH(1,1) on returns with train and test split.")


# 1. Upload CSV
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file with a date column and a price column to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head())

# 2. Choose columns
st.sidebar.header("2. Columns")
date_col = st.sidebar.selectbox("Date column", df.columns)
value_col = st.sidebar.selectbox(
    "Price column",
    [c for c in df.columns if c != date_col],
)

# 3. Prepare series (price and returns)
try:
    price, returns = prepare_series(df, date_col, value_col)
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

st.subheader("Cleaned price series")
st.line_chart(price)

st.subheader("Log return series")
st.line_chart(returns)

st.write(f"Number of price observations: {len(price)}")
st.write(f"Number of return observations: {len(returns)}")

# 4. Train and test split
st.sidebar.header("3. Train and test split")
test_size_pct = st.sidebar.slider("Test size percent", 10, 50, 20)

n = len(returns)
n_test = max(1, int(n * test_size_pct / 100))
n_train = n - n_test

train = returns.iloc[:n_train]
test = returns.iloc[n_train:]

st.write(f"Total returns: {n}, train: {n_train}, test: {n_test}")

min_obs = 30
if len(train) < min_obs:
    st.warning(
        f"Need at least {min_obs} training observations, "
        f"but only have {len(train)}. Reduce test size or use more data."
    )
    st.stop()

# 5. Model settings
st.sidebar.header("4. Model settings")

# ARIMA settings
p_arima = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1, step=1)
d_arima = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=0, step=1)
q_arima = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=0, step=1)

# OLS AR settings
p_ols = st.sidebar.number_input("OLS AR order p", min_value=1, max_value=5, value=1, step=1)

# GARCH settings
p_garch = st.sidebar.number_input("GARCH p", min_value=1, max_value=5, value=1, step=1)
q_garch = st.sidebar.number_input("GARCH q", min_value=1, max_value=5, value=1, step=1)

forecast_horizon = len(test)

run_models = st.sidebar.button("Run models")


def compute_errors(actual: pd.Series, forecast: pd.Series):
    """Return RMSE and MAE as floats."""
    errors = actual - forecast
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


if run_models:
    results = {}
    forecasts = {}

    # 6. ARIMA model on returns
    try:
        arima_model, arima_forecast = fit_arima(
            train_returns=train,
            order=(p_arima, d_arima, q_arima),
            h=forecast_horizon,
        )
        arima_forecast = pd.Series(arima_forecast.values, index=test.index)
        rmse_a, mae_a = compute_errors(test, arima_forecast)
        results["ARIMA"] = {"RMSE": rmse_a, "MAE": mae_a}
        forecasts["ARIMA"] = arima_forecast
    except Exception as e:
        st.error(f"ARIMA error: {e}")

    # 7. OLS AR(p) model on returns
    try:
        ols_model, ols_forecast = fit_ols_ar(
            train_returns=train,
            p=int(p_ols),
            h=forecast_horizon,
        )
        ols_forecast = pd.Series(ols_forecast.values, index=test.index)
        rmse_o, mae_o = compute_errors(test, ols_forecast)
        results["OLS_AR"] = {"RMSE": rmse_o, "MAE": mae_o}
        forecasts["OLS_AR"] = ols_forecast
    except Exception as e:
        st.error(f"OLS AR error: {e}")

    # 8. GARCH model on returns (volatility)
    garch_results = None
    try:
        garch_res, garch_var_forecast = fit_garch(
            train_returns=train,
            p=int(p_garch),
            q=int(q_garch),
            h=forecast_horizon,
        )
        # garch_var_forecast is a Series of variances for horizons 1..h
        garch_var_forecast = pd.Series(
            garch_var_forecast.values, index=test.index, name="garch_var"
        )

        # realised variance is squared returns in test period
        realised_var = test ** 2

        rmse_g, mae_g = compute_errors(realised_var, garch_var_forecast)
        garch_results = {"RMSE": rmse_g, "MAE": mae_g}
    except Exception as e:
        st.error(f"GARCH error: {e}")

    # 9. Output: return forecasts and metrics
    if results:
        st.subheader("Return forecasts versus actual, test sample")

        chart_df = pd.DataFrame({"actual": test})
        for name, fcast in forecasts.items():
            chart_df[f"{name}_forecast"] = fcast

        st.line_chart(chart_df)

        st.subheader("Forecast accuracy on returns, test sample")
        metrics_df = pd.DataFrame(results).T[["RMSE", "MAE"]]
        st.write(metrics_df)
    else:
        st.warning("No return model ran successfully, check errors above.")

    # 10. Output: volatility forecasts from GARCH
    if garch_results is not None:
        st.subheader("GARCH variance forecast versus realised variance (squared returns)")
        vol_df = pd.DataFrame(
            {
                "realised_var": test ** 2,
                "garch_var_forecast": garch_var_forecast,
            }
        )
        st.line_chart(vol_df)

        st.subheader("GARCH forecast accuracy on variance, test sample")
        st.write(pd.DataFrame({"GARCH": garch_results}).T)
else:
    st.info("Adjust settings in the sidebar and click 'Run models' to estimate ARIMA, OLS, and GARCH.")
