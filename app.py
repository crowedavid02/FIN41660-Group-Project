import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import t
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera

from models import prepare_series, fit_arima, fit_ols_ar, fit_garch

# ---------------------------------------------------------
# Page config and simple styling
# ---------------------------------------------------------
st.set_page_config(page_title="FIN41660 Forecasting Studio", layout="wide")

st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1rem 1.5rem;
        border-radius: 0.9rem;
        background: #111827;
        border: 1px solid #1f2933;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero section
col_hero_left, col_hero_right = st.columns([2.2, 1.2])

with col_hero_left:
    st.markdown(
        "<div class='big-title'>FIN41660 Forecasting Studio</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Upload any financial time series, estimate ARIMA, "
        "OLS AR(p) and GARCH models, and compare their forecasting performance.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **How to use this app**

        1. Use the sidebar to upload a CSV file with a index or price series.  
        2. Select the date and price columns.  
        3. Choose train–test split and model settings (or keep the defaults).  
        4. Click **Run models**.  
        5. Explore the tabs below for forecasts, volatility, diagnostics and downloads.  
        """
    )

with col_hero_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Quick checklist**")
    st.markdown(
        "- ✅ CSV format\n"
        "- ✅ One date column\n"
        "- ✅ One price/index column\n"
        "- ✅ Returns reasonably stationary\n"
        "- ✅ At least 50 observations recommended"
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def compute_errors(actual: pd.Series, forecast: pd.Series):
    """Return RMSE, MAE, MAPE (percent)."""
    errors = actual - forecast
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(actual != 0, errors / actual, np.nan)
    mape = float(np.nanmean(np.abs(pct))) * 100.0

    return rmse, mae, mape


def dm_test(e1: pd.Series, e2: pd.Series, power: int = 2):
    """
    Simple Diebold–Mariano style test comparing two forecast error series
    under squared-error loss. No HAC correction, fine for classroom use.
    """
    d = (np.abs(e1) ** power - np.abs(e2) ** power).dropna()
    if len(d) < 5:
        return np.nan, np.nan

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if np.isclose(var_d, 0.0):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    df = len(d) - 1
    p_value = 2.0 * (1.0 - t.cdf(np.abs(dm_stat), df=df))
    return float(dm_stat), float(p_value)


def plot_acf_pacf(series: pd.Series, title_prefix: str):
    """Return a matplotlib figure with ACF and PACF side by side."""
    series = series.dropna()
    lags = min(20, max(5, len(series) // 4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF")
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title_prefix} PACF")
    fig.tight_layout()
    return fig


def run_stationarity_tests(series: pd.Series):
    series = series.dropna()
    adf_stat, adf_p, _, _, crit_adf, _ = adfuller(series, autolag="AIC")
    try:
        kpss_stat, kpss_p, _, crit_kpss = kpss(series, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, crit_kpss = np.nan, np.nan, {}

    results = {
        "ADF": {
            "stat": adf_stat,
            "p_value": adf_p,
            "crit": crit_adf,
            "null": "unit root (non-stationary)",
        },
        "KPSS": {
            "stat": kpss_stat,
            "p_value": kpss_p,
            "crit": crit_kpss,
            "null": "stationary (level)",
        },
    }
    return results


def run_normality_test(resid: pd.Series):
    """Jarque–Bera normality test on residuals."""
    resid = resid.dropna()
    if len(resid) < 8:
        return np.nan, np.nan, np.nan, np.nan
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    return float(jb_stat), float(jb_p), float(skew), float(kurt)


def plot_qq(resid: pd.Series, title: str):
    fig = plt.figure(figsize=(4, 4))
    qqplot(resid.dropna(), line="s", ax=plt.gca())
    plt.title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Sidebar: data and settings
# ---------------------------------------------------------
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file with a date column and a price/index column to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head())

st.sidebar.header("2. Columns")
date_col = st.sidebar.selectbox("Date column", df.columns)
value_col = st.sidebar.selectbox(
    "Price / index column",
    [c for c in df.columns if c != date_col],
)

# prepare series
try:
    price, returns = prepare_series(df, date_col, value_col)
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

st.subheader("Price and return series")

c_price, c_ret = st.columns(2)
with c_price:
    st.markdown("**Cleaned price series**")
    df_price = price.reset_index()
    df_price.columns = ["Date", "Price"]
    fig_price = px.line(df_price, x="Date", y="Price")
    st.plotly_chart(fig_price, use_container_width=True)

with c_ret:
    st.markdown("**Log return series**")
    df_ret = returns.reset_index()
    df_ret.columns = ["Date", "Return"]
    fig_ret = px.line(df_ret, x="Date", y="Return")
    st.plotly_chart(fig_ret, use_container_width=True)

st.write(f"Number of price observations: {len(price)}")
st.write(f"Number of return observations: {len(returns)}")

st.sidebar.header("3. Train, test and horizon")
test_size_pct = st.sidebar.slider("Test set size (% of returns)", 10, 50, 20)

n = len(returns)
n_test = max(1, int(n * test_size_pct / 100))
n_train = n - n_test

train_full = returns.iloc[:n_train]
test_full = returns.iloc[n_train:]

st.write(f"Total returns: **{n}**, train: **{n_train}**, test window: **{n_test}** obs")

min_obs = 30
if len(train_full) < min_obs:
    st.warning(
        f"Need at least {min_obs} training observations, "
        f"but only have {len(train_full)}. Reduce test size or use more data."
    )
    st.stop()

forecast_horizon = st.sidebar.slider(
    "Forecast horizon within test window (steps ahead)",
    min_value=1,
    max_value=max(1, n_test),
    value=min(10, n_test),
)

# redefine train / test based on horizon
h = min(forecast_horizon, n_test)
train = returns.iloc[: n - h]
test = returns.iloc[n - h :]

st.write(
    f"Using last **{h}** returns as evaluation horizon. "
    f"Training sample size: **{len(train)}**."
)

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

run_models = st.sidebar.button("Run models")

if not run_models:
    st.info("Adjust settings in the sidebar and click **Run models** to estimate ARIMA, OLS and GARCH.")
    st.stop()

# ---------------------------------------------------------
# Model estimation
# ---------------------------------------------------------
results_returns = {}
results_var = {}
forecasts = {}
residuals = {}
info = {}

arima_model = None
ols_model = None
garch_res = None
garch_var_forecast = None
realised_var = None

# ARIMA
try:
    arima_model, arima_forecast = fit_arima(
        train_returns=train,
        order=(p_arima, d_arima, q_arima),
        h=h,
    )
    arima_forecast = pd.Series(arima_forecast.values, index=test.index)
    rmse_a, mae_a, mape_a = compute_errors(test, arima_forecast)
    results_returns["ARIMA"] = {"RMSE": rmse_a, "MAE": mae_a, "MAPE": mape_a}
    forecasts["ARIMA"] = arima_forecast
    residuals["ARIMA"] = pd.Series(arima_model.resid, index=train.index)
    info["ARIMA"] = {"AIC": arima_model.aic, "BIC": arima_model.bic}
except Exception as e:
    st.error(f"ARIMA error: {e}")

# OLS AR(p)
try:
    ols_model, ols_forecast = fit_ols_ar(
        train_returns=train,
        p=int(p_ols),
        h=h,
    )
    ols_forecast = pd.Series(ols_forecast.values, index=test.index)
    rmse_o, mae_o, mape_o = compute_errors(test, ols_forecast)
    results_returns["OLS_AR"] = {"RMSE": rmse_o, "MAE": mae_o, "MAPE": mape_o}
    forecasts["OLS_AR"] = ols_forecast
    residuals["OLS_AR"] = pd.Series(
        ols_model.resid,
        index=train.iloc[int(p_ols):].index,
    )
    info["OLS_AR"] = {"AIC": ols_model.aic, "BIC": ols_model.bic}
except Exception as e:
    st.error(f"OLS AR error: {e}")

# GARCH variance
garch_results = None
try:
    garch_res, garch_var_forecast = fit_garch(
        train_returns=train,
        p=int(p_garch),
        q=int(q_garch),
        h=h,
    )
    garch_var_forecast = pd.Series(
        garch_var_forecast.values, index=test.index, name="garch_var"
    )
    realised_var = test ** 2
    rmse_g, mae_g, mape_g = compute_errors(realised_var, garch_var_forecast)
    garch_results = {"RMSE": rmse_g, "MAE": mae_g, "MAPE": mape_g}
    results_var["GARCH"] = garch_results
    residuals["GARCH"] = pd.Series(garch_res.resid, index=train.index)

    # persistence: sum of alpha and beta coefficients
    params = garch_res.params
    alpha = sum(v for k, v in params.items() if "alpha" in k)
    beta = sum(v for k, v in params.items() if "beta" in k)
    info["GARCH"] = {
        "AIC": garch_res.aic,
        "BIC": garch_res.bic,
        "persistence": alpha + beta,
    }
except Exception as e:
    st.error(f"GARCH error: {e}")

if not results_returns and garch_results is None:
    st.warning("No model ran successfully. Check the error messages above.")
    st.stop()

# DM test between ARIMA and OLS on test errors
dm_results = {}
if "ARIMA" in forecasts and "OLS_AR" in forecasts:
    e_arima = test - forecasts["ARIMA"]
    e_ols = test - forecasts["OLS_AR"]
    dm_stat, dm_pval = dm_test(e_arima, e_ols)
    dm_results["ARIMA vs OLS_AR"] = {
        "DM_stat": dm_stat,
        "p_value": dm_pval,
    }

# Stationarity tests on returns (full series)
stationarity = run_stationarity_tests(returns)

# ---------------------------------------------------------
# Tabs for dashboard
# ---------------------------------------------------------
tab_summary, tab_returns, tab_vol, tab_diag, tab_downloads = st.tabs(
    ["Summary", "Return forecasts", "Volatility (GARCH)", "Diagnostics", "Downloads"]
)

# Summary tab
with tab_summary:
    st.subheader("Forecast accuracy on returns (test horizon)")

    metrics_df = pd.DataFrame(results_returns).T[["RMSE", "MAE", "MAPE"]]
    st.dataframe(metrics_df.style.format("{:.6f}"))

    cols = st.columns(len(results_returns))
    for col, (name, m) in zip(cols, results_returns.items()):
        col.metric(
            label=f"{name} RMSE",
            value=f"{m['RMSE']:.4f}",
            help="Lower is better on the held-out test horizon.",
        )

    best_model = min(
        results_returns.items(), key=lambda kv: kv[1]["RMSE"]
    )
    st.markdown(
        f"**Best performing return model by RMSE on this dataset: "
        f"{best_model[0]} (RMSE ≈ {best_model[1]['RMSE']:.4f}).**"
    )

    if results_var:
        st.subheader("Forecast accuracy on variance (squared returns, test horizon)")
        var_df = pd.DataFrame(results_var).T[["RMSE", "MAE", "MAPE"]]
        st.dataframe(var_df.style.format("{:.6f}"))

    if dm_results:
        st.subheader("Diebold–Mariano comparison (squared error loss)")
        dm_df = pd.DataFrame(dm_results).T
        st.table(dm_df.style.format("{:.4f}"))
        st.caption(
            "Low p values suggest a statistically significant difference in predictive "
            "accuracy between the two return models."
        )

    st.subheader("Model information (AIC, BIC, GARCH persistence)")
    info_df = pd.DataFrame(info).T
    st.dataframe(info_df.style.format("{:.4f}"))
    st.caption(
        "Lower AIC/BIC values indicate a better trade-off between fit and parsimony. "
        "For GARCH, persistence close to 1 indicates highly persistent volatility."
    )

# Returns tab
with tab_returns:
    st.subheader("Return forecasts versus actual (test horizon)")
    chart_df = pd.DataFrame({"actual": test})
    for name, fcast in forecasts.items():
        chart_df[f"{name}_forecast"] = fcast

    chart_df_plot = chart_df.reset_index()
    chart_df_plot.columns = ["Date"] + list(chart_df.columns)

    fig_ret_fc = px.line(chart_df_plot, x="Date", y=chart_df.columns)
    st.plotly_chart(fig_ret_fc, use_container_width=True)

    st.write("Test horizon data and forecasts")
    st.dataframe(chart_df.tail(20))

# Volatility (GARCH) tab
with tab_vol:
    if garch_results is None or garch_var_forecast is None:
        st.info("GARCH model did not run successfully.")
    else:
        st.subheader("Variance forecast versus realised variance (squared returns)")
        vol_df = pd.DataFrame(
            {
                "realised_var": realised_var,
                "garch_var_forecast": garch_var_forecast,
            }
        )
        vol_df_plot = vol_df.reset_index()
        vol_df_plot.columns = ["Date", "realised_var", "garch_var_forecast"]
        fig_vol = px.line(
            vol_df_plot,
            x="Date",
            y=["realised_var", "garch_var_forecast"],
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        st.dataframe(vol_df.tail(20))

# Diagnostics tab
with tab_diag:
    st.subheader("Stationarity tests on returns (full sample)")

    adf_res = stationarity["ADF"]
    kpss_res = stationarity["KPSS"]

    diag_df = pd.DataFrame(
        {
            "Test": ["ADF", "KPSS"],
            "Statistic": [adf_res["stat"], kpss_res["stat"]],
            "p_value": [adf_res["p_value"], kpss_res["p_value"]],
            "Null hypothesis": [adf_res["null"], kpss_res["null"]],
        }
    )
    st.table(diag_df.style.format({"Statistic": "{:.4f}", "p_value": "{:.4f}"}))
    st.caption(
        "ADF rejects the null of a unit root at low p values (favouring stationarity). "
        "KPSS rejects the null of stationarity at low p values (favouring a unit root)."
    )

    st.subheader("ACF and PACF of returns")
    fig = plot_acf_pacf(returns, title_prefix="Returns")
    st.pyplot(fig)

    # residual diagnostics
    if "ARIMA" in residuals:
        st.subheader("ARIMA residual diagnostics")
        fig_ar_acf = plot_acf_pacf(residuals["ARIMA"], title_prefix="ARIMA residuals")
        st.pyplot(fig_ar_acf)

        jb_stat, jb_p, skew, kurt = run_normality_test(residuals["ARIMA"])
        st.write(
            f"Jarque–Bera (ARIMA residuals): stat = {jb_stat:.4f}, "
            f"p-value = {jb_p:.4f}, skew = {skew:.3f}, kurtosis = {kurt:.3f}"
        )
        fig_qq_ar = plot_qq(residuals["ARIMA"], "ARIMA residuals QQ plot")
        st.pyplot(fig_qq_ar)

    if "OLS_AR" in residuals:
        st.subheader("OLS AR residual diagnostics")
        fig_ols_acf = plot_acf_pacf(residuals["OLS_AR"], title_prefix="OLS residuals")
        st.pyplot(fig_ols_acf)

        jb_stat, jb_p, skew, kurt = run_normality_test(residuals["OLS_AR"])
        st.write(
            f"Jarque–Bera (OLS residuals): stat = {jb_stat:.4f}, "
            f"p-value = {jb_p:.4f}, skew = {skew:.3f}, kurtosis = {kurt:.3f}"
        )
        fig_qq_ols = plot_qq(residuals["OLS_AR"], "OLS residuals QQ plot")
        st.pyplot(fig_qq_ols)

    st.caption(
        "Residual diagnostics help you check whether the mean and volatility models "
        "have captured most predictable structure. Large remaining autocorrelation or "
        "strong non-normality suggests scope for improvement."
    )

# Downloads tab
with tab_downloads:
    st.subheader("Download results")

    metrics_returns_df = pd.DataFrame(results_returns).T
    metrics_var_df = pd.DataFrame(results_var).T if results_var else pd.DataFrame()
    dm_df = pd.DataFrame(dm_results).T if dm_results else pd.DataFrame()
    info_out_df = pd.DataFrame(info).T

    st.write("Download forecast accuracy for returns")
    st.download_button(
        label="Download return metrics (CSV)",
        data=metrics_returns_df.to_csv().encode("utf-8"),
        file_name="return_metrics.csv",
        mime="text/csv",
    )

    if not metrics_var_df.empty:
        st.write("Download forecast accuracy for variance")
        st.download_button(
            label="Download variance metrics (CSV)",
            data=metrics_var_df.to_csv().encode("utf-8"),
            file_name="variance_metrics.csv",
            mime="text/csv",
        )

    if not dm_df.empty:
        st.write("Download Diebold–Mariano results")
        st.download_button(
            label="Download DM test results (CSV)",
            data=dm_df.to_csv().encode("utf-8"),
            file_name="dm_test_results.csv",
            mime="text/csv",
        )

    st.write("Download model information (AIC, BIC, persistence)")
    st.download_button(
        label="Download model info (CSV)",
        data=info_out_df.to_csv().encode("utf-8"),
        file_name="model_info.csv",
        mime="text/csv",
    )

    # forecasts and variance
    if forecasts:
        all_fc = pd.DataFrame({"actual": test})
        for name, fcast in forecasts.items():
            all_fc[f"{name}_forecast"] = fcast
        st.write("Download test horizon forecasts")
        st.download_button(
            label="Download forecasts (CSV)",
            data=all_fc.to_csv().encode("utf-8"),
            file_name="forecasts.csv",
            mime="text/csv",
        )

    if garch_var_forecast is not None:
        var_out = pd.DataFrame(
            {
                "realised_var": realised_var,
                "garch_var_forecast": garch_var_forecast,
            }
        )
        st.write("Download variance series")
        st.download_button(
            label="Download variance series (CSV)",
            data=var_out.to_csv().encode("utf-8"),
            file_name="variance_series.csv",
            mime="text/csv",
        )

    st.caption("These CSVs are ideal for including tables and plots in your written report.")
