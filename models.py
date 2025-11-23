import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Clean a price series and compute log returns.

    Handles:
    - date parsing
    - sorting and indexing by date
    - thousands separators like '87,537.10'
    """
    data = df[[date_col, value_col]].copy()

    # parse dates
    data[date_col] = pd.to_datetime(data[date_col])

    # sort and index
    data = data.sort_values(date_col).set_index(date_col)

    # clean numeric column, remove commas etc
    col = data[value_col].astype(str).str.replace(",", "").str.strip()
    data[value_col] = pd.to_numeric(col, errors="coerce")

    # drop missing
    data = data.dropna()

    price = data[value_col]
    returns = np.log(price).diff().dropna()
    returns.name = "log_return"

    return price, returns


def fit_arima(train_returns: pd.Series, order=(1, 0, 0), h: int = 10):
    """
    Fit an ARIMA model on train_returns and forecast h steps ahead.
    """
    model = ARIMA(train_returns, order=order).fit()
    forecast = model.forecast(steps=h)
    return model, forecast


def fit_ols_ar(train_returns: pd.Series, p: int = 1, h: int = 10):
    """
    Fit an AR(p) model using OLS:
        r_t = c + phi_1 r_{t-1} + ... + phi_p r_{t-p} + e_t

    Then recursively forecast h steps ahead.
    """
    y = train_returns.dropna()
    if len(y) <= p:
        raise ValueError(f"Not enough observations for OLS AR({p}), have {len(y)}.")

    # build lag matrix
    lags = [y.shift(i) for i in range(1, p + 1)]
    X = pd.concat(lags, axis=1)
    X.columns = [f"lag_{i}" for i in range(1, p + 1)]

    Y = y.iloc[p:]
    X = X.iloc[p:]
    X = sm.add_constant(X)

    ols_model = sm.OLS(Y, X).fit()

    # recursive forecasts
    last_values = list(y.iloc[-p:])  # oldest to newest
    forecasts = []

    const = float(ols_model.params["const"])
    coefs = [float(ols_model.params[f"lag_{i}"]) for i in range(1, p + 1)]

    for _ in range(h):
        # lag_1 is last observed return, lag_2 is previous, etc
        x_lags = list(reversed(last_values))  # [r_t, r_{t-1}, ..., r_{t-p+1}]
        y_hat = const + sum(c * x for c, x in zip(coefs, x_lags))
        forecasts.append(y_hat)
        # update lag buffer
        last_values = last_values[1:] + [y_hat]

    forecast_series = pd.Series(forecasts)
    return ols_model, forecast_series


def fit_garch(train_returns: pd.Series, p: int = 1, q: int = 1, h: int = 10):
    """
    Fit a GARCH(p, q) model on returns and forecast conditional variance h steps ahead.
    Mean is set to zero since you are already modelling returns.
    """
    y = train_returns.dropna()

    if len(y) < max(p, q) + 5:
        raise ValueError(
            f"Not enough observations for GARCH({p},{q}), have {len(y)}."
        )

    am = arch_model(y, vol="Garch", p=p, q=q, mean="Zero", dist="normal")
    res = am.fit(disp="off")

    # variance forecasts for next h steps
    # last row contains variance forecasts for horizon 1..h
    var_forecast = res.forecast(horizon=h).variance.iloc[-1]
    return res, var_forecast
