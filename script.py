import argparse
import pandas as pd
import numpy as np

from models import prepare_series, fit_arima, fit_ols_ar, fit_garch


def compute_errors(actual: pd.Series, forecast: pd.Series):
    errors = actual - forecast
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Run ARIMA, OLS AR(p), and GARCH(1,1) forecasting.")
    parser.add_argument("csv_file", type=str, help="Path to CSV file")
    parser.add_argument("--date", type=str, default="Date", help="Name of the date column")
    parser.add_argument("--price", type=str, default="Price", help="Name of the price column")
    parser.add_argument("--test_pct", type=float, default=20, help="Test size percent (default 20)")
    parser.add_argument("--arima", nargs=3, type=int, default=[1, 0, 0], help="ARIMA order p d q")
    parser.add_argument("--ols_p", type=int, default=1, help="OLS AR(p) lag order")
    parser.add_argument("--garch", nargs=2, type=int, default=[1, 1], help="GARCH(p, q)")
    parser.add_argument("--save", action="store_true", help="Save forecasts to CSV files")

    args = parser.parse_args()

    print("\n--- Loading data ---")
    df = pd.read_csv(args.csv_file)
    print(df.head(), "\n")

    print("--- Preparing series ---")
    price, returns = prepare_series(df, args.date, args.price)
    print(f"Price observations: {len(price)}")
    print(f"Return observations: {len(returns)}\n")

    # Train-test split
    n = len(returns)
    n_test = max(1, int(n * args.test_pct / 100))
    n_train = n - n_test

    train = returns.iloc[:n_train]
    test = returns.iloc[n_train:]

    print(f"Train size: {len(train)}, Test size: {len(test)}\n")

    forecasts = {}
    metrics = {}

    # ARIMA
    print("--- Running ARIMA ---")
    try:
        p, d, q = args.arima
        arima_model, arima_fc = fit_arima(train_returns=train, order=(p, d, q), h=len(test))
        arima_fc = pd.Series(arima_fc.values, index=test.index)
        rmse, mae = compute_errors(test, arima_fc)
        forecasts["ARIMA"] = arima_fc
        metrics["ARIMA"] = (rmse, mae)
        print(f"ARIMA RMSE={rmse:.6f}, MAE={mae:.6f}")
    except Exception as e:
        print(f"ARIMA error: {e}")

    # OLS AR
    print("\n--- Running OLS AR(p) ---")
    try:
        ols_model, ols_fc = fit_ols_ar(train_returns=train, p=args.ols_p, h=len(test))
        ols_fc = pd.Series(ols_fc.values, index=test.index)
        rmse, mae = compute_errors(test, ols_fc)
        forecasts["OLS_AR"] = ols_fc
        metrics["OLS_AR"] = (rmse, mae)
        print(f"OLS AR(p={args.ols_p}) RMSE={rmse:.6f}, MAE={mae:.6f}")
    except Exception as e:
        print(f"OLS error: {e}")

    # GARCH
    print("\n--- Running GARCH ---")
    try:
        p_g, q_g = args.garch
        garch_res, garch_var_fc = fit_garch(train_returns=train, p=p_g, q=q_g, h=len(test))
        garch_var_fc = pd.Series(garch_var_fc.values, index=test.index)
        realised_var = test ** 2
        rmse, mae = compute_errors(realised_var, garch_var_fc)
        forecasts["GARCH_VAR"] = garch_var_fc
        metrics["GARCH"] = (rmse, mae)
        print(f"GARCH RMSE={rmse:.6f}, MAE={mae:.6f}")
    except Exception as e:
        print(f"GARCH error: {e}")

    print("\n--- Summary of Errors ---")
    for model, (rmse, mae) in metrics.items():
        print(f"{model}: RMSE={rmse:.6f}, MAE={mae:.6f}")

    if args.save:
        print("\nSaving forecast CSV files...")
        for name, series in forecasts.items():
            series.to_csv(f"{name}_forecast.csv")
        print("Saved.\n")


if __name__ == "__main__":
    main()
