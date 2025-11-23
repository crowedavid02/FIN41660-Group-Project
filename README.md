# FIN41660 Forecasting Studio

Interactive forecasting studio for the FIN41660 Financial Econometrics group project at UCD.

The app lets you:

- Upload a financial time series in CSV format  
- Clean the data and compute log returns  
- Estimate OLS AR(p), ARIMA and GARCH models  
- Compare forecast accuracy on a held out test horizon  
- Inspect diagnostics, stationarity and residual behaviour  
- Download all key results for use in the written report  

There is also a standalone Python script which runs the same models and prints the same metrics outside the app, as required in the project brief.

---

## 1. Repository structure

```text
FIN41660-forecasting/
├─ app.py              # Streamlit application (main UI)
├─ models.py           # Model and data preparation functions
├─ script.py           # Standalone CLI script to reproduce results
├─ requirements.txt    # Python dependencies
├─ data/               # Optional folder for local CSV files (not required)
└─ README.md           # This file
````

---

## 2. Installation and setup

You need Python 3.10 or 3.11.

### 2.1. Clone the repository

```bash
git clone <your-repo-url>.git
cd FIN41660-forecasting
```

### 2.2. Create and activate a conda environment

If you use Anaconda or Miniconda:

```bash
conda create -n fin41660 python=3.11
conda activate fin41660
```

### 2.3. Install dependencies

Install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you run into missing package errors, you can install the core stack manually:

```bash
pip install streamlit pandas numpy statsmodels arch matplotlib scipy plotly
```

Then regenerate the requirements file:

```bash
pip freeze > requirements.txt
```

---

## 3. Data requirements

The app expects a CSV file with at least:

* One **date** column (daily, weekly or monthly timestamps)
* One **price or index** column (level, not log)

Example columns:

```text
Date,Price
2023-01-01,12345.67
2023-01-02,12410.21
...
```

The app will:

* Parse the date column
* Sort and index by date
* Clean numeric values including thousands separators like `87,537.10`
* Compute log returns

You can also upload other time series as long as they follow the same pattern.

---

## 4. Running the Streamlit app

From the project root, with the environment activated:

```bash
streamlit run app.py
```

This will open the app in your default browser.

### 4.1. Typical workflow in the app

1. **Upload data**
   Use the sidebar to upload your CSV file.

2. **Select columns**
   In the sidebar, choose the date column and the price or index column.

3. **Set train, test and forecast horizon**

   * Choose the percentage of observations to treat as the test window.
   * Choose the forecast horizon (number of last observations to evaluate on).

4. **Set model parameters**

   * ARIMA orders `(p, d, q)`
   * OLS AR order `p`
   * GARCH orders `(p, q)`

   You can keep the defaults for a first pass.

5. **Run models**
   Click the **Run models** button in the sidebar.

6. **Explore the tabs**

   * **Summary**

     * RMSE, MAE and MAPE for ARIMA and OLS on returns
     * RMSE, MAE and MAPE for GARCH on variance
     * Diebold Mariano test comparing ARIMA vs OLS
     * AIC and BIC for each model
     * GARCH volatility persistence measure

   * **Return forecasts**

     * Plot of actual returns vs ARIMA and OLS forecasts over the test horizon
     * Table of actual and predicted returns

   * **Volatility (GARCH)**

     * Plot of realised variance (squared returns) vs GARCH variance forecast
     * Table of variance series

   * **Diagnostics**

     * ADF and KPSS stationarity tests on returns
     * ACF and PACF of returns
     * ACF and PACF of ARIMA and OLS residuals
     * Jarque Bera residual normality test statistics and QQ plots

   * **Downloads**

     * CSVs for return metrics, variance metrics, DM test, model information
     * CSVs for forecasts and variance series

7. **Use downloads in the report**
   The downloaded CSV files can be used to build tables and figures in the written report, for example:

   * Forecast accuracy tables
   * Volatility forecast plots
   * Model comparison tables

---

## 5. Running the standalone script

The standalone script runs the same models outside the app. This is useful for:

* Reproducibility
* Batch runs
* Debugging

From the project root:

```bash
python script.py path/to/your_data.csv --date Date --price Price
```

Key options:

* `--date`
  Name of the date column. Default is `Date`.

* `--price`
  Name of the price or index column. Default is `Price`.

* `--test_size`
  Percentage of the sample to reserve as test data (default 20).

* `--arima`
  ARIMA orders `p d q`, for example:

  ```bash
  --arima 1 0 1
  ```

* `--ols_p`
  OLS AR lag order `p` (default 1).

* `--garch`
  GARCH orders `p q`, for example:

  ```bash
  --garch 1 1
  ```

* `--save`
  If included, the script will save forecast series to CSV files in the current folder.

Example:

```bash
python script.py bitcoin.csv \
  --date Date --price Price \
  --test_size 20 \
  --arima 1 0 1 --ols_p 2 --garch 1 1 \
  --save
```

The script prints:

* Number of observations
* Train and test sizes
* RMSE and MAE for ARIMA and OLS on returns
* RMSE and MAE for GARCH on variance

When run with `--save`, it also writes forecast CSV files.

---

## 6. For collaborators

If you are joining this project:

1. Clone the repository and create a Python environment as described above.
2. Run `streamlit run app.py` to explore the app.
3. Run `python script.py ...` to reproduce main results from the command line.
4. Keep all new model logic inside `models.py` so that both the app and script can reuse it.
5. Avoid editing `app.py` directly unless you are changing the interface.
6. Commit changes in small steps with clear messages:

```bash
git add <files>
git commit -m "Short description of the change"
git push
```

---

## 7. Notes

* The app is intended as a single series forecasting studio, not a multi asset system.
* The Diebold Mariano test implementation is simplified and does not include HAC corrections. It is suitable for classroom model comparison but not for production use.
* GARCH is estimated under a normal distribution. For heavy tailed assets you may want to experiment with Student t or skewed distributions in a future extension.

```
## Extra UI polish for `app.py`

Your current app is already very strong. The code you just ran is fine to keep. The only extra polish I would suggest now is *very minor* and does not warrant another giant paste unless you want specific cosmetic tweaks.

Given how much you have working, I would not keep churning the layout. From here, polish comes more from:

- consistent naming in labels  
- writing short, clear captions under key charts  
- adjusting default hyperparameters sensibly for your asset (for example ARIMA(1,0,1), GARCH(1,1))  
- being consistent between the app and the report, so what the examiner sees in the video matches what they read in the document.

If you really want to push cosmetics a bit further, you can:

- set a dark theme in `.streamlit/config.toml` (so the whole app uses a consistent dark mode)  
- add one or two `st.expander` blocks for "What does ARIMA do" and "What does GARCH do" so the app itself is self contained as a teaching tool  
- tighten some wording in the hero section and tab headings to match the style of your report title.

If you want those specific config or expander snippets, say so and I will give you short, targeted code blocks rather than another full app rewrite.
```
