import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import itertools
import matplotlib.pyplot as plt



# Define file paths using raw strings.
dataDay1  = r"C:\Users\conor\OneDrive\Documents\College\Prosperity\prices_round_2_day_1.csv"
dataDay0  = r"C:\Users\conor\OneDrive\Documents\College\Prosperity\prices_round_2_day_0.csv"
dataDaym1 = r"C:\Users\conor\OneDrive\Documents\College\Prosperity\prices_round_2_day_-1.csv"

# Read each CSV file with the specified delimiter.
df0  = pd.read_csv(dataDay0, sep=';')
dfm1 = pd.read_csv(dataDaym1, sep=';')
df1  = pd.read_csv(dataDay1, sep=';')

# --- For Day 0 ---
rainforest_resin_0 = df0[df0['product'] == 'RAINFOREST_RESIN']
djembe_0           = df0[df0['product'] == 'DJEMBES']
croissants_0       = df0[df0['product'] == 'CROISSANTS']
jams_0             = df0[df0['product'] == 'JAMS']
kelp_0             = df0[df0['product'] == 'KELP']
squid_ink_0        = df0[df0['product'] == 'SQUID_INK']
picnic_basket1_0   = df0[df0['product'] == 'PICNIC_BASKET1']
picnic_basket2_0   = df0[df0['product'] == 'PICNIC_BASKET2']

# --- For Day -1 ---
rainforest_resin_m1 = dfm1[dfm1['product'] == 'RAINFOREST_RESIN']
djembe_m1           = dfm1[dfm1['product'] == 'DJEMBES']
croissants_m1       = dfm1[dfm1['product'] == 'CROISSANTS']
jams_m1             = dfm1[dfm1['product'] == 'JAMS']
kelp_m1             = dfm1[dfm1['product'] == 'KELP']
squid_ink_m1        = dfm1[dfm1['product'] == 'SQUID_INK']
picnic_basket1_m1   = dfm1[dfm1['product'] == 'PICNIC_BASKET1']
picnic_basket2_m1   = dfm1[dfm1['product'] == 'PICNIC_BASKET2']

# --- For Day 1 ---
rainforest_resin_1 = df1[df1['product'] == 'RAINFOREST_RESIN']
djembe_1           = df1[df1['product'] == 'DJEMBES']
croissants_1       = df1[df1['product'] == 'CROISSANTS']
jams_1             = df1[df1['product'] == 'JAMS']
kelp_1             = df1[df1['product'] == 'KELP']
squid_ink_1        = df1[df1['product'] == 'SQUID_INK']
picnic_basket1_1   = df1[df1['product'] == 'PICNIC_BASKET1']
picnic_basket2_1   = df1[df1['product'] == 'PICNIC_BASKET2']



rainforest_resin_all = pd.concat([rainforest_resin_m1, rainforest_resin_0, rainforest_resin_1], axis=0).reset_index(drop=True)
djembe_all = pd.concat([djembe_m1, djembe_0, djembe_1], axis=0).reset_index(drop=True)
croissants_all = pd.concat([croissants_m1, croissants_0, croissants_1], axis=0).reset_index(drop=True)
jams_all = pd.concat([jams_m1, jams_0, jams_1], axis=0).reset_index(drop=True)
kelp_all = pd.concat([kelp_m1, kelp_0, kelp_1], axis=0).reset_index(drop=True)
squid_ink_all = pd.concat([squid_ink_m1, squid_ink_0, squid_ink_1], axis=0).reset_index(drop=True)
picnic_basket1_all = pd.concat([picnic_basket1_m1, picnic_basket1_0, picnic_basket1_1], axis=0).reset_index(drop=True)
picnic_basket2_all = pd.concat([picnic_basket2_m1, picnic_basket2_0, picnic_basket2_1], axis=0).reset_index(drop=True)




# Assume you have two time series from your trading products:
# For example, 'price_series_1' and 'price_series_2'
price_series_1 = rainforest_resin_all['mid_price']  # Replace with your actual column
price_series_2 = djembe_all['mid_price']  # Replace with your actual column

# Step 1: Regress one series on the other (here, price_series_1 on price_series_2)
X = sm.add_constant(price_series_2)
model = sm.OLS(price_series_1, X).fit()
residuals = model.resid






products = {

    "DJEMBES": djembe_all,
    "CROISSANTS": croissants_all,
    "JAMS": jams_all,
    "KELP": kelp_all,
    "SQUID_INK": squid_ink_all,
}

# Dictionary to store only the significant cointegration results.
significant_pairs = {}

# Loop over all unique pairs of products.
for prod1, prod2 in itertools.combinations(products.keys(), 2):
    # Extract the price series for each product.
    price_series_1 = products[prod1]['mid_price']
    price_series_2 = products[prod2]['mid_price']

    # Run the cointegration test using the Engle-Granger method.
    score, pvalue, _ = coint(price_series_1, price_series_2)

    # Only store this pair if pvalue is less than 0.05
    if pvalue < 0.05:
        significant_pairs[(prod1, prod2)] = pvalue

# Print out the significant cointegration test results.




# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


# --- Compute Cointegration Spread ---
def compute_cointegration(price_series1, price_series2):
    X = sm.add_constant(price_series2)
    model = sm.OLS(price_series1, X).fit()
    spread = model.resid
    return spread, model

# --- Compute Rolling Z-Score ---
def compute_rolling_zscore(spread, window=100):
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

# Example: Using your DJEMBES vs CROISSANTS data
price_series_A = djembe_all['mid_price']       # ensure djembe_all is defined
price_series_B = croissants_all['mid_price']    # ensure croissants_all is defined

spread, model = compute_cointegration(price_series_A, price_series_B)

# Set a rolling window (you can use a default or one based on your half-life)
window = 473
zscore = compute_rolling_zscore(spread, window=window)

# --- Generate Trading Signals ---
# Determine entry thresholds as the 5th and 95th percentiles
lower_entry = zscore.quantile(0.05)
upper_entry = zscore.quantile(0.95)

# Create a signals DataFrame.
signals = pd.DataFrame(index=spread.index)
signals['zscore'] = zscore
signals['signal'] = 0

# Entry signals: long if zscore < lower threshold, short if zscore > upper threshold.
signals.loc[zscore < lower_entry, 'signal'] = 1    # long signal
signals.loc[zscore > upper_entry, 'signal'] = -1   # short signal

# Exit signals: When the zscore reverts close to zero (using an exit threshold, e.g., 0.5).
exit_threshold = 0.5
signals.loc[(zscore > -exit_threshold) & (zscore < exit_threshold), 'signal'] = 0

# Forward-fill the position so that once you enter, the position is held until exit.
signals['position'] = signals['signal'].replace(to_replace=0, method='ffill')








def simulate_strategy(spread, zscore, lower_entry, upper_entry, exit_threshold):
    """
    Simulate a simple pairs trading strategy.

    Parameters:
      - spread: Cointegrating residual (the spread) as a pandas Series.
      - zscore: Rolling z-score of the spread.
      - lower_entry: Entry threshold for long signal (e.g., 5th percentile).
      - upper_entry: Entry threshold for short signal (e.g., 95th percentile).
      - exit_threshold: When abs(zscore) falls below this value, we exit the trade.

    Returns:
      - cumulative_return: Total return from the strategy.
      - sharpe_ratio: (Mean return / Std of return) annualized (using a factor for trading days).
      - signals: DataFrame with signals generated.
    """
    signals = pd.DataFrame(index=spread.index)
    signals['zscore'] = zscore
    signals['signal'] = 0

    # Entry signals:
    signals.loc[zscore < lower_entry, 'signal'] = 1   # Go long the spread.
    signals.loc[zscore > upper_entry, 'signal'] = -1   # Go short the spread.

    # Exit: When the absolute zscore is below exit_threshold, we set the signal to 0.
    signals.loc[(zscore > -exit_threshold) & (zscore < exit_threshold), 'signal'] = 0

    # Forward-fill positions until the signal changes, meaning you hold your position.
    signals['position'] = signals['signal'].replace(to_replace=0, method='ffill')

    # Compute returns: Assume strategy return = previous position * change in spread.
    # This is a simplified profit simulation.
    returns = signals['position'].shift(1) * spread.diff()
    returns = returns.fillna(0)

    # Calculate performance metrics
    cumulative_return = returns.sum()
    # Using 252 periods as annualization factor (if daily data). Adjust if time steps differ.
    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    return cumulative_return, sharpe_ratio, signals

# ------------------------------
# Apply to Your Data: DJEMBES vs CROISSANTS
# ------------------------------

# Assume these DataFrames are defined and each contains a 'mid_price' column.
price_series_A = jams_all['mid_price']
price_series_B = squid_ink_all['mid_price']

# Compute cointegration spread.
spread, model = compute_cointegration(price_series_A, price_series_B)

# Compute half-life (for information / possibly to set window).

# Set a rolling window; if half_life is valid, use it (rounded to int), else a default (e.g., 100).
#window = int(half_life) if (half_life is not np.nan and half_life > 0) else 100

# Compute rolling z-score.
zscore = compute_rolling_zscore(spread, window=window)

# Determine entry thresholds using quantiles from historical z-score.
lower_entry = zscore.quantile(0.05)  # e.g., around -2.14 from your previous run
upper_entry = zscore.quantile(0.95)  # e.g., around 2.24

print("Lower entry threshold (for long):", lower_entry)
print("Upper entry threshold (for short):", upper_entry)

# ------------------------------
# Testing Optimal Exit Thresholds
# ------------------------------

# Define a range of candidate exit threshold values.
exit_threshold_candidates = np.linspace(0.1, 2.0, 20)
results = []

for exit_th in exit_threshold_candidates:
    cum_ret, sharpe, _ = simulate_strategy(spread, zscore, lower_entry, upper_entry, exit_th)
    results.append({'exit_threshold': exit_th, 'cumulative_return': cum_ret, 'sharpe_ratio': sharpe})

results_df = pd.DataFrame(results)
print(results_df)