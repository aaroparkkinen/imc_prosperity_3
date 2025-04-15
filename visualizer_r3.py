#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:36:07 2025

@author: aaroparkkinen
"""
import math as m
import pandas as pd
import numpy as np
from scipy.stats import norm



# Define file paths using raw strings.
dataDay0  = r"/Users/aaroparkkinen/Desktop/prices_round_3_day_0.csv"
dataDay1  = r"/Users/aaroparkkinen/Desktop/prices_round_3_day_1.csv"
dataDay2 = r"/Users/aaroparkkinen/Desktop/prices_round_3_day_2.csv"

# Read each CSV file with the specified delimiter.
df0  = pd.read_csv(dataDay0, sep=';')
df1 = pd.read_csv(dataDay1, sep=';')
df2  = pd.read_csv(dataDay2, sep=';')

# --- For Day 0 ---
# (Optional) Add a column to indicate the day. This makes it easier to distinguish rows later.
df0['day'] = 0
df1['day'] = 1
df2['day'] = 2
# List of the products you need.
products = [
    'CROISSANTS',
    'DJEMBES',
    'JAMS',
    'KELP',
    'PICNIC_BASKET1',
    'PICNIC_BASKET2',
    'RAINFOREST_RESIN',
    'SQUID_INK',
    'VOLCANIC_ROCK',
    'VOLCANIC_ROCK_VOUCHER_9500',
    'VOLCANIC_ROCK_VOUCHER_9750',
    'VOLCANIC_ROCK_VOUCHER_10000',
    'VOLCANIC_ROCK_VOUCHER_10250',
    'VOLCANIC_ROCK_VOUCHER_10500'
]
 
# Combine the data from all days for each product.
dfs = [df0, df1, df2]
products_df = {
    product: pd.concat([df[df['product'] == product] for df in dfs], ignore_index=True)
    for product in products
}
 
# List of voucher product names
voucher_products = [
    'VOLCANIC_ROCK_VOUCHER_9500',
    'VOLCANIC_ROCK_VOUCHER_9750',
    'VOLCANIC_ROCK_VOUCHER_10000',
    'VOLCANIC_ROCK_VOUCHER_10250',
    'VOLCANIC_ROCK_VOUCHER_10500'
]
# Map product names to strike prices
strike_prices = {
    'VOLCANIC_ROCK_VOUCHER_9500': 9500,
    'VOLCANIC_ROCK_VOUCHER_9750': 9750,
    'VOLCANIC_ROCK_VOUCHER_10000': 10000,
    'VOLCANIC_ROCK_VOUCHER_10250': 10250,
    'VOLCANIC_ROCK_VOUCHER_10500': 10500
}

# Loop over each voucher product and compute the new TTE value
# Get underlying product prices (VOLCANIC_ROCK)
underlying_df = products_df['VOLCANIC_ROCK'][['timestamp', 'day', 'mid_price']].rename(columns={'mid_price': 'underlying_price'})

# Loop through voucher products
for product in voucher_products:
    if product in products_df:
        df = products_df[product]
        df['TTE'] = 7000000 - df['timestamp'] - (1000000 * df['day'])
        df['K'] = strike_prices[product]
        df = df[df['TTE'] > 0]

        # Merge underlying price into the voucher df based on timestamp and day
        df = df.merge(underlying_df, on=['timestamp', 'day'], how='left')

        # Calculate m_t using the underlying price (VOLCANIC_ROCK)
        df['m_t'] = np.log(df['K'] / df['mid_price']) / np.sqrt(df['TTE'])

        # Save updated dataframe back
        products_df[product] = df

def bs_price(option_type, S, K, sigma, r, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def implied_vol(option_type, S, K, r, T, market_price, v_star=0.3, eps=1e-5, max_iter=100):
    if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
        return np.nan

    sigma = v_star
    for _ in range(max_iter):
        try:
            price = bs_price(option_type, S, K, sigma, r, T)
            vega = (bs_price(option_type, S, K, sigma + 1e-5, r, T) -
                    bs_price(option_type, S, K, sigma - 1e-5, r, T)) / (2e-5)
            
            if vega == 0:
                return np.nan

            diff = price - market_price
            if abs(diff) < eps:
                return sigma

            sigma -= diff / vega

        except (ZeroDivisionError, FloatingPointError, ValueError):
            return np.nan

    return np.nan  # if convergence fails


r = 0.0  # risk-free rate

for product in voucher_products:
    df = products_df[product]

    # Only keep rows with valid data
    df = df[(df['TTE'] > 0) & (df['underlying_price'] > 0) & (df['mid_price'] > 0)]

    # Convert TTE to years (assuming 1M = 1 trading day, 252 days per year)
    df['T'] = df['TTE'] / (1_000_000 * 252)

    # Compute Black-Scholes prices
    df['call_price'] = df.apply(
        lambda row: bs_price('call', row['underlying_price'], row['K'], 0.3, r, row['T']),
        axis=1
    )

    df['put_price'] = df.apply(
        lambda row: bs_price('put', row['underlying_price'], row['K'], 0.3, r, row['T']),
        axis=1
    )

    # Compute implied volatilities from market price
    df['iv_call'] = df.apply(
        lambda row: implied_vol('call', row['underlying_price'], row['K'], r, row['T'], row['mid_price']),
        axis=1
    )

    df['iv_put'] = df.apply(
        lambda row: implied_vol('put', row['underlying_price'], row['K'], r, row['T'], row['mid_price']),
        axis=1
    )

    # Save the updated DataFrame
    products_df[product] = df

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for product in voucher_products:
    df = products_df[product]
    
    # Filter clean data
    plot_df = df[(df['TTE'] > 0) & df['iv_call'].notna()]
    
    plt.plot(
        plot_df['TTE'],
        plot_df['iv_call'],
        label=product,
        alpha=0.8
    )

plt.xlabel('TTE (Time to Expiry in microseconds)')
plt.ylabel('Implied Volatility (Call)')
plt.title('Implied Volatility vs TTE for Voucher Products')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


 
 



