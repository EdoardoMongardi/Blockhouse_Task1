import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mbp10_from_csv(path):
    """
    Load one MBP-10 snapshot from CSV.
    Expects columns:
       bid_px_00…bid_px_09, bid_sz_00…bid_sz_09,
       ask_px_00…ask_px_09, ask_sz_00…ask_sz_09
    """
    df = pd.read_csv(path)

    # read prices and sizes (00 through 09)
    bid_prices = df[[f"bid_px_{i:02d}" for i in range(10)]].iloc[0].values
    bid_sizes  = df[[f"bid_sz_{i:02d}" for i in range(10)]].iloc[0].values
    ask_prices = df[[f"ask_px_{i:02d}" for i in range(10)]].iloc[0].values
    ask_sizes  = df[[f"ask_sz_{i:02d}" for i in range(10)]].iloc[0].values

    # mid‐price is the midpoint of best bid/ask
    mid_price  = (bid_prices[0] + ask_prices[0]) / 2
    return bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price

def impact_for_buy(volume, ask_prices, ask_sizes, mid_price):
    vol = volume
    total_cost = 0.0
    for price, avail in zip(ask_prices, ask_sizes):
        if vol <= avail:
            total_cost += price * vol
            vol = 0
            break
        total_cost += price * avail
        vol -= avail
    if vol > 0:
        total_cost += ask_prices[-1] * vol
    avg_price = total_cost / volume
    return avg_price - mid_price

def fit_power_law(volumes, impacts):
    log_v, log_i = np.log(volumes), np.log(impacts)
    beta, log_alpha = np.polyfit(log_v, log_i, 1)
    return np.exp(log_alpha), beta

def main():
    volumes = np.array([100, 500, 1000, 2000, 5000], dtype=float)
    # recurse into Data/CRWV, Data/FROG, Data/SOUN
    for csv_path in glob.glob("Data/**/*.csv", recursive=True):
        bid_p, bid_s, ask_p, ask_s, mid = load_mbp10_from_csv(csv_path)
        impacts = np.array([impact_for_buy(X, ask_p, ask_s, mid) for X in volumes])

        print(f"\nSnapshot: {csv_path}")
        for X, imp in zip(volumes, impacts):
            print(f"  {int(X):6d} → impact = {imp:.4f}")
        alpha, beta = fit_power_law(volumes, impacts)
        print(f"  Fit: g(X) = {alpha:.6f} * X^{beta:.3f}")

if __name__ == "__main__":
    main()
