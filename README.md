# A research-grade Python framework for power-law market-impact modeling and optimized trade execution.

This directory contains a pure Python script to simulate and analyze the temporary market impact function.

## Files

* `impact_modeling.py`: Main script that simulates an MBP-10 order book snapshot, computes slippage for buy orders of various sizes, fits a power-law model, and generates plots.
* `requirements.txt`: List of required Python packages.
* `impact_result.txt`: Output summary of fitted model parameters and console logs.

## Plot Folders

To keep the output organized, plots are saved automatically into two dedicated folders:

* **Graph\_LogLog/**: Contains log–log impact fit plots for each snapshot. Each file is named `impact_loglog_<snapshot_basename>.png`, showing the raw data points and fitted power-law line on log–log axes.
* **Graph\_Residuals/**: Contains residual error plots for each snapshot. Each file is named `residuals_<snapshot_basename>.png`, illustrating the percentage error between observed and predicted impact values.

Use these folders to review and validate the fit quality across all processed snapshots.

