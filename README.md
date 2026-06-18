# Portfolio Backtester

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hedmon1/portfolio-backtester/blob/main/portfolio_backtester.ipynb)

A single-file portfolio backtesting engine that runs in Google Colab with no setup. Edit one inputs section, run all cells, and get a full research report: performance and risk metrics, a benchmark comparison, a per-stock leaderboard, a Monte Carlo forecast, sector analysis, charts, and a written summary with a final grade.

Click the **Open in Colab** badge above to run it in your browser — nothing to install.

## Quick start

1. Click the **Open in Colab** badge.
2. In Colab, choose **Runtime → Run all**.
3. To test your own portfolio, edit the `CUSTOM PORTFOLIO INPUTS` cell and run again.

## What you can configure

| Input | Example |
|---|---|
| Tickers (10–20) | `["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","JPM","V"]` |
| Benchmark | `"SPY"` |
| Date range | `2019-01-01` → `2024-12-31` |
| Initial investment | `100,000` |
| Weighting | Equal / Market-cap / Custom |
| Rebalancing | None / Monthly / Quarterly / Yearly |
| Risk-free rate | `0.04` |
| Transaction costs | `0.0010` (10 bps on turnover) |

## What it produces

- Performance metrics: return, CAGR, volatility, Sharpe, Sortino, Calmar, Treynor, information ratio, alpha, beta, R², up/down capture, rolling return/vol/Sharpe.
- Benchmark comparison vs SPY, including outperformance, correlation, and a beat-the-market verdict.
- Per-stock leaderboard ranked by return, with each holding's contribution to the total.
- Risk analysis: VaR (95%), conditional VaR, largest drawdowns, best/worst days, rolling beta and correlation, covariance and correlation matrices.
- Monte Carlo: 10,000 bootstrapped paths over 1/3/5 years with percentile bands, probability of loss, and probability of beating the benchmark.
- Sector breakdown, 14 charts, and an automated report ending in an Excellent / Good / Average / Poor grade.

## Built with

`pandas` · `numpy` · `yfinance` · `matplotlib` · `seaborn` · `scipy` · `statsmodels` · `tabulate`

## Disclaimer

For educational and research purposes only. Not investment advice. Backtested results do not guarantee future performance. Market data comes from Yahoo Finance via `yfinance` and may contain errors.
