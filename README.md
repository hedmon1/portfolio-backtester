# 📊 Institutional-Grade Portfolio Backtester

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hedmon1/portfolio-backtester/blob/main/portfolio_backtester.ipynb)

A single-file, hedge-fund-style portfolio backtesting engine that runs **end-to-end in Google Colab with no setup**. Edit one clearly-labeled inputs section, press *Run all*, and get a full institutional research report: performance analytics, risk metrics, benchmark comparison, a Monte Carlo forecast, sector analysis, 14 professional charts, and an automated written investment report with a final letter grade.

> Click the **Open in Colab** badge above to launch it instantly — no installation required.

---

## 🚀 Quick start

1. Click the **Open in Colab** badge.
2. In Colab, go to **Runtime → Run all**.
3. To test your own portfolio, edit the **`CUSTOM PORTFOLIO INPUTS`** cell and re-run.

That's it. Colab runs on Google's servers, so anyone can run it from a browser with nothing installed.

## 🎛️ What you can configure (one variable each)

| Input | Example |
|---|---|
| Portfolio tickers (10–20) | `["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","JPM","V"]` |
| Benchmark | `"SPY"` |
| Date range | `2019-01-01` → `2024-12-31` |
| Initial investment | `100,000` |
| Weighting | Equal / Market-cap / Custom |
| Rebalancing | None / Monthly / Quarterly / Yearly |
| Risk-free rate | `0.04` |
| Transaction costs | `0.0010` (10 bps on turnover) |

## 📈 What it produces

- **Performance analytics** — Absolute & annualized return, CAGR, volatility, Sharpe, Sortino, Calmar, Treynor, Information Ratio, alpha, beta, R², up/down capture, win rate, monthly stats, rolling return/vol/Sharpe.
- **Benchmark comparison** — Total return, CAGR, volatility, Sharpe, max drawdown vs SPY, outperformance, correlation, and a *"Did the portfolio beat the market?"* verdict.
- **Individual stock leaderboard** — Per-stock return, volatility, Sharpe, drawdown, and contribution to portfolio return, ranked best-to-worst.
- **Risk analysis** — Value at Risk (95%), Conditional VaR, largest drawdown episodes, best/worst days, rolling beta & correlation, covariance & correlation matrices.
- **Monte Carlo simulation** — 10,000 bootstrapped paths over 1/3/5 years with median, 10th/90th percentile, probability of loss, and probability of beating the benchmark.
- **Sector analysis** — Auto-classified sector weights, contribution, and a pie chart of exposure.
- **14 professional charts** and a full **automated investment report** ending in an `EXCELLENT / GOOD / AVERAGE / POOR` grade.

## 🛠️ Built with

`pandas` · `numpy` · `yfinance` · `matplotlib` · `seaborn` · `scipy` · `statsmodels` · `tabulate`

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not investment advice. Past performance and backtested results do not guarantee future returns. Market data is provided by Yahoo Finance via `yfinance` and may contain inaccuracies.
