# ======================================================================================
#  INSTITUTIONAL-GRADE PORTFOLIO BACKTESTING ENGINE
#  --------------------------------------------------------------------------------------
#  A single-file, hedge-fund-style backtester designed to run end-to-end in Google Colab.
#
#  HOW TO USE (Google Colab):
#    1. Open a new Colab notebook (https://colab.research.google.com).
#    2. Paste this entire file into a single cell (or split on the "# %%" markers
#       into multiple cells -- either works).
#    3. Edit ONLY the "CUSTOM PORTFOLIO INPUTS" section below.
#    4. Press Run.  Everything else is automatic.
#
#  The notebook is organized into clearly-labeled, modular sections:
#      0. Package installation
#      1. Imports & global configuration
#      2. CUSTOM PORTFOLIO INPUTS   <-- the only section you need to edit
#      3. Data collection (yfinance, robust to missing data)
#      4. Portfolio construction (weights, rebalancing, transaction costs)
#      5. Performance analytics (Sharpe, Sortino, alpha, beta, capture, etc.)
#      6. Benchmark comparison
#      7. Individual stock leaderboard
#      8. Risk analytics (VaR, CVaR, drawdowns, rolling beta)
#      9. Monte Carlo simulation (10,000+ paths)
#     10. Sector analysis
#     11. Professional charts (14 charts)
#     12. Automated written investment report
#     13. Executive summary + investment grade
#
#  Every function is documented.  Beginners can read top-to-bottom; quants can lift
#  individual functions.  All assumptions are editable from ONE place (Section 2).
# ======================================================================================


# %% ===================================================================================
#  SECTION 0 -- PACKAGE INSTALLATION
#  -------------------------------------------------------------------------------------
#  Colab already ships with most of these, but we install/upgrade to be safe.
#  The "!" prefix is a Colab/IPython shell magic.  If you run this outside Colab as a
#  plain .py file, comment this block out and "pip install" from your terminal instead.
# ======================================================================================
try:
    import google.colab  # noqa: F401  -> only present inside Google Colab
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    # -q = quiet.  yfinance is upgraded because the API changes often.
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "yfinance", "pandas", "numpy", "matplotlib", "seaborn",
         "scipy", "statsmodels", "tabulate"],
        check=False,
    )


# %% ===================================================================================
#  SECTION 1 -- IMPORTS & GLOBAL CONFIGURATION
# ======================================================================================
import warnings
warnings.filterwarnings("ignore")  # keep the report clean of library deprecation noise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf

from scipy import stats
import statsmodels.api as sm
from tabulate import tabulate

# ---- Plot styling: a clean, institutional look ---------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 110,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "font.size": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})

# ---- Constants -----------------------------------------------------------------------
TRADING_DAYS = 252          # standard number of trading days per year
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# ---- pandas frequency aliases (version-proof) ----------------------------------------
#  pandas 2.2+ renamed the resample aliases: 'M'->'ME', 'Q'->'QE', 'A'/'Y'->'YE'.
#  Google Colab currently ships pandas >= 2.2, but we detect the version so this file
#  runs unchanged on older installs too.  (Period codes used by .to_period() differ from
#  resample codes, so they are kept separate.)
_PANDAS_GE_22 = tuple(int(p) for p in pd.__version__.split(".")[:2]) >= (2, 2)
RS_M = "ME" if _PANDAS_GE_22 else "M"     # month-end resample
RS_Q = "QE" if _PANDAS_GE_22 else "Q"     # quarter-end resample
RS_Y = "YE" if _PANDAS_GE_22 else "A"     # year-end resample
P_M, P_Q = "M", "Q"                       # period codes (stable across versions)
P_Y = "Y" if _PANDAS_GE_22 else "A"       # annual period code


# %% ===================================================================================
#  SECTION 2 -- CUSTOM PORTFOLIO INPUTS  <<<<<<<<<<  EDIT THIS SECTION ONLY  >>>>>>>>>>
#  -------------------------------------------------------------------------------------
#  Every assumption in the engine is controlled by exactly one variable below.
# ======================================================================================

# --- 2.1  The portfolio: 10-20 tickers ------------------------------------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
           "META", "AVGO", "TSLA", "JPM", "V"]

# --- 2.2  Benchmark (default = SPY, the S&P 500 ETF) ----------------------------------
BENCHMARK = "SPY"

# --- 2.3  Backtest window -------------------------------------------------------------
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# --- 2.4  Capital ---------------------------------------------------------------------
INITIAL_INVESTMENT = 100_000.0     # dollars

# --- 2.5  Weighting method: "equal", "market_cap", or "custom" ------------------------
WEIGHTING_METHOD = "equal"

#       If WEIGHTING_METHOD == "custom", define weights here (they will be normalized
#       automatically so they do not need to sum to exactly 1.0).  Any ticker omitted
#       gets a weight of 0.  Example:
CUSTOM_WEIGHTS = {
    "AAPL": 0.15, "MSFT": 0.15, "NVDA": 0.15, "AMZN": 0.10, "GOOGL": 0.10,
    "META": 0.10, "AVGO": 0.05, "TSLA": 0.05, "JPM": 0.10, "V": 0.05,
}

# --- 2.6  Rebalancing frequency: "none", "monthly", "quarterly", "yearly" -------------
REBALANCE_FREQUENCY = "quarterly"

# --- 2.7  Risk-free rate (annualized, decimal). e.g. 0.04 = 4% ------------------------
RISK_FREE_RATE = 0.04

# --- 2.8  Transaction cost per trade, as a fraction of traded notional -----------------
#       0.0010 = 10 basis points = 0.10% charged on the dollar value that is bought/sold
#       at each rebalance (and on the initial purchase).
TRANSACTION_COST = 0.0010

# --- 2.9  Monte Carlo settings --------------------------------------------------------
MC_SIMULATIONS = 10_000            # number of simulated paths (>= 10,000 as requested)
MC_HORIZONS_YEARS = [1, 3, 5]      # forecast horizons

# --- 2.10 Misc ------------------------------------------------------------------------
ROLLING_WINDOW = TRADING_DAYS      # 1-year rolling window (252 trading days)


# Convenience: bundle everything into a single config dict so functions stay pure.
CONFIG = dict(
    tickers=[t.upper().strip() for t in TICKERS],
    benchmark=BENCHMARK.upper().strip(),
    start=START_DATE,
    end=END_DATE,
    initial=float(INITIAL_INVESTMENT),
    weighting=WEIGHTING_METHOD.lower().strip(),
    custom_weights={k.upper(): float(v) for k, v in CUSTOM_WEIGHTS.items()},
    rebalance=REBALANCE_FREQUENCY.lower().strip(),
    rf=float(RISK_FREE_RATE),
    cost=float(TRANSACTION_COST),
    mc_sims=int(MC_SIMULATIONS),
    mc_horizons=list(MC_HORIZONS_YEARS),
    roll=int(ROLLING_WINDOW),
)


# %% ===================================================================================
#  SECTION 3 -- DATA COLLECTION
#  -------------------------------------------------------------------------------------
#  Downloads adjusted-close prices and volume from Yahoo Finance, plus (optionally)
#  market caps and sectors.  Designed to fail gracefully: tickers with too much missing
#  data are dropped (with a warning) rather than crashing the whole run.
# ======================================================================================

def download_prices(tickers, start, end, max_missing_pct=0.10):
    """Download adjusted-close prices and volume for a list of tickers.

    Returns
    -------
    prices : DataFrame  (adjusted close, one column per surviving ticker)
    volume : DataFrame  (daily volume, same columns)
    dropped: list       (tickers removed because of excessive missing data)

    Notes
    -----
    * auto_adjust=True makes the 'Close' column already split/dividend adjusted,
      which is exactly the "Adjusted Close" we want for total-return backtesting.
    * Any ticker missing more than `max_missing_pct` of its observations is dropped.
    * Remaining gaps are forward-filled then back-filled so returns are well-defined.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="column")

    # yfinance returns a different shape for one vs. many tickers -> normalize both.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        # single ticker -> raw is a flat frame with Open/High/Low/Close/Volume
        prices = raw[["Close"]].copy()
        prices.columns = tickers
        volume = raw[["Volume"]].copy()
        volume.columns = tickers

    # Ensure DataFrame (not Series) and consistent column ordering
    prices = prices.dropna(how="all")
    volume = volume.reindex(columns=prices.columns)

    # --- Drop tickers with too much missing data ------------------------------------
    dropped = []
    keep = []
    for col in prices.columns:
        missing = prices[col].isna().mean()
        if missing > max_missing_pct:
            dropped.append(col)
            print(f"  [!] Dropping {col}: {missing:.0%} of data missing.")
        else:
            keep.append(col)
    prices = prices[keep]
    volume = volume.reindex(columns=keep)

    # --- Fill remaining small gaps --------------------------------------------------
    prices = prices.ffill().bfill()
    volume = volume.ffill().fillna(0)

    return prices, volume, dropped


def fetch_fundamentals(tickers):
    """Fetch market caps and sectors per ticker, failing gracefully on each one.

    Returns a DataFrame indexed by ticker with columns ['market_cap', 'sector'].
    Missing values are filled (market cap -> NaN, sector -> 'Unknown').
    """
    rows = {}
    for t in tickers:
        mcap, sector = np.nan, "Unknown"
        try:
            info = yf.Ticker(t).info or {}
            mcap = info.get("marketCap", np.nan)
            sector = info.get("sector", None) or "Unknown"
        except Exception as e:
            print(f"  [!] Could not fetch fundamentals for {t}: {e}")
        rows[t] = {"market_cap": mcap, "sector": sector}
    return pd.DataFrame(rows).T


# %% ===================================================================================
#  SECTION 4 -- PORTFOLIO CONSTRUCTION
#  -------------------------------------------------------------------------------------
#  Builds target weights, then simulates the portfolio day-by-day, applying periodic
#  rebalancing and transaction costs, and tracking total dollar value over time.
# ======================================================================================

def build_weights(tickers, method, custom_weights, market_caps):
    """Return a normalized weight Series (sums to 1.0) for the chosen method."""
    if method == "equal":
        w = pd.Series(1.0, index=tickers)

    elif method == "market_cap":
        caps = market_caps.reindex(tickers).astype(float)
        if caps.isna().all() or caps.fillna(0).sum() == 0:
            print("  [!] Market caps unavailable -> falling back to equal weight.")
            w = pd.Series(1.0, index=tickers)
        else:
            # tickers with unknown caps get the median cap so they are not zeroed out
            caps = caps.fillna(caps.median())
            w = caps

    elif method == "custom":
        w = pd.Series({t: custom_weights.get(t, 0.0) for t in tickers})
        if w.sum() == 0:
            print("  [!] Custom weights all zero -> falling back to equal weight.")
            w = pd.Series(1.0, index=tickers)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    w = w.clip(lower=0)              # no shorting in this engine
    return w / w.sum()              # normalize to 1.0


def get_rebalance_dates(dates, frequency):
    """Return the set of trading days on which we rebalance.

    We pick the LAST available trading day inside each calendar period so that the
    rebalance always lands on a real market day.
    """
    if frequency == "none":
        return set()
    rule = {"monthly": P_M, "quarterly": P_Q, "yearly": P_Y}[frequency]
    periods = dates.to_period(rule)
    last_per_period = pd.Series(dates, index=periods).groupby(level=0).last()
    return set(pd.to_datetime(last_per_period.values))


def simulate_portfolio(prices, weights, initial, rebalance, cost):
    """Day-by-day simulation of the portfolio's dollar value.

    Mechanics
    ---------
    * Day 0: buy the target weights (initial purchase pays transaction cost).
    * Each subsequent day: every holding grows by its asset's daily return; weights
      drift naturally.
    * On a rebalance date: trade back to target weights and pay `cost` on the dollar
      turnover (sum of |buys| + |sells| / 2, i.e. the notional that actually trades).

    Returns
    -------
    value : Series   portfolio dollar value per day
    weights_history : DataFrame  end-of-day weights per asset (for drift inspection)
    """
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    assets = prices.columns
    target = weights.reindex(assets).fillna(0.0)
    rebalance_dates = get_rebalance_dates(dates, rebalance)

    # --- Day 0: initial purchase ----------------------------------------------------
    cash = initial * (1.0 - cost)             # pay cost on the full initial notional
    holdings = target * cash                  # dollars in each asset
    values = []
    w_hist = []

    for i, date in enumerate(dates):
        if i > 0:
            holdings = holdings * (1.0 + rets.iloc[i])    # mark-to-market growth
        total = holdings.sum()

        # Rebalance at period boundaries (never on day 0 -- already at target)
        if i > 0 and date in rebalance_dates:
            desired = target * total
            turnover = (desired - holdings).abs().sum() / 2.0   # one-way traded notional
            total -= turnover * cost                            # subtract trading cost
            holdings = target * total                           # reset to target

        values.append(total)
        w_hist.append((holdings / total).values if total > 0 else target.values)

    value = pd.Series(values, index=dates, name="Portfolio")
    weights_history = pd.DataFrame(w_hist, index=dates, columns=assets)
    return value, weights_history


def benchmark_value_series(bench_prices, initial):
    """Grow `initial` dollars along the benchmark's adjusted-close path (buy & hold)."""
    bench = bench_prices.iloc[:, 0] if isinstance(bench_prices, pd.DataFrame) else bench_prices
    growth = bench / bench.iloc[0]
    return (growth * initial).rename("Benchmark")


# %% ===================================================================================
#  SECTION 5 -- PERFORMANCE ANALYTICS
#  -------------------------------------------------------------------------------------
#  All the classic institutional metrics, each as a small, testable function.
# ======================================================================================

def to_returns(value):
    """Daily simple returns from a value/price series."""
    return value.pct_change().dropna()


def cagr(value, periods_per_year=TRADING_DAYS):
    """Compound Annual Growth Rate from a value series."""
    n_years = len(value) / periods_per_year
    if n_years <= 0:
        return np.nan
    return (value.iloc[-1] / value.iloc[0]) ** (1 / n_years) - 1


def annualized_return(returns, periods_per_year=TRADING_DAYS):
    """Arithmetic annualized return (mean daily return * 252)."""
    return returns.mean() * periods_per_year


def annualized_vol(returns, periods_per_year=TRADING_DAYS):
    """Annualized volatility (std of daily returns * sqrt(252))."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, rf=0.0, periods_per_year=TRADING_DAYS):
    """Annualized Sharpe ratio using a daily risk-free rate."""
    rf_daily = rf / periods_per_year
    excess = returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns, rf=0.0, periods_per_year=TRADING_DAYS):
    """Annualized Sortino ratio (penalizes only downside deviation)."""
    rf_daily = rf / periods_per_year
    excess = returns - rf_daily
    downside = excess[excess < 0]
    dd = downside.std()
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (excess.mean() / dd) * np.sqrt(periods_per_year)


def drawdown_series(value):
    """Running drawdown (current value / running peak - 1)."""
    running_max = value.cummax()
    return value / running_max - 1.0


def max_drawdown(value):
    """The single worst peak-to-trough decline (a negative number)."""
    return drawdown_series(value).min()


def calmar_ratio(value, periods_per_year=TRADING_DAYS):
    """CAGR divided by the absolute value of the maximum drawdown."""
    mdd = abs(max_drawdown(value))
    if mdd == 0:
        return np.nan
    return cagr(value, periods_per_year) / mdd


def beta_alpha(returns, bench_returns, rf=0.0, periods_per_year=TRADING_DAYS):
    """CAPM beta, annualized Jensen alpha and R-squared via OLS regression.

    Regresses portfolio EXCESS returns on benchmark EXCESS returns:
        (r_p - rf) = alpha + beta * (r_b - rf) + e
    """
    rf_daily = rf / periods_per_year
    df = pd.concat([returns, bench_returns], axis=1).dropna()
    df.columns = ["p", "b"]
    y = df["p"] - rf_daily
    X = sm.add_constant(df["b"] - rf_daily)
    model = sm.OLS(y, X).fit()
    alpha_daily = model.params["const"]
    beta = model.params["b"]
    r_squared = model.rsquared
    alpha_annual = (1 + alpha_daily) ** periods_per_year - 1   # compounded annual alpha
    return beta, alpha_annual, r_squared


def treynor_ratio(returns, bench_returns, rf=0.0, periods_per_year=TRADING_DAYS):
    """Excess annualized return per unit of systematic risk (beta)."""
    beta, _, _ = beta_alpha(returns, bench_returns, rf, periods_per_year)
    if beta == 0 or np.isnan(beta):
        return np.nan
    return (annualized_return(returns, periods_per_year) - rf) / beta


def tracking_error(returns, bench_returns, periods_per_year=TRADING_DAYS):
    """Annualized standard deviation of the active (portfolio - benchmark) return."""
    active = (returns - bench_returns).dropna()
    return active.std() * np.sqrt(periods_per_year)


def information_ratio(returns, bench_returns, periods_per_year=TRADING_DAYS):
    """Active return divided by tracking error (skill per unit of active risk)."""
    active = (returns - bench_returns).dropna()
    if active.std() == 0:
        return np.nan
    return (active.mean() / active.std()) * np.sqrt(periods_per_year)


def capture_ratios(returns, bench_returns):
    """Upside / downside capture ratios, computed on monthly returns.

    Upside capture = geo-mean portfolio return in up-benchmark months
                     / geo-mean benchmark return in those months  (x100).
    >100 upside and <100 downside is the desirable "convex" profile.
    """
    p_m = (1 + returns).resample(RS_M).prod() - 1
    b_m = (1 + bench_returns).resample(RS_M).prod() - 1
    df = pd.concat([p_m, b_m], axis=1).dropna()
    df.columns = ["p", "b"]

    def geo(x):
        return (np.prod(1 + x)) ** (1 / len(x)) - 1 if len(x) else np.nan

    up, down = df[df["b"] > 0], df[df["b"] < 0]
    up_cap = (geo(up["p"]) / geo(up["b"]) * 100) if len(up) and geo(up["b"]) != 0 else np.nan
    dn_cap = (geo(down["p"]) / geo(down["b"]) * 100) if len(down) and geo(down["b"]) != 0 else np.nan
    return up_cap, dn_cap


def monthly_stats(value):
    """Dictionary of month-level statistics from a daily value series."""
    m = value.resample(RS_M).last().pct_change().dropna()
    return {
        "monthly_returns": m,
        "best_month": m.max(),
        "worst_month": m.min(),
        "positive_month_pct": (m > 0).mean(),
        "avg_monthly_return": m.mean(),
        "win_rate_daily": (to_returns(value) > 0).mean(),
    }


def value_at_risk(returns, level=0.95):
    """Historical Value-at-Risk: the loss not exceeded with `level` confidence (positive number)."""
    return -np.percentile(returns, (1 - level) * 100)


def conditional_var(returns, level=0.95):
    """Conditional VaR / Expected Shortfall: average loss in the worst (1-level) tail."""
    var = -value_at_risk(returns, level)   # threshold (negative)
    tail = returns[returns <= var]
    return -tail.mean() if len(tail) else np.nan


def compute_all_metrics(value, bench_value, rf, label="Portfolio"):
    """Assemble the full institutional metric set for one value series.

    Returns an ordered dict ready to drop into a DataFrame.
    """
    r = to_returns(value)
    br = to_returns(bench_value)
    beta, alpha, r2 = beta_alpha(r, br, rf)
    up_cap, dn_cap = capture_ratios(r, br)
    ms = monthly_stats(value)

    return {
        "Absolute Return %":      (value.iloc[-1] / value.iloc[0] - 1) * 100,
        "Annualized Return %":    annualized_return(r) * 100,
        "CAGR %":                 cagr(value) * 100,
        "Annualized Volatility %": annualized_vol(r) * 100,
        "Sharpe Ratio":           sharpe_ratio(r, rf),
        "Sortino Ratio":          sortino_ratio(r, rf),
        "Max Drawdown %":         max_drawdown(value) * 100,
        "Calmar Ratio":           calmar_ratio(value),
        "Beta vs SPY":            beta,
        "Alpha (annual) %":       alpha * 100,
        "Treynor Ratio":          treynor_ratio(r, br, rf),
        "Information Ratio":      information_ratio(r, br),
        "Tracking Error %":       tracking_error(r, br) * 100,
        "Upside Capture %":       up_cap,
        "Downside Capture %":     dn_cap,
        "Win Rate (daily) %":     ms["win_rate_daily"] * 100,
        "Best Month %":           ms["best_month"] * 100,
        "Worst Month %":          ms["worst_month"] * 100,
        "Positive Months %":      ms["positive_month_pct"] * 100,
        "Avg Monthly Return %":   ms["avg_monthly_return"] * 100,
        "VaR 95% (daily) %":      value_at_risk(r) * 100,
        "CVaR 95% (daily) %":     conditional_var(r) * 100,
        "R-squared vs SPY":       r2,
    }


# %% ===================================================================================
#  SECTION 6 -- BENCHMARK COMPARISON
# ======================================================================================

def benchmark_comparison(port_value, bench_value, rf):
    """Side-by-side comparison table + a plain-English 'beat the market?' verdict."""
    pr, brr = to_returns(port_value), to_returns(bench_value)

    rows = {
        "Total Return %":   [(port_value.iloc[-1] / port_value.iloc[0] - 1) * 100,
                             (bench_value.iloc[-1] / bench_value.iloc[0] - 1) * 100],
        "CAGR %":           [cagr(port_value) * 100, cagr(bench_value) * 100],
        "Volatility %":     [annualized_vol(pr) * 100, annualized_vol(brr) * 100],
        "Sharpe Ratio":     [sharpe_ratio(pr, rf), sharpe_ratio(brr, rf)],
        "Max Drawdown %":   [max_drawdown(port_value) * 100, max_drawdown(bench_value) * 100],
    }
    table = pd.DataFrame(rows, index=["Portfolio", "Benchmark (SPY)"]).T
    table["Difference"] = table["Portfolio"] - table["Benchmark (SPY)"]

    # Extra relationship stats
    beta, alpha, r2 = beta_alpha(pr, brr, rf)
    corr = pr.corr(brr)
    outperf = (port_value.iloc[-1] / port_value.iloc[0]) - (bench_value.iloc[-1] / bench_value.iloc[0])

    extras = {
        "Outperformance % (total)": outperf * 100,
        "Annual Alpha %":           alpha * 100,
        "Correlation to SPY":       corr,
        "R-squared":                r2,
    }

    beat = port_value.iloc[-1] > bench_value.iloc[-1]
    verdict = ("YES -- the portfolio BEAT the market."
               if beat else
               "NO -- the portfolio TRAILED the market.")
    return table, extras, beat, verdict


# %% ===================================================================================
#  SECTION 7 -- INDIVIDUAL STOCK ANALYSIS / LEADERBOARD
# ======================================================================================

def stock_leaderboard(prices, weights, port_value, rf):
    """Per-stock analytics ranked best-to-worst, plus contribution to portfolio return.

    Contribution is approximated as weight * stock total return (a first-order,
    Brinson-style attribution that sums close to the portfolio's total return).
    """
    port_rets = to_returns(port_value)
    rows = {}
    for t in prices.columns:
        v = prices[t]
        r = to_returns(v)
        rows[t] = {
            "Weight %":            weights.get(t, 0.0) * 100,
            "Total Return %":      (v.iloc[-1] / v.iloc[0] - 1) * 100,
            "Annualized Return %": annualized_return(r) * 100,
            "Volatility %":        annualized_vol(r) * 100,
            "Sharpe":              sharpe_ratio(r, rf),
            "Max Drawdown %":      max_drawdown(v) * 100,
            "Contribution %":      weights.get(t, 0.0) * (v.iloc[-1] / v.iloc[0] - 1) * 100,
            "Corr w/ Portfolio":   r.corr(port_rets),
        }
    lb = pd.DataFrame(rows).T
    lb = lb.sort_values("Total Return %", ascending=False)
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    return lb


# %% ===================================================================================
#  SECTION 8 -- RISK ANALYSIS
# ======================================================================================

def largest_drawdowns(value, top_n=5):
    """Identify the `top_n` deepest distinct drawdown episodes with their dates."""
    dd = drawdown_series(value)
    episodes = []
    in_dd = False
    peak_date = value.index[0]
    for date, d in dd.items():
        if d < 0 and not in_dd:
            in_dd, peak_date = True, date
        elif d == 0 and in_dd:
            in_dd = False
            window = dd.loc[peak_date:date]
            trough_date = window.idxmin()
            episodes.append({
                "Start": peak_date.date(),
                "Trough": trough_date.date(),
                "End": date.date(),
                "Depth %": window.min() * 100,
                "Length (days)": (date - peak_date).days,
            })
    # capture an ongoing drawdown at the end of the sample
    if in_dd:
        window = dd.loc[peak_date:]
        episodes.append({
            "Start": peak_date.date(),
            "Trough": window.idxmin().date(),
            "End": "ongoing",
            "Depth %": window.min() * 100,
            "Length (days)": (value.index[-1] - peak_date).days,
        })
    out = pd.DataFrame(episodes).sort_values("Depth %").head(top_n)
    return out.reset_index(drop=True)


def best_worst_days(value, n=5):
    """The n best and n worst single-day returns."""
    r = to_returns(value) * 100
    worst = r.nsmallest(n).rename("Return %").to_frame()
    best = r.nlargest(n).rename("Return %").to_frame()
    worst.index = worst.index.date
    best.index = best.index.date
    return best, worst


def rolling_beta(returns, bench_returns, window=TRADING_DAYS):
    """Rolling CAPM beta = rolling cov(port, bench) / rolling var(bench)."""
    df = pd.concat([returns, bench_returns], axis=1).dropna()
    df.columns = ["p", "b"]
    cov = df["p"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    return (cov / var).dropna()


def rolling_correlation(returns, bench_returns, window=TRADING_DAYS):
    """Rolling correlation of portfolio vs benchmark daily returns."""
    df = pd.concat([returns, bench_returns], axis=1).dropna()
    df.columns = ["p", "b"]
    return df["p"].rolling(window).corr(df["b"]).dropna()


# %% ===================================================================================
#  SECTION 9 -- MONTE CARLO SIMULATION
#  -------------------------------------------------------------------------------------
#  Forecasts future portfolio value by BOOTSTRAPPING historical daily returns (sampling
#  real days with replacement).  Bootstrapping preserves the fat tails and skew of the
#  actual return distribution far better than assuming a clean normal distribution.
# ======================================================================================

def monte_carlo(port_value, bench_value, initial, horizons_years, n_sims,
                periods_per_year=TRADING_DAYS, seed=42):
    """Run Monte Carlo forecasts for each horizon.

    Returns a dict keyed by horizon (years) -> {
        'paths_sample', 'final_values', 'median', 'p10', 'p90',
        'prob_loss', 'prob_beat_benchmark'
    }
    The benchmark hurdle is the benchmark's own historical CAGR compounded over the
    horizon (a fair "buy SPY instead" comparison).
    """
    rng = np.random.default_rng(seed)
    daily = to_returns(port_value).values
    bench_cagr = cagr(bench_value)

    results = {}
    for years in horizons_years:
        days = int(years * periods_per_year)
        # Sample (days x n_sims) daily returns from history, with replacement
        sampled = rng.choice(daily, size=(days, n_sims), replace=True)
        growth = np.cumprod(1 + sampled, axis=0)          # cumulative growth per path
        paths = initial * growth                          # dollar paths
        finals = paths[-1, :]

        bench_hurdle = initial * (1 + bench_cagr) ** years

        results[years] = {
            "paths_sample": paths[:, :200],               # keep 200 paths for plotting
            "final_values": finals,
            "median": np.median(finals),
            "p10": np.percentile(finals, 10),
            "p90": np.percentile(finals, 90),
            "prob_loss": float(np.mean(finals < initial)),
            "prob_beat_benchmark": float(np.mean(finals > bench_hurdle)),
            "bench_hurdle": bench_hurdle,
        }
    return results


# %% ===================================================================================
#  SECTION 10 -- SECTOR ANALYSIS
# ======================================================================================

def sector_analysis(prices, weights, fundamentals):
    """Aggregate weights and performance by GICS sector.

    Returns a DataFrame indexed by sector with weight, weighted return contribution,
    and average stock return inside the sector.
    """
    sectors = fundamentals["sector"].reindex(prices.columns).fillna("Unknown")
    rows = {}
    for t in prices.columns:
        sec = sectors[t]
        tot_ret = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        w = weights.get(t, 0.0)
        rows.setdefault(sec, {"Weight": 0.0, "Contribution": 0.0, "rets": []})
        rows[sec]["Weight"] += w
        rows[sec]["Contribution"] += w * tot_ret
        rows[sec]["rets"].append(tot_ret)

    data = {
        sec: {
            "Weight %": v["Weight"] * 100,
            "Contribution %": v["Contribution"] * 100,
            "Avg Stock Return %": np.mean(v["rets"]) * 100,
            "# Stocks": len(v["rets"]),
        }
        for sec, v in rows.items()
    }
    df = pd.DataFrame(data).T.sort_values("Weight %", ascending=False)
    return df


# %% ===================================================================================
#  SECTION 11 -- PROFESSIONAL CHARTS (14)
# ======================================================================================

def _fmt_dollar(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))


def make_all_charts(prices, port_value, bench_value, weights, mc_results,
                    sector_df, leaderboard, rf, roll):
    """Render all 14 institutional charts.  Each is its own figure for clean Colab output."""
    pr, brr = to_returns(port_value), to_returns(bench_value)

    # --- 1. Portfolio value vs SPY ---------------------------------------------------
    fig, ax = plt.subplots()
    port_value.plot(ax=ax, label="Portfolio", lw=2, color="#1f4e79")
    bench_value.plot(ax=ax, label="SPY", lw=2, color="#c0392b", alpha=0.85)
    ax.set_title("1. Portfolio Value vs. SPY"); ax.set_ylabel("Value")
    _fmt_dollar(ax); ax.legend(); plt.tight_layout(); plt.show()

    # --- 2. Growth of $10,000 --------------------------------------------------------
    fig, ax = plt.subplots()
    (port_value / port_value.iloc[0] * 10_000).plot(ax=ax, label="Portfolio", lw=2, color="#1f4e79")
    (bench_value / bench_value.iloc[0] * 10_000).plot(ax=ax, label="SPY", lw=2, color="#c0392b", alpha=0.85)
    ax.axhline(10_000, color="gray", ls="--", lw=1)
    ax.set_title("2. Growth of $10,000"); ax.set_ylabel("Value")
    _fmt_dollar(ax); ax.legend(); plt.tight_layout(); plt.show()

    # --- 3. Drawdown chart -----------------------------------------------------------
    fig, ax = plt.subplots()
    dd = drawdown_series(port_value) * 100
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4)
    dd.plot(ax=ax, color="#c0392b", lw=1)
    ax.set_title("3. Portfolio Drawdown"); ax.set_ylabel("Drawdown %")
    plt.tight_layout(); plt.show()

    # --- 4. Rolling Sharpe ratio -----------------------------------------------------
    fig, ax = plt.subplots()
    rs = (pr.rolling(roll).mean() - rf / TRADING_DAYS) / pr.rolling(roll).std() * np.sqrt(TRADING_DAYS)
    rs.plot(ax=ax, color="#1f4e79", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=1); ax.axhline(1, color="green", ls=":", lw=1)
    ax.set_title(f"4. Rolling {roll}-Day Sharpe Ratio"); ax.set_ylabel("Sharpe")
    plt.tight_layout(); plt.show()

    # --- 5. Rolling volatility -------------------------------------------------------
    fig, ax = plt.subplots()
    (pr.rolling(roll).std() * np.sqrt(TRADING_DAYS) * 100).plot(ax=ax, color="#8e44ad", lw=1.5, label="Portfolio")
    (brr.rolling(roll).std() * np.sqrt(TRADING_DAYS) * 100).plot(ax=ax, color="#c0392b", lw=1.2, alpha=0.7, label="SPY")
    ax.set_title(f"5. Rolling {roll}-Day Annualized Volatility"); ax.set_ylabel("Volatility %")
    ax.legend(); plt.tight_layout(); plt.show()

    # --- 6. Monthly returns heatmap --------------------------------------------------
    m = port_value.resample(RS_M).last().pct_change().dropna() * 100
    heat = m.to_frame("ret")
    heat["Year"] = heat.index.year
    heat["Month"] = heat.index.strftime("%b")
    pivot = heat.pivot_table(index="Year", columns="Month", values="ret")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[mo for mo in month_order if mo in pivot.columns])
    fig, ax = plt.subplots(figsize=(12, max(3, 0.6 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, cbar_kws={"label": "Return %"}, ax=ax)
    ax.set_title("6. Monthly Returns Heatmap (%)"); plt.tight_layout(); plt.show()

    # --- 7. Correlation matrix heatmap -----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = prices.pct_change().dropna().corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title("7. Stock Correlation Matrix"); plt.tight_layout(); plt.show()

    # --- 8. Sector allocation pie ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    sw = sector_df["Weight %"]
    ax.pie(sw.values, labels=sw.index, autopct="%1.1f%%", startangle=90,
           colors=sns.color_palette("tab20", len(sw)))
    ax.set_title("8. Sector Allocation"); plt.tight_layout(); plt.show()

    # --- 9. Monte Carlo fan chart (longest horizon) ----------------------------------
    longest = max(mc_results.keys())
    mc = mc_results[longest]
    fig, ax = plt.subplots()
    sample = mc["paths_sample"]
    x = np.arange(sample.shape[0])
    ax.plot(x, sample, color="#1f4e79", alpha=0.05)
    ax.plot(x, np.median(sample, axis=1), color="black", lw=2, label="Median path")
    ax.axhline(mc["p10"], color="#c0392b", ls="--", lw=1, label="10th pct (final)")
    ax.axhline(mc["p90"], color="green", ls="--", lw=1, label="90th pct (final)")
    ax.set_title(f"9. Monte Carlo Fan Chart -- {longest}-Year Forecast ({len(mc['final_values']):,} sims)")
    ax.set_xlabel("Trading days ahead"); ax.set_ylabel("Value"); _fmt_dollar(ax)
    ax.legend(); plt.tight_layout(); plt.show()

    # --- 10. Histogram of daily returns ----------------------------------------------
    fig, ax = plt.subplots()
    ax.hist(pr * 100, bins=80, color="#1f4e79", alpha=0.7, edgecolor="white")
    ax.axvline((pr * 100).mean(), color="green", ls="--", lw=1.5, label="Mean")
    ax.axvline(-value_at_risk(pr) * 100, color="#c0392b", ls="--", lw=1.5, label="VaR 95%")
    ax.set_title("10. Distribution of Daily Returns"); ax.set_xlabel("Daily return %")
    ax.legend(); plt.tight_layout(); plt.show()

    # --- 11. Distribution of rolling 1-year returns ----------------------------------
    fig, ax = plt.subplots()
    roll_ret = (port_value / port_value.shift(roll) - 1).dropna() * 100
    ax.hist(roll_ret, bins=50, color="#8e44ad", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="#c0392b", ls="--", lw=1.5)
    ax.set_title(f"11. Distribution of Rolling {roll}-Day Returns"); ax.set_xlabel("Return %")
    plt.tight_layout(); plt.show()

    # --- 12. Top contributors bar chart ----------------------------------------------
    fig, ax = plt.subplots()
    contrib = leaderboard["Contribution %"].sort_values()
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in contrib.values]
    ax.barh(contrib.index, contrib.values, color=colors)
    ax.set_title("12. Contribution to Portfolio Return by Stock"); ax.set_xlabel("Contribution %")
    plt.tight_layout(); plt.show()

    # --- 13. Cumulative return: portfolio vs benchmark -------------------------------
    fig, ax = plt.subplots()
    ((port_value / port_value.iloc[0] - 1) * 100).plot(ax=ax, label="Portfolio", lw=2, color="#1f4e79")
    ((bench_value / bench_value.iloc[0] - 1) * 100).plot(ax=ax, label="SPY", lw=2, color="#c0392b", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("13. Cumulative Return: Portfolio vs SPY"); ax.set_ylabel("Cumulative return %")
    ax.legend(); plt.tight_layout(); plt.show()

    # --- 14. Annual returns comparison -----------------------------------------------
    fig, ax = plt.subplots()
    p_ann = port_value.resample(RS_Y).last().pct_change().dropna() * 100
    b_ann = bench_value.resample(RS_Y).last().pct_change().dropna() * 100
    ann = pd.DataFrame({"Portfolio": p_ann, "SPY": b_ann})
    ann.index = ann.index.year
    ann.plot(kind="bar", ax=ax, color=["#1f4e79", "#c0392b"])
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("14. Annual Returns: Portfolio vs SPY"); ax.set_ylabel("Return %")
    ax.set_xlabel("Year"); plt.tight_layout(); plt.show()


# %% ===================================================================================
#  SECTION 12 -- AUTOMATED INVESTMENT REPORT (written narrative)
# ======================================================================================

def grade_portfolio(metrics, beat_market, outperf_pct):
    """Translate the quantitative results into a single investment grade + score.

    Simple, transparent scoring rubric (max 6 points):
        +1 Sharpe > 1.0     +1 Sharpe > 1.5
        +1 beat the market  +1 outperformance > 10%
        +1 max drawdown shallower than -30%
        +1 positive annual alpha
    """
    score = 0
    score += metrics["Sharpe Ratio"] > 1.0
    score += metrics["Sharpe Ratio"] > 1.5
    score += bool(beat_market)
    score += outperf_pct > 10
    score += metrics["Max Drawdown %"] > -30
    score += metrics["Alpha (annual) %"] > 0

    if score >= 5:
        return "EXCELLENT", score
    elif score >= 4:
        return "GOOD", score
    elif score >= 2:
        return "AVERAGE", score
    return "POOR", score


def generate_report(metrics, bench_extras, beat, verdict, leaderboard,
                    sector_df, mc_results, dd_table, grade):
    """Produce a clean, sectioned, hedge-fund-style written report (returns a string)."""
    best = leaderboard.head(3)
    worst = leaderboard.tail(3)

    strengths, weaknesses = [], []
    if metrics["Sharpe Ratio"] > 1:
        strengths.append(f"Strong risk-adjusted returns (Sharpe {metrics['Sharpe Ratio']:.2f}).")
    else:
        weaknesses.append(f"Modest risk-adjusted returns (Sharpe {metrics['Sharpe Ratio']:.2f}).")
    if metrics["Alpha (annual) %"] > 0:
        strengths.append(f"Positive annual alpha of {metrics['Alpha (annual) %']:.2f}% vs SPY.")
    else:
        weaknesses.append(f"Negative annual alpha of {metrics['Alpha (annual) %']:.2f}% vs SPY.")
    if metrics["Max Drawdown %"] > -30:
        strengths.append(f"Contained drawdowns (worst {metrics['Max Drawdown %']:.1f}%).")
    else:
        weaknesses.append(f"Deep drawdown risk (worst {metrics['Max Drawdown %']:.1f}%).")
    if metrics["Sortino Ratio"] > metrics["Sharpe Ratio"]:
        strengths.append("Downside-favorable return profile (Sortino exceeds Sharpe).")
    if metrics["Annualized Volatility %"] > 25:
        weaknesses.append(f"Elevated volatility ({metrics['Annualized Volatility %']:.1f}% annualized).")

    longest = max(mc_results.keys())
    mc = mc_results[longest]

    L = []
    add = L.append
    add("=" * 86)
    add("                    AUTOMATED INVESTMENT REPORT")
    add("=" * 86)

    add("\n--- PORTFOLIO SUMMARY ---------------------------------------------------------")
    add(f"Absolute return:        {metrics['Absolute Return %']:.2f}%")
    add(f"CAGR:                   {metrics['CAGR %']:.2f}%")
    add(f"Annualized volatility:  {metrics['Annualized Volatility %']:.2f}%")
    add(f"Sharpe / Sortino:       {metrics['Sharpe Ratio']:.2f} / {metrics['Sortino Ratio']:.2f}")
    add(f"Max drawdown:           {metrics['Max Drawdown %']:.2f}%")
    add(f"Beta / Alpha:           {metrics['Beta vs SPY']:.2f} / {metrics['Alpha (annual) %']:.2f}%")

    add("\n--- STRENGTHS -----------------------------------------------------------------")
    for s in strengths or ["No standout strengths identified."]:
        add(f"  + {s}")

    add("\n--- WEAKNESSES ----------------------------------------------------------------")
    for w in weaknesses or ["No material weaknesses identified."]:
        add(f"  - {w}")

    add("\n--- RISK ANALYSIS -------------------------------------------------------------")
    add(f"Daily VaR (95%):        {metrics['VaR 95% (daily) %']:.2f}%  (expected worst day in 20)")
    add(f"Daily CVaR (95%):       {metrics['CVaR 95% (daily) %']:.2f}%  (avg loss in the tail)")
    add(f"Downside capture:       {metrics['Downside Capture %']:.1f}%  (vs 100% = matches SPY down-moves)")
    add(f"Upside capture:         {metrics['Upside Capture %']:.1f}%")
    if len(dd_table):
        d = dd_table.iloc[0]
        add(f"Deepest drawdown:       {d['Depth %']:.1f}% from {d['Start']} to {d['End']}")

    add("\n--- BEST PERFORMING STOCKS ----------------------------------------------------")
    for t, row in best.iterrows():
        add(f"  {int(row['Rank'])}. {t:<6} total return {row['Total Return %']:7.1f}%   "
            f"contribution {row['Contribution %']:6.2f}%")

    add("\n--- WORST PERFORMING STOCKS ---------------------------------------------------")
    for t, row in worst.iloc[::-1].iterrows():
        add(f"  {int(row['Rank'])}. {t:<6} total return {row['Total Return %']:7.1f}%   "
            f"contribution {row['Contribution %']:6.2f}%")

    add("\n--- OUTPERFORMANCE vs BENCHMARK -----------------------------------------------")
    add(f"Total outperformance:   {bench_extras['Outperformance % (total)']:.2f}%")
    add(f"Annual alpha:           {bench_extras['Annual Alpha %']:.2f}%")
    add(f"Correlation to SPY:     {bench_extras['Correlation to SPY']:.2f}   "
        f"(R-squared {bench_extras['R-squared']:.2f})")
    add(f"VERDICT:                {verdict}")

    add("\n--- FORWARD-LOOKING (MONTE CARLO) ---------------------------------------------")
    add(f"{longest}-year median projection:  ${mc['median']:,.0f}")
    add(f"  10th-90th pct range:      ${mc['p10']:,.0f}  to  ${mc['p90']:,.0f}")
    add(f"  Probability of loss:      {mc['prob_loss']*100:.1f}%")
    add(f"  Prob. of beating SPY:     {mc['prob_beat_benchmark']*100:.1f}%")

    add("\n--- RISK-ADJUSTED PERFORMANCE -------------------------------------------------")
    add(f"Sharpe {metrics['Sharpe Ratio']:.2f} | Sortino {metrics['Sortino Ratio']:.2f} | "
        f"Calmar {metrics['Calmar Ratio']:.2f} | Treynor {metrics['Treynor Ratio']:.3f} | "
        f"Info Ratio {metrics['Information Ratio']:.2f}")

    add("\n--- KEY TAKEAWAYS -------------------------------------------------------------")
    add(f"  * The portfolio {'BEAT' if beat else 'TRAILED'} SPY over the test window.")
    add(f"  * Risk-adjusted quality grade: {grade[0]} (score {grade[1]}/6).")
    add(f"  * Largest single sector exposure: {sector_df.index[0]} "
        f"({sector_df.iloc[0]['Weight %']:.1f}%).")
    add("=" * 86)
    return "\n".join(L)


# %% ===================================================================================
#  SECTION 13 -- MAIN ORCHESTRATION
#  -------------------------------------------------------------------------------------
#  Runs the entire pipeline end-to-end and prints every table, chart and report.
# ======================================================================================

def run_backtest(cfg):
    print("=" * 86)
    print("  INSTITUTIONAL PORTFOLIO BACKTESTER  |  initializing")
    print("=" * 86)

    # ----- 1. DATA -------------------------------------------------------------------
    print("\n[1/9] Downloading price data ...")
    prices, volume, dropped = download_prices(cfg["tickers"], cfg["start"], cfg["end"])
    if prices.shape[1] < 2:
        raise RuntimeError("Fewer than 2 tickers survived data cleaning -- aborting.")
    bench_raw, _, _ = download_prices(cfg["benchmark"], cfg["start"], cfg["end"])
    bench_prices = bench_raw.iloc[:, 0]

    # Align portfolio and benchmark on common trading days
    common = prices.index.intersection(bench_prices.index)
    prices, bench_prices = prices.loc[common], bench_prices.loc[common]
    print(f"      Universe: {list(prices.columns)}")
    print(f"      Period:   {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices)} trading days)")

    print("\n[2/9] Fetching market caps & sectors ...")
    fundamentals = fetch_fundamentals(list(prices.columns))

    # ----- 2. WEIGHTS & SIMULATION ---------------------------------------------------
    print("\n[3/9] Building weights & simulating portfolio ...")
    weights = build_weights(list(prices.columns), cfg["weighting"],
                            cfg["custom_weights"], fundamentals["market_cap"])
    port_value, weights_history = simulate_portfolio(
        prices, weights, cfg["initial"], cfg["rebalance"], cfg["cost"])
    bench_value = benchmark_value_series(bench_prices, cfg["initial"])

    weights_pct = (weights * 100).round(2).sort_values(ascending=False)
    print("\n      Target weights (%):")
    print(tabulate(weights_pct.to_frame("Weight %"), headers="keys", tablefmt="github"))

    # ----- 3. PERFORMANCE METRICS ----------------------------------------------------
    print("\n[4/9] Computing performance analytics ...")
    metrics = compute_all_metrics(port_value, bench_value, cfg["rf"])
    bench_metrics = compute_all_metrics(bench_value, bench_value, cfg["rf"])
    metric_table = pd.DataFrame({"Portfolio": metrics, "SPY": bench_metrics}).round(3)
    print("\n" + "=" * 86)
    print("  PERFORMANCE ANALYTICS")
    print("=" * 86)
    print(tabulate(metric_table, headers="keys", tablefmt="github", floatfmt=",.3f"))

    # Rolling diagnostics (printed as compact tails)
    pr = to_returns(port_value)
    roll_1y_ret = (port_value / port_value.shift(cfg["roll"]) - 1).dropna() * 100
    roll_vol = (pr.rolling(cfg["roll"]).std() * np.sqrt(TRADING_DAYS) * 100).dropna()
    roll_sharpe = ((pr.rolling(cfg["roll"]).mean() - cfg["rf"] / TRADING_DAYS)
                   / pr.rolling(cfg["roll"]).std() * np.sqrt(TRADING_DAYS)).dropna()
    print("\n  Rolling 1Y diagnostics (latest values):")
    print(f"    Rolling 1Y return: {roll_1y_ret.iloc[-1]:.2f}%   "
          f"Rolling vol: {roll_vol.iloc[-1]:.2f}%   "
          f"Rolling Sharpe: {roll_sharpe.iloc[-1]:.2f}")

    # ----- 4. BENCHMARK COMPARISON ---------------------------------------------------
    print("\n" + "=" * 86)
    print("  BENCHMARK COMPARISON")
    print("=" * 86)
    comp_table, bench_extras, beat, verdict = benchmark_comparison(
        port_value, bench_value, cfg["rf"])
    print(tabulate(comp_table.round(3), headers="keys", tablefmt="github", floatfmt=",.3f"))
    print("\n  Relationship statistics:")
    for k, v in bench_extras.items():
        print(f"    {k:<28} {v:,.3f}")
    print(f"\n  >>> DID PORTFOLIO BEAT THE MARKET?  {verdict}")

    # ----- 5. STOCK LEADERBOARD ------------------------------------------------------
    print("\n" + "=" * 86)
    print("  INDIVIDUAL STOCK LEADERBOARD  (best -> worst)")
    print("=" * 86)
    leaderboard = stock_leaderboard(prices, weights, port_value, cfg["rf"])
    print(tabulate(leaderboard.round(2), headers="keys", tablefmt="github", floatfmt=",.2f"))

    # ----- 6. RISK ANALYSIS ----------------------------------------------------------
    print("\n" + "=" * 86)
    print("  RISK ANALYSIS")
    print("=" * 86)
    dd_table = largest_drawdowns(port_value)
    best_days, worst_days = best_worst_days(port_value)
    print("\n  Largest drawdowns:")
    print(tabulate(dd_table, headers="keys", tablefmt="github", floatfmt=",.2f"))
    print("\n  Best days (%):")
    print(tabulate(best_days.round(2), headers="keys", tablefmt="github"))
    print("\n  Worst days (%):")
    print(tabulate(worst_days.round(2), headers="keys", tablefmt="github"))

    print(f"\n  Value at Risk (95%, daily):        {metrics['VaR 95% (daily) %']:.2f}%")
    print(f"  Conditional VaR (95%, daily):      {metrics['CVaR 95% (daily) %']:.2f}%")
    print(f"  Portfolio beta vs SPY:             {metrics['Beta vs SPY']:.3f}")

    cov_matrix = prices.pct_change().dropna().cov() * TRADING_DAYS  # annualized
    corr_matrix = prices.pct_change().dropna().corr()
    print("\n  Annualized covariance matrix (head):")
    print(tabulate(cov_matrix.iloc[:5, :5].round(4), headers="keys", tablefmt="github"))

    # ----- 7. MONTE CARLO ------------------------------------------------------------
    print("\n" + "=" * 86)
    print(f"  MONTE CARLO SIMULATION  ({cfg['mc_sims']:,} paths, bootstrap)")
    print("=" * 86)
    mc_results = monte_carlo(port_value, bench_value, cfg["initial"],
                             cfg["mc_horizons"], cfg["mc_sims"])
    mc_rows = {
        f"{yrs}-Year": {
            "Median $": r["median"],
            "10th pct $": r["p10"],
            "90th pct $": r["p90"],
            "P(loss) %": r["prob_loss"] * 100,
            "P(beat SPY) %": r["prob_beat_benchmark"] * 100,
        }
        for yrs, r in mc_results.items()
    }
    print(tabulate(pd.DataFrame(mc_rows).T, headers="keys", tablefmt="github", floatfmt=",.1f"))

    # ----- 8. SECTOR ANALYSIS --------------------------------------------------------
    print("\n" + "=" * 86)
    print("  SECTOR ANALYSIS")
    print("=" * 86)
    sector_df = sector_analysis(prices, weights, fundamentals)
    print(tabulate(sector_df.round(2), headers="keys", tablefmt="github", floatfmt=",.2f"))
    print(f"\n  Top contributing sector:  {sector_df['Contribution %'].idxmax()}")
    print(f"  Worst contributing sector: {sector_df['Contribution %'].idxmin()}")

    # ----- 9. CHARTS -----------------------------------------------------------------
    print("\n" + "=" * 86)
    print("  CHARTS  (rendering 14 figures)")
    print("=" * 86)
    make_all_charts(prices, port_value, bench_value, weights, mc_results,
                    sector_df, leaderboard, cfg["rf"], cfg["roll"])

    # ----- REPORT & GRADE ------------------------------------------------------------
    grade = grade_portfolio(metrics, beat, bench_extras["Outperformance % (total)"])
    report = generate_report(metrics, bench_extras, beat, verdict, leaderboard,
                             sector_df, mc_results, dd_table, grade)
    print("\n" + report)

    # ----- EXECUTIVE OUTPUT ----------------------------------------------------------
    print("\n" + "=" * 86)
    print("  EXECUTIVE SUMMARY")
    print("=" * 86)
    summary = [
        ["Initial Investment",      f"${cfg['initial']:,.0f}"],
        ["Final Portfolio Value",   f"${port_value.iloc[-1]:,.0f}"],
        ["Benchmark Final Value",   f"${bench_value.iloc[-1]:,.0f}"],
        ["Total Return %",          f"{metrics['Absolute Return %']:.2f}%"],
        ["CAGR",                    f"{metrics['CAGR %']:.2f}%"],
        ["Sharpe Ratio",            f"{metrics['Sharpe Ratio']:.2f}"],
        ["Max Drawdown",            f"{metrics['Max Drawdown %']:.2f}%"],
        ["Alpha (annual)",          f"{metrics['Alpha (annual) %']:.2f}%"],
        ["Beta",                    f"{metrics['Beta vs SPY']:.2f}"],
        ["Outperformance % (total)", f"{bench_extras['Outperformance % (total)']:.2f}%"],
        ["INVESTMENT GRADE",        f"{grade[0]}  (score {grade[1]}/6)"],
    ]
    print(tabulate(summary, headers=["Metric", "Value"], tablefmt="github"))
    print("=" * 86)

    # Return everything so the user can inspect objects after the run
    return dict(
        prices=prices, volume=volume, fundamentals=fundamentals, weights=weights,
        port_value=port_value, bench_value=bench_value, metrics=metric_table,
        comparison=comp_table, leaderboard=leaderboard, sector=sector_df,
        monte_carlo=mc_results, drawdowns=dd_table, cov=cov_matrix, corr=corr_matrix,
        grade=grade, report=report,
    )


# %% ===================================================================================
#  RUN IT
# ======================================================================================
if __name__ == "__main__":
    results = run_backtest(CONFIG)
