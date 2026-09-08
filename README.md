# RiskLens — Portfolio Risk and Stress Testing

RiskLens asks a quantitative-risk question: **does a more stable covariance estimate produce a less volatile portfolio, and does that make its tail-risk forecast reliable?** It compares sample, exponentially weighted and Ledoit–Wolf covariance estimates on real daily industry research returns, with constrained optimization, chronological evaluation and explicit transaction-cost assumptions.

The answer in this experiment is mixed: minimum-variance baskets have lower realized volatility and drawdown than equal weighting, but more turnover and lower hypothetical returns. All tested Gaussian 99% VaR models substantially exceed the nominal exception rate. **Lower volatility is not a reliable-tail guarantee or evidence of trading alpha.**

![Held-out volatility and drawdown](outputs/risk-comparison.png)

## Technical snapshot

| Question | Implementation |
|---|---|
| What is estimated? | Rolling sample, EWMA and Ledoit–Wolf covariance matrices |
| What is optimized? | Long-only minimum variance, weights sum to one, 15% maximum industry weight |
| Baselines | Monthly equal weight and inverse volatility |
| Data | French 48-industry value-weighted daily returns; 9,067 observations, 1990–2025 |
| Development | 2013–2018; choose EWMA decay from 0.94/0.97/0.99 using realized volatility only |
| Frozen evaluation | 2019–2025: 1,760 trading days and 84 monthly rebalances |
| Risk evaluation | Volatility, drawdown, turnover, covariance conditioning, 95/99% VaR exceptions and stress windows |
| Not demonstrated | Executable instruments, net trading alpha, live deployment, regulatory validation |

## System flow

```text
official pinned daily research returns -> value-weighted table and sentinel checks
    -> development-only EWMA decay selection
    -> prior 252 trading days -> covariance -> constrained monthly weights
    -> next-day returns with daily holdings drift -> turnover and cost scenarios
    -> frozen-test volatility, drawdown, covariance audits and tail exceptions
```

## Architecture

```mermaid
flowchart LR
    A[Official French research returns] --> B[Snapshot hash and data validation]
    B --> C[Past-only rolling histories]
    C --> D[Sample covariance]
    C --> E[EWMA covariance]
    C --> F[Ledoit-Wolf shrinkage]
    V[2013-2018 development] --> G[Select EWMA decay]
    G --> E
    D --> H[Constrained minimum variance]
    E --> H
    F --> H
    C --> I[Equal-weight and inverse-vol baselines]
    H --> J[Next-day weights and daily drift]
    I --> J
    J --> K[2019-2025 frozen risk evaluation]
    K --> L[Cost sensitivity and VaR failures]
```

### Pre-processing and chronology

The parser selects the **value-weighted daily** table, not the later equal-weighted table. Percentage returns become decimal simple returns. Missing sentinels `-99.99`/`-999` cause the entire date to be removed; no interpolation or forward fill is used. This snapshot has **zero missing dates** within the requested range, January 2, 1990–December 31, 2025.

The downloaded July 2026 CRSP-based snapshot is pinned by SHA-256. Data from 1990–2012 provide historical context; rolling estimation always uses only the prior 252 available trading days. EWMA decay selection uses 2013–2018 only. The other methods, window length, weight cap, rebalance frequency, cost scenarios and stress dates are fixed rather than test-tuned. All three attempted decay values and their development results remain in the report.

### Covariance and optimization

Sample covariance uses the unbiased sample estimate. EWMA normalizes exponentially decaying weights and uses a weighted-centering/effective-sample correction. Ledoit–Wolf estimates its shrinkage intensity on each historical window. A tiny diagonal numerical floor preserves positive definiteness. Audits record the minimum eigenvalue, condition number, maximum weight, history endpoints and shrinkage coefficient where applicable.

SLSQP solves the long-only quadratic program with an analytic gradient, fully invested weights and a 15% per-industry cap. Failed optimization or violated constraints raises an error rather than silently substituting a baseline. Equal-weight and inverse-volatility baselines are separately defined reference allocations, not tuned optimizers.

### Portfolio accounting and risk forecasts

Weights are formed from history ending **before** the first trading day of the month. Holdings drift with each day's returns between rebalances; this is not a hidden daily-rebalanced portfolio. All methods begin each evaluation period from an equal-weight allocation, and the first rebalance incurs its applicable adjustment cost.

One-way turnover is `0.5 * sum(abs(target - drifted_previous_weights))`. Hypothetical costs charge 0/5/10/25 basis points on the **full traded notional**, with net daily return `(1 - cost) * (1 + gross_return) - 1`. Costs do not alter relative holdings because they are deducted proportionally. Underlying industry-basket constituent turnover, spread, borrowing and market impact are not modeled.

VaR uses a zero-mean Gaussian one-day model. Its covariance is **frozen between monthly updates**, while exposure weights drift daily. Exceptions compare gross daily returns with the predicted loss threshold. Kupiec coverage statistics are exploratory diagnostics under idealized assumptions, not regulatory certification or a test of exception independence.

## What works today

- Pinned official-source download, table parsing, missing-sentinel and date checks.
- Three covariance estimators and bounded numerical portfolio optimization.
- Development-only parameter choice, lagged allocations and daily holdings accounting.
- Frozen risk metrics, four cost scenarios, monthly covariance audits and predeclared stress windows.
- Reproducible CLI, real-data plots and small contract-test CI without dataset downloads.

## Measured results, with context

The selected EWMA decay is **0.99**. The following are **gross hypothetical research-basket** outcomes across 1,760 held-out trading days, 2019–2025. See [full results](outputs/evaluation.json) and [source provenance](outputs/provenance.json).

| Method | Annualized volatility | Maximum drawdown | Annual one-way turnover | 99% VaR exceptions |
|---|---:|---:|---:|---:|
| Equal weight | 20.38% | −38.24% | 0.22× | 42 / 1,760 |
| Inverse volatility | 19.32% | −37.36% | 0.24× | 41 / 1,760 |
| Sample min-variance | 15.30% | −29.48% | 1.41× | 43 / 1,760 |
| EWMA min-variance | 15.17% | −28.01% | 1.91× | 42 / 1,760 |
| Ledoit–Wolf min-variance | 15.31% | −29.59% | 1.38× | 43 / 1,760 |

The **2.33–2.44%** observed 99%-VaR exception rates exceed the nominal 1%; lower realized volatility does not rescue the stale Gaussian tail forecast. Shrinkage does not outperform sample covariance on realized volatility in this run.

The risk reduction is not a free return improvement: gross annualized research-basket return is **10.58% for EWMA versus 15.22% for equal weighting**. With a hypothetical 25-bp charge on full traded notional, EWMA's annualized return becomes **9.53%**. These are scenario calculations on research baskets, not achievable investment returns or a recommendation.

Stress reports separately cover February 19–April 30, 2020; calendar 2022; and April 2025. These historically chosen episodes are illustrations, not independent confirmations. Drawdown in each stress slice is measured relative to that slice's starting capital and running peak.

## Requirements

Python 3.12 and CPU. The official ZIP is about 4 MB; the 48-column daily matrix and optimization workload are small. Numeric libraries are limited to two threads. No GPU, paid dataset, brokerage access or trading account is required. Raw returns and daily strategy streams remain local and are excluded from Git.

## Run on macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python risklens.py prepare
python -m pytest -q
python risklens.py run
```

The authoring host reused a compatible existing Python environment; the same pinned requirements support this standalone setup. There is no trained neural checkpoint to download: covariance models and allocations are re-estimated from historical windows by `run`.

The official URL is updated over time. Exact reproduction requires the original local ZIP with SHA-256 `d7701a576a75b2e1630de546ef4ff614cb54edf62bba52f503a975d50e287b60`. The program refuses a different vintage by default. `python risklens.py prepare --accept-new-vintage` explicitly starts a **new-vintage experiment**, not an exact reproduction of these results.

## Validate the installation

```bash
python -m pytest -q
```

Tests cover covariance positive definiteness, optimizer feasibility and caps, holdings drift, starting-capital drawdown, VaR exceptions and future-perturbation invariance. Their synthetic fixtures test software contracts only; all reported portfolio results use real source returns.

## Limitations and next experiments

- French industry portfolios are research series, not directly executable funds or point-in-time holdings.
- Historical CRSP revisions and the 2025 source-format transition limit investability claims.
- Monthly-frozen Gaussian covariance fails to represent crisis tails; report the failure rather than call the model validated.
- Hypothetical transaction costs omit internal basket turnover and execution constraints.
- One dataset vintage, fixed strategy families and a small predeclared development grid; no broad search or significance-adjusted claim of a winning strategy.
- Next: daily-updated risk forecasts, heavy-tailed innovations, volatility-targeted comparisons and independent-vintage sensitivity, with new holdouts before further model selection.

## Sources and licences

- [Kenneth French official Data Library, industry returns and methodology notes](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
- [Official 48-industry daily download](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip).
- [Scikit-learn Ledoit–Wolf estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

Code is MIT licensed. Source data remain subject to their provider's terms and are not redistributed or relicensed here. Aggregate reports and plots identify their data source; no endorsement by the source providers is implied.
