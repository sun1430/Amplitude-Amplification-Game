# Observable-Only Markov Benchmark Summary

This note summarizes the stricter Markov special-case benchmark where the task is **not** to reconstruct the full terminal distribution `p_T`, but only to estimate payoff-relevant observables under a fixed query budget.

## What Was Compared

- `amplitude_estimation`
  - quantum side
  - starts from the signal induced by the single-step reversible sparse Markov operator
  - estimates observable expectations directly from the signal
- `monte_carlo`
  - classical side
  - rolls out Markov trajectories from the same source distribution
  - estimates the same observables from sampled terminal states

This benchmark avoids the invalid shortcut of first materializing the full terminal distribution and then sampling from it.

## Main Figures

- Strategy-level tradeoff: [markov_special_case_estimation_strategy.png](markov_special_case_estimation_strategy.png)
- Observable-error tradeoff: [markov_special_case_observable.png](markov_special_case_observable.png)

## Aggregated Strategy Results

Averaged across the tested Markov horizons:

| method | query budget | strategy accuracy | mean regret |
| --- | ---: | ---: | ---: |
| amplitude estimation | 8 | 0.9125 | 0.002147 |
| monte carlo | 8 | 0.2167 | 0.056008 |
| amplitude estimation | 32 | 0.9333 | 0.001474 |
| monte carlo | 32 | 0.4000 | 0.033695 |
| amplitude estimation | 128 | 0.9667 | 0.000139 |
| monte carlo | 128 | 0.5667 | 0.014227 |

## Aggregated Observable-Error Results

| method | query budget | observable error | utility error |
| --- | ---: | ---: | ---: |
| amplitude estimation | 8 | 0.026568 | 0.026568 |
| monte carlo | 8 | 0.108330 | 0.108330 |
| amplitude estimation | 32 | 0.014516 | 0.014516 |
| monte carlo | 32 | 0.053534 | 0.053534 |
| amplitude estimation | 128 | 0.004719 | 0.004719 |
| monte carlo | 128 | 0.023802 | 0.023802 |

## Interpretation

Three points are robust in this benchmark.

1. Amplitude estimation preserves strategic decisions much better than classical trajectory sampling under the same query budget.
   - At budget `8`, strategy accuracy is `0.9125` for AE versus `0.2167` for Monte Carlo.
   - At budget `128`, AE reaches `0.9667`, while Monte Carlo reaches `0.5667`.

2. The regret gap is also large.
   - At budget `8`, AE mean regret is `0.002147`, compared with `0.056008` for Monte Carlo.
   - At budget `128`, AE drops to `0.000139`, while Monte Carlo remains at `0.014227`.

3. The same pattern appears at the observable-estimation level.
   - At budget `8`, AE error is about `4.1x` lower than Monte Carlo.
   - At budget `32`, AE error is about `3.7x` lower.
   - At budget `128`, AE error is about `5.0x` lower.

## Bottom Line

Under the stricter condition that the algorithm does **not** explicitly reconstruct the full terminal distribution and only reads payoff-relevant observables, amplitude estimation clearly outperforms classical trajectory sampling in this reversible sparse Markov special case.
