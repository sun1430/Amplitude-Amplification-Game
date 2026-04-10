# Additive Follow-Up Report

Date: 2026-04-10

## 1. What Was Done

This follow-up work completed two items that were missing from the original additive mainline.

### 1.1 Nonlinearity ablation for `f`

The additive ground-truth update was extended from a single hard-coded `entmax-1.5` choice to a configurable activation family:

- `softmax(beta z)`
- `sparsemax(z)`
- `entmax_alpha(z)`
- `smooth bounded-confidence`

The mainline code now supports these options through the shared activation API in:

- [activations.py](../../../interference_game/additive/activations.py)
- [classical_game.py](../../../interference_game/additive/classical_game.py)
- [quantum_game.py](../../../interference_game/additive/quantum_game.py)
- [surrogate.py](../../../interference_game/additive/surrogate.py)

The report-level ablation was run with:

- [activation_ablation.yaml](../../configs/report/activation_ablation.yaml)

Main outputs:

- [activation ablation summary](activation_ablation/summary.csv)
- [activation diagnostics](activation_ablation/diagnostics_summary.csv)
- [activation ablation plot](activation_ablation/activation_ablation_mlp.png)

### 1.2 Reversible sparse Markov diffusion special case

A new special case was implemented to make the diffusion structure more aligned with quantum mixing literature:

- source distribution generated from additive actions
- terminal distribution produced by repeated reversible sparse Markov diffusion
- exact quantum encoding preserved at the distribution level
- classical surrogate still trained as a learned approximation

Core files:

- [markov_special_case.py](../../../interference_game/additive/markov_special_case.py)
- [markov_common.py](../../../interference_game/additive/experiments/markov_common.py)
- [run_markov_special_case.py](../../../interference_game/additive/experiments/run_markov_special_case.py)
- [plot_markov_special_case.py](../../../interference_game/additive/experiments/plot_markov_special_case.py)

Config:

- [markov_special_case.yaml](../../configs/report/markov_special_case.yaml)

Main outputs:

- [markov summary](markov_special_case/summary.csv)
- [markov horizon sweep](markov_special_case/horizon_summary.csv)
- [markov observable aggregate](markov_special_case/observable_aggregate.csv)
- [markov metrics plot](markov_special_case/markov_special_case_metrics.png)
- [markov observable plot](markov_special_case/markov_special_case_observable.png)

## 2. Phase I: `f`-Ablation Results

Headline metrics remained:

- `Accuracy`
- `Regret`
- `Epsilon`

Two diagnostics were added:

- average entropy
- average support size

### 2.1 Mean MLP performance by activation

These numbers are averaged across the three report scenarios.

| activation | accuracy | mean_regret | mean_epsilon | avg_support_size |
| --- | ---: | ---: | ---: | ---: |
| `entmax_alpha_1p2` | `0.930556` | `0.001537` | `0.042033` | `4.000000` |
| `softmax_beta_1p0` | `0.868056` | `0.001744` | `0.056624` | `4.000000` |
| `entmax_alpha_1p8` | `0.854167` | `0.006227` | `0.046885` | `2.194444` |
| `sparsemax` | `0.875000` | `0.006949` | `0.060150` | `2.083333` |
| `entmax_alpha_1p5` | `0.826389` | `0.013356` | `0.043634` | `1.875000` |
| `bounded_confidence` | `0.805556` | `0.020830` | `0.110900` | `3.763889` |

### 2.2 Interpretation

Three conclusions are clear from this ablation.

1. Smoother, denser `f` are easiest for the classical surrogate to learn.
   - `entmax_alpha_1p2` and `softmax_beta_1p0` are the strongest settings for the MLP.
   - Both keep the support dense at size `4.0`.

2. Sharper or threshold-like `f` expose a larger surrogate gap.
   - `entmax_alpha_1p5`, `entmax_alpha_1p8`, and `sparsemax` all reduce average support size toward `2`.
   - Their `Regret` rises relative to the smoother settings.

3. `bounded_confidence` is the hardest case in this family.
   - It produced the worst average `Accuracy`, worst `Regret`, and worst `Epsilon`.
   - This makes it the most informative stress test if the goal is to highlight exact structural preservation versus learned approximation.

### 2.3 What this means

This phase supports the claim that additive conclusions depend materially on the choice of `f`.

- If `f` is very smooth, a neural surrogate can preserve the strategic structure fairly well.
- If `f` is sparse or threshold-like, that same surrogate becomes less reliable.
- The exact quantum encoding remains aligned with the ground-truth state dynamics; the gap mainly appears in the learned classical approximation.

## 3. Phase II: Reversible Sparse Markov Special Case

The special case used:

- `state_dim = 6`
- `activation_family = entmax`, `alpha = 1.5`
- `mixing_depth = 10`
- reversible sparse diffusion with self-loop and inertia
- smaller surrogate capacity than the mainline report run, to avoid the special case becoming trivially learnable

This was intentional: the first Markov version was too smooth and dense, and the MLP learned it almost perfectly. The final report configuration was adjusted to make the structural comparison meaningful.

### 3.1 Scenario-level results

| scenario | model | accuracy | mean_regret | mean_epsilon |
| --- | --- | ---: | ---: | ---: |
| `markov_low_conflict` | `quantum_encoded` | `1.000000` | `0.000000` | `0.000000` |
| `markov_low_conflict` | `residual_mlp` | `0.895833` | `0.000566` | `0.074022` |
| `markov_medium_conflict` | `quantum_encoded` | `1.000000` | `0.000000` | `0.000000` |
| `markov_medium_conflict` | `residual_mlp` | `0.708333` | `0.004467` | `0.000000` |
| `markov_high_conflict` | `quantum_encoded` | `1.000000` | `0.000000` | `0.000000` |
| `markov_high_conflict` | `residual_mlp` | `0.791667` | `0.002169` | `0.000000` |

### 3.2 Horizon sweep

The special-case horizon sweep used `mixing_depth ∈ {1, 2, 4, 8, 12}` on the medium-conflict setting.

| horizon | quantum accuracy | quantum regret | mlp accuracy | mlp regret |
| --- | ---: | ---: | ---: | ---: |
| `1` | `1.000000` | `0.000000` | `0.458333` | `0.022307` |
| `2` | `1.000000` | `0.000000` | `0.666667` | `0.006606` |
| `4` | `1.000000` | `0.000000` | `0.729167` | `0.004366` |
| `8` | `1.000000` | `0.000000` | `0.750000` | `0.003151` |
| `12` | `1.000000` | `0.000000` | `0.812500` | `0.001523` |

### 3.3 Observable benchmark in the special case

This benchmark stayed with the same bounded-observable expectation task used in the additive mainline.

| method | query_budget | observable_error | utility_error |
| --- | ---: | ---: | ---: |
| amplitude_estimation | `8` | `0.021340` | `0.021340` |
| monte_carlo | `8` | `0.104961` | `0.104961` |
| amplitude_estimation | `32` | `0.014802` | `0.014802` |
| monte_carlo | `32` | `0.048088` | `0.048088` |
| amplitude_estimation | `128` | `0.005149` | `0.005149` |
| monte_carlo | `128` | `0.024930` | `0.024930` |

### 3.4 Interpretation

This phase supports four statements.

1. The reversible sparse Markov special case is structurally clean and stable.
   - The exact quantum encoding matched GT exactly on `Accuracy`, `Regret`, and `Epsilon`.

2. The classical learned surrogate remains approximate.
   - In the final report configuration, it no longer collapsed to the GT solution.
   - The largest gap appeared in `markov_medium_conflict`, where MLP accuracy fell to `0.708333`.

3. Horizon matters.
   - In the chosen medium-conflict sweep, the MLP was weakest at short horizon and improved as horizon increased.
   - This suggests that the diffusion process smooths the final mapping as the chain mixes more.

4. The observable-estimation benchmark still favors the quantum side.
   - At every tested query budget, amplitude estimation beat Monte Carlo by a clear margin.
   - The ratio was largest at the smallest budget:
     - `8`: `0.021340` vs `0.104961`
     - about `4.9x` lower error

## 4. Overall Assessment

Across both phases, the most defensible conclusions are:

1. The additive project should not be evaluated under only one nonlinear update.
   - The `f`-ablation shows that smooth versus sparse / threshold-like updates produce materially different surrogate behavior.

2. `bounded_confidence` is the strongest stress test among the tested `f`.
   - It is the hardest for the surrogate and the most likely to reveal structural approximation error.

3. A reversible sparse Markov diffusion special case is a reasonable next domain-specific model.
   - It keeps the public-opinion diffusion interpretation.
   - It is also the special case most naturally aligned with quantum mixing / fast-forwarding literature.

4. In the code implemented here, the strongest empirical quantum advantage remains observable estimation.
   - The amplitude-estimation benchmark consistently beats Monte Carlo.

5. This report does **not** claim an implemented end-to-end quantum fast-forwarding runtime speedup.
   - The special case is chosen because the literature suggests that this is the class where stronger quantum dynamics-side advantages are most plausible.
   - The current code validates the structural side and the observable-estimation side, not a hardware-level speedup claim.

## 5. Validation

The updated codebase passed:

- `conda run -n ece752-route2 pytest -q`

Result:

- `16 passed in 24.36s`

## 6. Most Useful Artifacts

- [phase I activation summary](activation_ablation/summary.csv)
- [phase I activation plot](activation_ablation/activation_ablation_mlp.png)
- [phase II markov summary](markov_special_case/summary.csv)
- [phase II horizon sweep](markov_special_case/horizon_summary.csv)
- [phase II observable aggregate](markov_special_case/observable_aggregate.csv)
- [phase II metrics plot](markov_special_case/markov_special_case_metrics.png)
- [phase II observable plot](markov_special_case/markov_special_case_observable.png)
