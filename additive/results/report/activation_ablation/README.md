# Activation Ablation Results

This report-level run compares the additive mainline under six nonlinear updates:

- `softmax_beta_1p0`
- `entmax_alpha_1p2`
- `entmax_alpha_1p5`
- `entmax_alpha_1p8`
- `sparsemax`
- `bounded_confidence`

across three scenarios:

- `shallow_low_conflict`
- `medium_conflict`
- `deep_high_conflict`

## Key files

- `summary.csv`: combined summary with `accuracy`, `mean_regret`, `mean_epsilon`, `avg_entropy`, `avg_support_size`
- `strategy_summary.csv`
- `regret_summary.csv`
- `epsilon_summary.csv`
- `diagnostics_summary.csv`
- `activation_ablation_mlp.png`

## Main findings

1. `ResidualMLP` performs best on the smoother / denser nonlinearities.
   - Best average `mean_regret`: `entmax_alpha_1p2` (`0.001537`)
   - Next best: `softmax_beta_1p0` (`0.001744`)

2. Sharper or threshold-like nonlinearities make the surrogate less reliable.
   - `bounded_confidence` is the hardest on average:
     - `accuracy = 0.805556`
     - `mean_regret = 0.020830`
     - `mean_epsilon = 0.110900`
   - `entmax_alpha_1p5`, `entmax_alpha_1p8`, and `sparsemax` also degrade surrogate quality relative to `softmax` / `entmax_alpha_1p2`.

3. Distribution sparsity tracks the difficulty pattern.
   - `softmax_beta_1p0` and `entmax_alpha_1p2` stay dense with average support size `4.0`
   - `entmax_alpha_1p5`, `entmax_alpha_1p8`, and `sparsemax` reduce support size to roughly `1.9` to `2.2`

4. `QuantumEncodedGame` remains nearly exact across all tested nonlinearities.
   - For `entmax_alpha_1p5` and `sparsemax`, the report-level summary is exact to the displayed precision
   - Small nonzero `regret` or sub-1 `accuracy` under `softmax` / `bounded_confidence` come from observable evaluation via amplitude-estimation discretization rather than a changed state dynamics

## Interpretation

This run supports the hypothesis that the additive conclusions depend meaningfully on the choice of `f`.

- If `f` is very smooth and dense, a learned classical surrogate can preserve the strategic structure well.
- If `f` is sparse or threshold-like, the same surrogate becomes less trustworthy on `Accuracy`, `Regret`, and `Epsilon`.
- The exact quantum encoding continues to preserve the additive GT dynamics themselves; the visible residual gap is dominated by the observable-estimation layer.
