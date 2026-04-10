# Additive Experiments: Survey and Proposed Plan

Last updated: 2026-04-09

## 1. Context

The current additive mainline uses

- a classical ground-truth update
  `p_{t+1} = entmax_{1.5}(M p_t + Σ_i B_i a_i)`
- an exact quantum encoding that preserves the terminal influence distribution
- an observable-based payoff, implemented as the expectation of a bounded event observable over the terminal distribution

In the current codebase:

- the nonlinearity `f` is fixed to `entmax-1.5`
- the observable payoff uses a top-topic event indicator per agent target
- the quantum advantage that is currently justified most cleanly is amplitude-estimation-style acceleration for bounded observable expectation estimation

Relevant code:

- [activations.py](../../interference_game/additive/activations.py)
- [classical_game.py](../../interference_game/additive/classical_game.py)
- [scoring.py](../../interference_game/additive/scoring.py)
- [quantum_game.py](../../interference_game/additive/quantum_game.py)

## 2. What The Current Results Support

### 2.1 Nonlinearity coverage is currently too narrow

Only one `f` has been tested so far: `entmax-1.5`.

That is not enough to support a claim that the additive conclusions are robust to the choice of nonlinear mixing map. The next experiments should therefore test a small, principled family of nonlinearities rather than one ad hoc replacement.

### 2.2 The current quantum advantage is mainly in observable estimation

For the present payoff design, the cleanest quantum claim is:

- estimating a bounded observable expectation over the terminal influence distribution can receive a near-quadratic query advantage via amplitude estimation relative to classical Monte Carlo

This is aligned with:

- Brassard, Høyer, Mosca, Tapp (2000), *Quantum Amplitude Amplification and Estimation*  
  Source: <https://arxiv.org/abs/quant-ph/0005055>
- Montanaro (2015), *Quantum speedup of Monte Carlo methods*  
  Source: <https://arxiv.org/abs/1504.06987>

This does **not** by itself justify a stronger claim such as a general end-to-end quantum speedup for the full additive game.

## 3. Literature Survey: Reasonable Nonlinear `f`

This section separates direct source support from modeling choices that would be adapted to the current additive framework.

### 3.1 Softmax family

#### `softmax(β z)`

Why it should be tested:

- it is the smoothest and densest probability-map baseline
- it is the most obvious control against which `entmax-1.5` should be judged
- if MLP surrogates perform well only under very smooth `f`, this should show up clearly here

Relevant source:

- Peters, Niculae, Martins (2019), *Sparse Sequence-to-Sequence Models*  
  Source: <https://arxiv.org/abs/1905.05702>

Note:

- This source studies `alpha-entmax`; softmax is the dense endpoint of that family.

### 3.2 Sparse simplex projection family

#### `sparsemax(z)`

Why it should be tested:

- it outputs a simplex distribution like entmax, but is harder and more piecewise-linear
- it should create sharper support changes than softmax
- it is the cleanest sparse baseline next to entmax

Relevant source:

- Martins, Astudillo (2016), *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*  
  Source: <https://arxiv.org/abs/1602.02068>

### 3.3 `alpha-entmax` sweep

#### `entmax_alpha(z)` for `alpha in {1.2, 1.5, 1.8, 2.0}`

Why it should be tested:

- the current setup only tests one point, `alpha = 1.5`
- sweeping `alpha` gives a principled smooth-to-sparse path
- this is the most coherent ablation because all choices stay in one activation family

Relevant source:

- Peters, Niculae, Martins (2019), *Sparse Sequence-to-Sequence Models*  
  Source: <https://arxiv.org/abs/1905.05702>

### 3.4 Smooth bounded-confidence style nonlinearity

#### Smooth bounded-confidence gate

Why it is relevant:

- bounded-confidence models are among the most standard nonlinear opinion-dynamics models
- they are directly tied to polarization, clustering, and selective influence
- this family is more domain-specific to public-opinion shaping than generic neural activations

Relevant sources:

- Hegselmann, Krause (2002), *Opinion dynamics and bounded confidence models, analysis, and simulation*  
  Source: <https://jasss.soc.surrey.ac.uk/5/3/2.html>
- Kurahashi-Nakamura, Mäs, Lorenz (2016), *Robust Clustering in Generalized Bounded Confidence Models*  
  Source: <https://www.jasss.org/19/4/7.html>

What is directly supported by the literature:

- hard bounded-confidence and smooth acceptance variants both matter structurally
- smooth acceptance functions can preserve clustering behavior that a fully smooth consensus model may wash out

What is a modeling inference for this project:

- to use bounded-confidence in the current additive simplex-logit pipeline, we would need to adapt it into a compatible probability map or influence gate
- one reasonable project-specific form would be a smooth thresholded gate applied to logits before simplex renormalization

This adaptation is **not** taken directly from the papers above; it would be a new implementation choice consistent with their modeling intuition.

## 4. Literature Survey: Where Quantum Can Do Better Than AE-Only

### 4.1 General bounded-observable payoff estimation

If the task remains:

- prepare or access a distribution
- estimate `E[g(Z)]` for a bounded observable `g`

then the most standard quantum improvement is still near-quadratic query improvement through amplitude estimation / quantum Monte Carlo.

Supported by:

- Brassard et al. (2000): <https://arxiv.org/abs/quant-ph/0005055>
- Montanaro (2015): <https://arxiv.org/abs/1504.06987>

### 4.2 Special case with stronger structure: reversible sparse Markov diffusion

This is the most promising special case for the current project.

Suggested model form:

- `p_T = P^T p_0(a)`
- or `p_T = P^T p_0 + Σ_{k=0}^{T-1} P^k b(a)`

where:

- `P` is sparse
- `P` is reversible
- `P` is stochastic

Why this is attractive:

- it still matches public-opinion shaping as diffusion / platform redistribution
- it connects naturally to Markov mixing and stationary-distribution preparation
- quantum speedups are available at the level of the diffusion process itself, not only final payoff sampling

Relevant sources:

- Orsucci, Briegel, Dunjko (2018), *Faster quantum mixing for slowly evolving sequences of Markov chains*  
  Source: <https://arxiv.org/abs/1503.01334>
- Li, Shang (2022), *Faster quantum mixing of Markov chains in non-regular graph with fewer qubits*  
  Source: <https://arxiv.org/abs/2205.06099>

Interpretation:

- this special case offers a more compelling quantum story than the current general additive GT, because the quantum algorithmic advantage can attach to diffusion / mixing itself

### 4.3 Linear affine dynamics / linear ODE special case

Another plausible special case is to linearize the dynamics:

- `p_{t+1} = A p_t + b(a)`
- or a continuous-time linear ODE analogue

Relevant sources:

- Harrow, Hassidim, Lloyd (2009), *Quantum Algorithm for Linear Systems of Equations*  
  Source: <https://arxiv.org/abs/0811.3171>
- Berry, Childs, Ostrander, Wang (2017), *Quantum algorithm for linear differential equations with exponentially improved dependence on precision*  
  Source: <https://arxiv.org/abs/1701.03684>

Interpretation:

- this route is mathematically clean, but less domain-specific than Markov diffusion
- it is strongest when the output of interest is a quantum state or an observable, not a fully materialized dense vector

### 4.4 Search advantage in a small strategic special case

If the game is reduced to a more structured finite strategic problem, a separate advantage may come from faster search over deviations or responses.

Relevant source:

- Dürr, Høyer (1996), *A Quantum Algorithm for Finding the Minimum*  
  Source: <https://arxiv.org/abs/quant-ph/9607014>

Interpretation:

- this is more naturally a best-response or deviation-search story than a dynamics story
- it is less tightly matched to the current additive opinion-shaping interpretation than the Markov-diffusion special case

## 5. Recommended Experimental Plan

## Phase I: Nonlinearity Ablation

Goal:

- test whether the additive conclusions survive beyond `entmax-1.5`
- identify which nonlinearities produce the largest gap between exact structural encoding and learned surrogate approximation

### 5.1 Models

Keep the current three-model structure:

- `ClassicalGroundTruthGame`
- `QuantumEncodedGame`
- `ResidualMLPSurrogate`

Important note:

- the current `QuantumEncodedGame` is an exact encoding of the GT distributional dynamics
- therefore, if it is updated consistently for each `f`, it should continue to match GT exactly
- the informative comparison in Phase I is therefore mainly between GT/exact-quantum and the MLP surrogate

### 5.2 Candidate nonlinearities

Implement and test:

1. `softmax(beta z)`
2. `sparsemax(z)`
3. `entmax_alpha(z)` with `alpha in {1.2, 1.5, 1.8, 2.0}`
4. `smooth bounded-confidence gate`

### 5.3 Metrics

Keep the current headline metrics:

- `Accuracy`
- `Regret`
- `Epsilon`

Add only two diagnostics in raw outputs:

- distribution entropy
- support size

These two diagnostics help explain whether a given `f` is dense, sparse, or threshold-like.

### 5.4 Expected outcome

Most likely:

- `softmax` will be easiest for the MLP to fit
- `sparsemax` and higher-`alpha` entmax will be harder and may expose larger structural errors in `Regret` and `Epsilon`
- bounded-confidence style nonlinearities may produce the sharpest support changes and the most informative stress test

## Phase II: Special Case With Stronger Quantum Leverage

Goal:

- move beyond the statement that the payoff estimator gets a near-quadratic query advantage
- identify a domain-reasonable special case where the quantum side has structural algorithmic advantages

### 5.5 Recommended special case

Prioritize:

- `reversible sparse Markov diffusion`

Reason:

- strongest match to opinion diffusion / attention redistribution
- best-supported by existing quantum mixing literature
- most plausible path to a stronger claim than AE-only

### 5.6 Experiments for the special case

Run two experiments:

1. `special_case_strategy`
   - report `Accuracy`, `Regret`, `Epsilon`
   - compare the strategic structure induced under the special-case dynamics

2. `cost_vs_error`
   - compare classical rollout / Monte Carlo against quantum-style estimation under matched query budgets
   - this is the place to show whether the special case gives cleaner separation than the current generic additive setting

### 5.7 Secondary option

If the Markov-diffusion route becomes too heavy, a fallback special case is:

- linear affine dynamics with observable readout

This is mathematically cleaner to implement, but less directly tied to public-opinion diffusion.

## 6. Proposed Implementation Order

1. Add an `activation_family` configuration to the additive mainline
2. Implement:
   - `softmax`
   - `sparsemax`
   - generic `entmax_alpha`
   - one smooth bounded-confidence variant
3. Run Phase I on the existing three scenarios:
   - `shallow_low_conflict`
   - `medium_conflict`
   - `deep_high_conflict`
4. Compare `Accuracy`, `Regret`, `Epsilon` across nonlinearities
5. Pick the most informative `f`
6. Add the `reversible_sparse_markov_diffusion` special case
7. Run the special-case strategy experiment and cost-vs-error benchmark
8. Update report and slides only after these results settle

## 7. Bottom-Line Assessment

At this point, the most defensible statements are:

- the additive project should test a family of nonlinear `f`, not only `entmax-1.5`
- the most principled immediate ablation is `softmax / sparsemax / entmax-alpha / smooth bounded-confidence`
- for the current observable-payoff setting, the clean quantum advantage is still near-quadratic estimation speedup
- if we want a stronger and more domain-relevant quantum story, the best next special case is reversible sparse Markov diffusion

## 8. Source List

- Martins, Astudillo (2016), *From Softmax to Sparsemax*  
  <https://arxiv.org/abs/1602.02068>
- Peters, Niculae, Martins (2019), *Sparse Sequence-to-Sequence Models*  
  <https://arxiv.org/abs/1905.05702>
- Hegselmann, Krause (2002), *Opinion dynamics and bounded confidence models, analysis, and simulation*  
  <https://jasss.soc.surrey.ac.uk/5/3/2.html>
- Kurahashi-Nakamura, Mäs, Lorenz (2016), *Robust Clustering in Generalized Bounded Confidence Models*  
  <https://www.jasss.org/19/4/7.html>
- Brassard, Høyer, Mosca, Tapp (2000), *Quantum Amplitude Amplification and Estimation*  
  <https://arxiv.org/abs/quant-ph/0005055>
- Montanaro (2015), *Quantum speedup of Monte Carlo methods*  
  <https://arxiv.org/abs/1504.06987>
- Orsucci, Briegel, Dunjko (2018), *Faster quantum mixing for slowly evolving sequences of Markov chains*  
  <https://arxiv.org/abs/1503.01334>
- Li, Shang (2022), *Faster quantum mixing of Markov chains in non-regular graph with fewer qubits*  
  <https://arxiv.org/abs/2205.06099>
- Harrow, Hassidim, Lloyd (2009), *Quantum Algorithm for Linear Systems of Equations*  
  <https://arxiv.org/abs/0811.3171>
- Berry, Childs, Ostrander, Wang (2017), *Quantum algorithm for linear differential equations with exponentially improved dependence on precision*  
  <https://arxiv.org/abs/1701.03684>
- Dürr, Høyer (1996), *A Quantum Algorithm for Finding the Minimum*  
  <https://arxiv.org/abs/quant-ph/9607014>
