# Mixing-Enhanced Interference Game

Implementation of the proposal in [route2_proposal.pdf](route2_proposal.pdf) as a reproducible Python project with:

- exact state evolution for the mixing-enhanced interference game
- aggregate, mean-field, sampling, and no-mixing baselines
- discrete equilibrium enumeration for small games
- continuous-time style dynamics simulations
- config-driven experiments that save raw outputs and generate figures from saved results

## Modeling Assumptions

- `N` is the number of agents; `L` is the number of mixing rounds.
- In each round, every agent applies `D_i(theta_i / max(L, 1))` in fixed order.
- When `L > 0`, the same round-specific mixer is applied after each agent action in that round.
- `L = 0` reduces to the no-mixing diagonal phase model.
- Target conflict is controlled by a shared-anchor target generator; overlap is recorded rather than forced exactly.
- The sampling baseline estimates utilities from Monte Carlo samples over the computational-basis outcome distribution, so it intentionally loses amplitude-sensitive interference information.

## Environment

Create the isolated Conda environment declared in [environment.yml](environment.yml):

```powershell
conda env create -f environment.yml
conda activate ece752-route2
```

## Quick Start

Run the full smoke-test experiment suite:

```powershell
python -m interference_game.experiments.run_sanity --config configs/quick/sanity.yaml
python -m interference_game.experiments.run_payoff_distortion --config configs/quick/payoff.yaml
python -m interference_game.experiments.run_equilibrium_distortion --config configs/quick/equilibrium.yaml
python -m interference_game.experiments.run_dynamics --config configs/quick/dynamics.yaml
python -m interference_game.experiments.run_ablation --config configs/quick/ablation.yaml
pytest -q
```

Generated artifacts are written below [results](results).

## Additive Workspace

The additive project materials are now grouped under [additive](additive):

- [additive/configs](additive/configs): additive experiment configs
- [additive/results](additive/results): additive outputs, summaries, plots, and reports
- [additive/references](additive/references): additive papers and planning notes
- [additive/slides](additive/slides): additive-only slide deck assets and rendered outputs

The additive package code still remains in [interference_game/additive](interference_game/additive) so existing imports and scripts continue to work.

## Project Layout

- [interference_game/models](interference_game/models): exact model, baselines, and target generation
- [interference_game/equilibrium](interference_game/equilibrium): exhaustive discrete equilibrium search
- [interference_game/dynamics](interference_game/dynamics): best-response, projected gradient, and extra-gradient simulations
- [interference_game/utils](interference_game/utils): configs, metrics, serialization, and plotting helpers
- [interference_game/experiments](interference_game/experiments): config-driven experiment runners and plot scripts
- [configs](configs): quick and full YAML configs
- [tests](tests): unit and integration tests
