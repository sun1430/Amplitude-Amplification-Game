# Additive Test Navigation

The additive-specific pytest entry point remains in:

- `../../tests/test_additive_mainline.py`

It stays under the repository-level `tests/` folder so the current pytest configuration continues to collect it automatically.

That file covers:

- additive distribution outputs and scoring
- quick strategy / regret / epsilon runs
- observable-estimation benchmark runs
- activation-family ablation
- Markov special-case runs
