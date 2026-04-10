# Additive Workspace

This top-level folder is the working home for the additive project materials that were previously spread across multiple repo directories.

## Layout

- `configs/`
  - experiment configs for quick and report runs
- `results/`
  - generated experiment outputs, plots, summaries, and follow-up reports
- `references/`
  - downloaded papers and planning notes
- `slides/`
  - additive-only presentation source, assets, and rendered slide outputs
- `code/`
  - navigation note for the additive package modules that remain under `interference_game/additive/`
- `tests/`
  - navigation note for the additive-specific pytest entry point

## Code Location

The executable package code still lives in:

- `interference_game/additive/`

It was intentionally left there to avoid breaking imports, packaging, and tests.

The additive-specific pytest coverage also still lives in:

- `tests/test_additive_mainline.py`

That test entry point was left in the main `tests/` tree so the existing pytest configuration keeps working without extra collection rules.

## Main Entry Points

- Configs: `additive/configs/report/`
- Report results: `additive/results/report/`
- References: `additive/references/`
- Slides: `additive/slides/additive_report/`
- Code: `interference_game/additive/`
- Tests: `tests/test_additive_mainline.py`

## Notes

- New additive experiment outputs now write to `additive/results/...`
- The Waterloo additive deck now lives under `additive/slides/additive_report/`
- This folder is the human-facing home for additive work; the Python package and pytest entry point remain in their standard locations for compatibility.
