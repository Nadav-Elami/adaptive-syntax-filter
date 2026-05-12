# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a Python research package implementing an **adaptive Kalman–EM algorithm** for learning time-varying syntax rules in behavioral sequences (primarily canary birdsong). The algorithm tracks how transition probabilities between symbols (song phrases) change over time by fitting a state-space model where latent logit parameters evolve according to block-diagonal linear dynamics.

The mathematical model: each "song" (sequence) is an observation; the hidden state is a logit vector `x_k ∈ ℝ^{R^(order+1)}` that maps to transition probabilities via block-wise softmax. The EM algorithm alternates between Kalman filtering/RTS smoothing (E-step) and closed-form parameter updates (M-step) for `Θ = {x_0, Σ, F, u}`.

## Commands

```bash
# Install in development mode (required before running anything)
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/adaptive_syntax_filter

# Run a specific test file
pytest tests/test_core/test_kalman.py

# Run a single test function
pytest tests/test_core/test_kalman.py::TestKalmanFilter::test_forward_filter

# Skip slow/performance tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Lint
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/

# Format
black src/ tests/
isort src/ tests/

# Run a pipeline experiment
python scripts/cli.py run --config configs/minimal.yml
python scripts/cli.py run --config configs/higher_order.yml --verbose

# Evaluate existing results
python scripts/cli.py evaluate <results_dir>

# Export publication figures
python scripts/cli.py export <output_dir> --format pdf

# Aggregate analysis (large-scale, multi-seed statistical analysis)
python aggregate_analysis/aggregate_analysis.py --config-id 1 --n-seeds 100 --test-mode
python aggregate_analysis/aggregate_analysis.py --config-id all --n-seeds 20000 --parallel 8
```

## Architecture

The package lives under `src/adaptive_syntax_filter/` with four subpackages:

### `core/` — The Algorithm

- **`state_space.py`**: `StateSpaceManager` is central to everything. For a Markov model of order `k` with alphabet size `R`, the state vector has dimension `R^(k+1)`, organized as `R^k` blocks of size `R`. Each block corresponds to one context (history of `k` symbols), and each element within a block is the logit for transitioning to one target symbol. `StateSpaceManager` encodes/decodes contexts, extracts blocks, and assembles block-diagonal matrices.

- **`observation_model.py`**: Block-wise softmax (`softmax_observation_model`), its Jacobian (`log_softmax_jacobian`) and Hessian (`log_softmax_hessian`). The Hessian is used directly in the Kalman update step to linearize the non-Gaussian observation model.

- **`kalman.py`**: `KalmanFilter` with `forward_filter()` (eqs. 4.1–4.6) and `rts_smoother()` (eqs. 5.1–5.4). The filter treats each song as one time step `k`; within a song, it accumulates sufficient statistics `δ_k` and `H_k` across all transitions `(y_{m-1} → y_m)` before performing the update. Positions where `np.isinf(x)` is true (forbidden `-∞` logits) are preserved through the state transition and update steps without modification.

- **`em_algorithm.py`**: `EMAlgorithm` orchestrates the full EM loop. The M-step updates `F` and `u` block-by-block via ridge-regularized least squares (`λ` controls regularization). `Sigma` is updated as a diagonal matrix from squared residuals. Adaptive damping (`damping_factor ∈ [0.1, 1.0]`) is adjusted based on consecutive log-likelihood increases/decreases. The class separately tracks `best_params` (highest log-likelihood iteration) vs `final_params`.

### `data/` — Synthetic Data Generation

- **`constraint_system.py`**: Enforces canary song grammar. The alphabet convention is `['<', phrase_1, ..., phrase_N, '>']` where `<` is start and `>` is end. Forbidden transitions (e.g., `> → phrase`) are set to `-np.inf`; required transitions are set to `1e8` (large-but-finite for numerical stability). `encode_context` / `decode_context` map between symbol sequences and integer context indices using base-`R` encoding.

- **`temporal_evolution.py`**: Implements six evolution schedules for ground-truth logit trajectories used in synthetic data: `linear`, `exponential`, `sigmoid`, `piecewise`, `oscillatory`, `constant`. Returns a `(state_dim, n_sequences)` array of true logit vectors used for data generation.

- **`sequence_generator.py`**: `SequenceGenerator` and `generate_dataset` produce synthetic sequences from a given logit trajectory. `softmax_mc_higher_order` is the reference block-wise softmax used throughout — also imported by `kalman.py` and `observation_model.py`.

- **`alphabet_manager.py`**: Utilities for alphabet sizing, memory estimation, and preset configurations. Presets define alphabet size only — `'bengalese_finch'` uses 16 symbols, `'canary'` uses 40 symbols; Markov order is configured separately.

- **`dataset_builder.py`**: `DatasetBuilder` assembles `Dataset` objects with metadata; `create_research_datasets` builds standard benchmark datasets.

### `config/` — Configuration

- **`settings.py`**: `Settings` dataclass with TOML load/save. A global singleton is accessed via `get_config()` / `set_config()`. Preset names: `'bengalese_finch'`, `'canary'`, `'minimal'`.
- **`defaults.py`**: `DefaultConfig` and `RESEARCH_CONFIGS` dict with preset values.
- **`random_state.py`**: `set_global_seed()` used by `conftest.py` to seed all tests.

### `viz/` — Visualization

- **`logit_evolution.py`**: `LogitEvolutionDashboard` — main tool for visualizing how logit parameters evolve over sequences (estimated vs. true).
- **`probability_evolution.py`**: Transition probability heatmap series.
- **`performance_assessment.py`**: Cross-entropy, RMSE, and other model fit metrics.
- **`publication_figures.py`**: `PublicationFigureManager` for paper-ready figures.

### `aggregate_analysis/` — Large-Scale Statistical Analysis

Scripts run from the repo root (not installed as a package). They use `from src.adaptive_syntax_filter...` imports directly (relying on the repo root being in `sys.path`), and also import `config_cli` / `research_pipeline` as bare names (siblings in `scripts/`), so `scripts/` must also be on `sys.path` — easiest achieved by running from the repo root with `python aggregate_analysis/aggregate_analysis.py ...`. Four canonical experiment configs (`configs/aggregate_config_*.yml`) cover: linear/sigmoid evolution × 1st/2nd order Markov. Analysis uses `h5py` + `pandas` for storing results and `seaborn` for visualization.

### `scripts/` — Research Pipeline

- **`cli.py`**: Main CLI entry point. Sub-commands: `run`, `evaluate`, `export`, `clean`. Dispatches to `research_pipeline.py` via a bare `import research_pipeline` (sibling module, not a package import). Running `python scripts/cli.py ...` from the repo root works because Python automatically adds the script's directory (`scripts/`) to `sys.path`.
- **`research_pipeline.py`**: `run_pipeline(config)` — top-level function called by the CLI.
- **`config_cli.py`**: Config reading utilities used by `aggregate_analysis/`.

## Key Conventions

**State vector layout**: Element at index `context_idx * R + target_idx` is the logit for transitioning from context `context_idx` to symbol `target_idx`. For 1st order with alphabet `['<', 'a', 'b', '>']` (R=4), element `4*i + j` is logit for `alphabet[i] → alphabet[j]`.

**Alphabet convention**: Always `['<', content_symbols..., '>']`. The `<` start token must always appear first, `>` end token last. Constraints set `<` as a forbidden *destination* and `>` as a forbidden *source* for mid-sequence transitions.

**Observations format**: Sequences passed to `EMAlgorithm.fit()` are `List[np.ndarray]` where each array contains integer symbol indices (not strings). Use `sequences_to_observations(sequences, alphabet)` to convert string sequences.

**Import paths**: Core modules use a mixed style. Within-subpackage imports use relative form (`from .kalman import KalmanFilter`), while cross-subpackage imports use the full `src.*` path (`from src.adaptive_syntax_filter.data.constraint_system import encode_context`). The `src.*` form works because `src/` is on `sys.path` when the package is installed in editable mode or when scripts manually set it up.

**Test markers**: `slow` (auto-applied to tests with "performance" or "large" in node id), `integration`, `visual`. Use `-m "not slow"` for quick iteration.

**Config files**: YAML configs under `configs/` specify `data.*` (alphabet, order, n_sequences, evolution_type) and `em.*` (max_iterations, tolerance, regularization_lambda, damping_factor, adaptive_damping). The `aggregate_config_*.yml` files are the four canonical benchmark configurations.
