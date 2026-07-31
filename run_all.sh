#!/usr/bin/env bash
# Reproduce every number and figure in the paper from the cached traces.
# NO API calls: all analyses read results/main_records_8k.jsonl, results/main_records.jsonl,
# and results/cache/. Deterministic (all scripts seed numpy with default_rng(0)).
#
# Usage:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/7] non-termination rates + mechanism + reach-then-abandon"
python3 scripts/non_termination_deep.py

echo "==> [2/7] truncation confound: valid-only selective prediction"
python3 scripts/selective_prediction.py results/main_records_8k.jsonl --valid-only

echo "==> [3/7] comparison + ablation (trace signals vs paid baselines)"
python3 scripts/comparison_ablation.py

echo "==> [4/7] rigorous statistics (Fisher, OR, Wilson CI, permutation)"
python3 scripts/rigor_stats.py

echo "==> [5/7] robustness / sensitivity (definitions, marker LOO, seeds)"
python3 scripts/robustness.py

echo "==> [5c] generalization (2nd model, 16k budget, multi-distractor; API, cached)"
python3 scripts/generalization.py --n 30 --n_e2 20 --e2_cap 16384

echo "==> [5b] multi-agent liveness + mitigation (offline pipeline simulation)"
python3 scripts/liveness.py
python3 scripts/two_agent_demo.py --n 60
python3 scripts/hedged_retry_real.py

echo "==> [5d] cross-vendor probes (relay; cap screen first, then the probes)"
python3 scripts/relay_cap_calibration.py
python3 scripts/multivendor_probe.py --n 60
python3 scripts/multivendor_stats.py
python3 scripts/multivendor_distractor_transfer.py --n 60

echo "==> [6/7] figures: pipeline, budget, mechanism, trajectory, early-commit, survival"
python3 scripts/make_flowchart.py
python3 scripts/make_figures_nonterm.py
python3 scripts/make_figures_liveness.py
python3 scripts/make_figure_survival.py
# rigor figures LAST so the CI-error-bar fig_nonterm + forest plot are the final versions
python3 scripts/make_figures_rigor.py

echo "==> [7/7] done. JSON summaries in results/, figures in results/figures/"
ls -1 results/*.json
