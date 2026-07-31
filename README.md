# When Reasoning Fails to Stop — Reproducibility

Code and data for the DAI 2026 submission on **distraction-induced non-termination** in
reasoning LLMs. Everything runs on CPU; no GPU or training. Every API call is cached
(SHA-256 of the request), and **all analyses in the paper are computed offline from that
cache at zero additional cost**.

## Environment
- Python 3.13; packages: `openai numpy scipy matplotlib pyyaml`
- API keys only needed to regenerate the cache (not to reproduce the analyses). In `../.env`:
  ```
  DEEPSEEK_API_KEY=sk-...
  DEEPSEEK_BASE_URL=https://api.deepseek.com
  # Only for the cross-vendor probes: any OpenAI-compatible relay
  RELAY_API_KEY=sk-...
  RELAY_BASE_URL=https://<relay-host>/v1
  ```

## Data (see ../data/PROVENANCE.md for licenses)
GSM8K test (MIT), MATH-500 (MIT), GPQA-Diamond (CC BY 4.0; do not redistribute — carries a
canary string, download from the official repo).

## One-command reproduction (offline, $0)
From this `code/` directory:
```bash
bash run_all.sh
```
This reruns every analysis and regenerates every figure from the cached traces
(`results/main_records_8k.jsonl`, `results/main_records.jsonl`, `results/cache/`).
It is deterministic: all scripts seed numpy with `default_rng(0)`.

## What maps to which paper claim
| Paper element | Script | Output |
|---|---|---|
| Non-termination rates, mechanism, reach-then-abandon | `scripts/non_termination_deep.py` | `results/non_termination_deep.json` |
| Truncation confound (valid-only AUROC 0.46) | `scripts/selective_prediction.py --valid-only` | `results/selective_summary.json` |
| Ablation (budget×population) + comparison vs baselines | `scripts/comparison_ablation.py` | `results/comparison_ablation.json` |
| Significance tests, odds ratios, Wilson/bootstrap CIs, permutation | `scripts/rigor_stats.py` | `results/rigor_stats.json` |
| Robustness (definition, marker LOO, seed stability) | `scripts/robustness.py` | `results/robustness.json` |
| Fig 1 pipeline | `scripts/make_flowchart.py` (TikZ source: `scripts/fig_pipeline.tex`, needs `pdflatex`) | `results/figures/fig_pipeline.pdf` |
| Non-termination bar (Wilson CIs), budget, mechanism | `scripts/make_figures_nonterm.py` | `results/figures/fig_{nonterm,budget,mechanism}.pdf` |
| Odds-ratio forest plot, CI bar chart | `scripts/make_figures_rigor.py` | `results/figures/fig_{forest,nonterm}.pdf` |
| Large-sample replication (n=1195) | `scripts/expand_run.py`, `scripts/expand_analyze.py` | `results/expand_summary.json` |
| Kaplan–Meier survival + budget extrapolation (Fig. 3) | `scripts/make_figure_survival.py` | `results/figures/fig_survival.pdf`, `results/survival_summary.json` |
| Cap-honouring screen for relayed models (App. C) | `scripts/relay_cap_calibration.py` | `results/relay_cap_calibration.json` |
| Cross-vendor probe, 7 models / 6 vendors (Table 3) | `scripts/multivendor_probe.py --n 60`, `scripts/multivendor_stats.py` | `results/multivendor_{probe,stats}.json` |
| Distractor-content-mismatch control on `o3` (App. D) | `scripts/multivendor_distractor_transfer.py --n 60` | `results/multivendor_distractor_transfer.json` |
| Real two-agent pipeline (n=60) | `scripts/two_agent_demo.py --n 60` | `results/two_agent.json` |
| Measured hedged-retry escape rates | `scripts/hedged_retry_real.py` | `results/hedged_retry_real.json` |

## Shared utilities
`src/analysis_utils.py` centralizes the common primitives (`auroc`, `auroc_ci`, `nonterm`,
`zscore_within`, `wilson`, `boot_ci`) used by the new analysis scripts; each is seeded and
regression-checked against the paper's headline numbers.

## Regenerating the cache (optional; costs API $)
```bash
LLM_CACHE_DIR=results/cache python3 scripts/run_main.py \
    --config configs/main_8k.yaml --datasets gsm8k math500 gpqa_diamond \
    --conditions clean adversarial --out results/main_records_8k.jsonl
```
- `configs/main_8k.yaml`: reasoner `deepseek-v4-pro` (exposes `reasoning_content` +
  `reasoning_tokens`), 8192-token budget, distractor text, seed 0, temperature 0.
- A hard `MAX_SPEND_USD` guard aborts before overspending. Full 8k study cost ≈ $1.11;
  every experiment in the paper together cost $4.20 on the DeepSeek endpoint plus 775
  relayed calls for the cross-vendor probes.

Cross-vendor calls are cached separately in `results/cache_multivendor/`. Only models that
provably honour `max_tokens` are eligible — run `scripts/relay_cap_calibration.py` first,
since on a stack that ignores the cap every run terminates by construction and a 0%
non-termination rate would measure the plumbing rather than the model.

## Determinism
Given the cache, every reported number is bit-reproducible. `src/llm_client.py` keys the
cache on (model, messages, temperature, max_tokens, top_p, seed, sample_idx); statistical
tests use fixed RNG seeds.
