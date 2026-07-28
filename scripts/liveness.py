"""
Multi-agent LIVENESS experiment (offline, $0): how non-termination propagates in a
distributed LLM pipeline.

Motivation for DAI: a single reasoning step that never returns is not just a local
error, it is a *liveness* failure that stalls everything downstream of it. We model a
staged agent pipeline and measure how the empirically-measured per-step non-termination
rate (from our 360 real runs) degrades end-to-end throughput and latency.

Model. A task flows through a chain of K agents (e.g. solver -> verifier -> aggregator).
Each agent is an independent reasoning call that, with probability q (its
non-termination rate), hangs until a wall-clock timeout T instead of returning after a
normal latency. A task COMPLETES only if every stage terminates normally; if any stage
hangs, the task is BLOCKED (in a naive pipeline the whole chain waits out the timeout).

Two regimes are compared, driven by the real clean vs adversarial rates:
  - clean:        q = 0.10 per stage
  - adversarial:  q = 0.33 per stage
End-to-end completion probability is (1-q)^K, so non-termination compounds with pipeline
depth: the deeper the multi-agent system, the more catastrophic distraction becomes.

Metrics (Monte Carlo over N tasks, seeded):
  - completion rate            = fraction of tasks with all K stages terminating
  - blocked rate               = 1 - completion rate
  - mean end-to-end latency    = sum of stage latencies, with hung stages costing T
  - wasted compute fraction    = time spent in stages that ultimately hang

Latencies are sampled from the real latency distribution in logs/calls.jsonl; the hang
timeout T is set to the reasoning cap's observed max latency. No API calls.

Writes results/liveness.json + console tables.
"""
import json, pathlib, statistics
from collections import defaultdict
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
CAP = 8192


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nonterm(r, cap=CAP):
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= cap - 8


def real_rates(recs):
    """Empirical per-stage non-termination rate, clean vs adversarial (pooled)."""
    out = {}
    for cond in ("clean", "adversarial"):
        sub = [r for r in recs if r["condition"] == cond]
        out[cond] = sum(nonterm(r) for r in sub) / len(sub)
    return out


def latency_samples():
    """Normal-return latencies (terminating calls) and the hang timeout T.
    Terminating latency ~ non-length finishes; T ~ the max latency seen (budget-exhausting
    calls run to the cap)."""
    lats_ok, lats_all = [], []
    for l in open(ROOT / "logs/calls.jsonl"):
        d = json.loads(l)
        lat = d.get("lat")
        if lat is None:
            continue
        lats_all.append(lat)
        # a terminating call is short; the reasoning cap (8192) calls are the long tail
        if d.get("ct", 0) < CAP - 8:
            lats_ok.append(lat)
    T = max(lats_all) if lats_all else 120.0        # hang timeout = worst observed wall-clock
    return np.array(lats_ok), T


def simulate(q, K, N, ok_lats, T):
    """Monte Carlo N tasks through a K-stage chain; each stage hangs w.p. q (costs T),
    else returns after a sampled normal latency. Task completes iff all stages return."""
    completed = 0
    e2e_lat = []
    wasted = []
    for _ in range(N):
        total = 0.0; waste = 0.0; ok = True
        for _ in range(K):
            if RNG.random() < q:          # this stage hangs
                total += T; waste += T; ok = False
                # naive pipeline: downstream still waits; we stop accumulating useful work
                # but keep charging the timeout for this blocked stage only (already added)
                break
            else:
                total += ok_lats[RNG.integers(0, len(ok_lats))]
        if ok:
            completed += 1
        e2e_lat.append(total)
        wasted.append(waste)
    return {"completion_rate": completed / N,
            "blocked_rate": 1 - completed / N,
            "mean_latency": float(np.mean(e2e_lat)),
            "p95_latency": float(np.percentile(e2e_lat, 95)),
            "wasted_fraction": float(np.sum(wasted) / np.sum(e2e_lat)) if np.sum(e2e_lat) else 0.0}


def simulate_mitigated(q, K, N, ok_lats, T, t_kill, max_retries):
    """MITIGATION: hedged timeout + resampling retry (cf. Dean & Barroso, 'The Tail at
    Scale'). If a stage does not return within a short deadline t_kill (<< T), abort it
    and retry with a fresh sample. ASSUMPTION: retries resample at temperature>0 so each
    attempt hangs independently with probability q; a stage then succeeds within
    (max_retries+1) attempts with probability 1-q^(max_retries+1). This is the honest
    best case for retry; correlated failures would reduce the gain."""
    completed = 0
    e2e_lat = []
    for _ in range(N):
        total = 0.0; ok = True
        for _ in range(K):
            stage_done = False
            for attempt in range(max_retries + 1):
                if RNG.random() < q:               # attempt hangs -> killed at t_kill
                    total += t_kill                # abort early instead of waiting full T
                else:                               # attempt returns
                    total += ok_lats[RNG.integers(0, len(ok_lats))]
                    stage_done = True
                    break
            if not stage_done:
                ok = False
                break
        if ok:
            completed += 1
        e2e_lat.append(total)
    return {"completion_rate": completed / N,
            "blocked_rate": 1 - completed / N,
            "mean_latency": float(np.mean(e2e_lat)),
            "p95_latency": float(np.percentile(e2e_lat, 95))}


def main():
    recs = load(ROOT / "results/main_records_8k.jsonl")
    rates = real_rates(recs)
    ok_lats, T = latency_samples()
    N = 20000

    print(f"=== Multi-agent liveness (offline; q from real data, N={N}) ===")
    print(f"  per-stage non-termination rate: clean q={rates['clean']:.2f}, "
          f"adversarial q={rates['adversarial']:.2f}")
    print(f"  normal-return latency: median {np.median(ok_lats):.1f}s; hang timeout T={T:.0f}s\n")

    out = {"rates": rates, "timeout_s": T, "median_ok_latency_s": float(np.median(ok_lats)),
           "N": N, "pipelines": {}}

    print(f"  {'depth K':>8}{'cond':>13}{'completion':>12}{'blocked':>10}"
          f"{'mean lat(s)':>13}{'wasted%':>9}")
    for K in (1, 2, 3, 5):
        for cond in ("clean", "adversarial"):
            s = simulate(rates[cond], K, N, ok_lats, T)
            out["pipelines"][f"K{K}_{cond}"] = s
            print(f"  {K:>8}{cond:>13}{s['completion_rate']:>12.1%}{s['blocked_rate']:>10.1%}"
                  f"{s['mean_latency']:>13.1f}{s['wasted_fraction']:>9.1%}")

    # headline: completion collapse with depth under distraction
    print("\n  Completion probability (1-q)^K compounds with pipeline depth:")
    for cond in ("clean", "adversarial"):
        q = rates[cond]
        chain = "  ".join(f"K{K}:{(1-q)**K:.0%}" for K in (1, 2, 3, 5))
        print(f"    {cond:12} {chain}")
    k3 = out["pipelines"]
    drop = k3["K3_clean"]["completion_rate"] - k3["K3_adversarial"]["completion_rate"]
    out["headline"] = {
        "clean_K3_completion": k3["K3_clean"]["completion_rate"],
        "adv_K3_completion": k3["K3_adversarial"]["completion_rate"],
        "completion_drop_K3": drop,
        "adv_K3_blocked": k3["K3_adversarial"]["blocked_rate"]}
    print(f"\n  At depth 3, distraction drops end-to-end completion from "
          f"{k3['K3_clean']['completion_rate']:.0%} to {k3['K3_adversarial']['completion_rate']:.0%} "
          f"({drop:.0%} absolute); {k3['K3_adversarial']['blocked_rate']:.0%} of tasks stall.")

    # ---- mitigation: hedged timeout + resampling retry ----
    t_kill = float(np.percentile(ok_lats, 95))     # kill a stage at the 95th pct of normal latency
    print(f"\n=== Mitigation: hedged timeout + retry (t_kill={t_kill:.0f}s, <<T={T:.0f}s) ===")
    print(f"  {'depth K':>8}{'cond':>13}{'naive compl.':>14}{'mitigated':>11}{'recovered':>11}")
    out["mitigation"] = {"t_kill_s": t_kill, "max_retries": 2}
    for K in (2, 3, 5):
        for cond in ("clean", "adversarial"):
            naive = out["pipelines"][f"K{K}_{cond}"]["completion_rate"]
            mit = simulate_mitigated(rates[cond], K, N, ok_lats, T, t_kill, max_retries=2)
            rec = mit["completion_rate"] - naive
            out["mitigation"][f"K{K}_{cond}"] = {**mit, "naive_completion": naive,
                                                 "recovered": rec}
            print(f"  {K:>8}{cond:>13}{naive:>14.1%}{mit['completion_rate']:>11.1%}"
                  f"{rec:>+11.1%}")
    m3 = out["mitigation"]["K3_adversarial"]
    print(f"\n  At depth 3 under distraction, retry lifts completion "
          f"{m3['naive_completion']:.0%} -> {m3['completion_rate']:.0%} "
          f"({m3['recovered']:+.0%}), and caps per-stage stall at t_kill instead of T.")
    print("  Caveat: assumes retries resample independently (temperature>0); correlated "
          "hangs would reduce this. Retry does not fix the model, only the pipeline's liveness.")

    json.dump(out, open(ROOT / "results/liveness.json", "w"), indent=2)
    print(f"\nwrote {ROOT/'results/liveness.json'}")


if __name__ == "__main__":
    main()
