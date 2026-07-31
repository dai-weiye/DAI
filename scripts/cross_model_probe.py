"""
Cross-model non-termination probe (reviewer Cons-2: generalization beyond DeepSeek).

Runs the EXACT same paradigm as scripts/generalization.py -- same GPQA-Diamond items,
same original distractor template, same gold-independent non-termination criterion
(empty final answer AND completion_tokens >= cap-8, read from the API's own usage
accounting) -- but against a NON-DeepSeek model served over an OpenAI-compatible relay.

This isolates whether distraction-induced non-termination is a DeepSeek-specific
pathology or reproduces on a different vendor's reasoning model.

Design:
- Self-contained client (does NOT import llm_client, so the DeepSeek reproducible cache
  is never touched). Separate on-disk cache dir; re-runs are free.
- Hard MAX_SPEND guard is unavailable for third-party relays (no reliable pricing), so we
  instead cap the number of NEW (uncached) calls with --max-new-calls.
- reasoning_tokens may be absent for non-DeepSeek models; the non-termination test does
  not use it (only completion_tokens + empty answer), so the core result is unaffected.

Usage (run yourself with the ! prefix so the key/spend stay under your control):
  OPENAI_BASE_URL=https://api.v36.cm OPENAI_API_KEY=sk-... \
    python3 scripts/cross_model_probe.py --model claude-opus-4-8 --n 20
"""
import sys, os, json, pathlib, argparse, time, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from datasets_load import load_gpqa_diamond
from trace_method import extract_final_answer
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(os.environ.get("XMODEL_CACHE_DIR", str(ROOT / "results/cache_xmodel")))
CACHE.mkdir(parents=True, exist_ok=True)
WORKERS = 6

# Identical original distractor to the main study (configs/main_8k.yaml).
DISTRACTOR = (" Note: a similar-looking problem circulated online with a different answer, "
             "and some solvers double the final total by mistake; think very carefully and "
             "do not be misled.")

_client = None
_new_calls = 0


def client():
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("RELAY_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("RELAY_BASE_URL")
        if not key or not base:
            raise SystemExit("set OPENAI_API_KEY and OPENAI_BASE_URL")
        _client = OpenAI(api_key=key, base_url=base)
    return _client


def _key(model, messages, cap):
    return hashlib.sha256(json.dumps(
        {"m": model, "msg": messages, "cap": cap}, sort_keys=True, ensure_ascii=False
    ).encode()).hexdigest()


def call(model, messages, cap, max_new):
    """Cached chat call. Tries max_tokens; retries with max_completion_tokens for models
    (o-series / some relays) that reject max_tokens. Returns dict or None on hard failure."""
    global _new_calls
    ck = _key(model, messages, cap)
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        d = json.loads(cf.read_text()); d["cached"] = True; return d
    if _new_calls >= max_new:
        return None  # budget guard: stop issuing NEW calls
    _new_calls += 1
    last = None
    for attempt in range(4):
        for tok_param in ("max_tokens", "max_completion_tokens"):
            try:
                kw = {"model": model, "messages": messages, "temperature": 0.0,
                      tok_param: cap}
                r = client().chat.completions.create(**kw)
                u = r.usage
                msg = r.choices[0].message
                rt = 0
                ctd = getattr(u, "completion_tokens_details", None)
                if ctd is not None:
                    rt = getattr(ctd, "reasoning_tokens", 0) or 0
                d = {"text": msg.content or "",
                     "completion_tokens": u.completion_tokens,
                     "reasoning_tokens": rt,
                     "finish_reason": r.choices[0].finish_reason,
                     "model": r.model, "cached": False}
                cf.write_text(json.dumps(d, ensure_ascii=False))
                return d
            except Exception as e:
                last = e
                if "max_tokens" in str(e) and tok_param == "max_tokens":
                    continue  # try max_completion_tokens
                break
        time.sleep(min(2 ** attempt, 20))
    print(f"    [warn] call failed: {last}", flush=True)
    return None


def is_nonterm(d, cap):
    """Gold-independent, model-agnostic: empty extracted answer AND budget exhausted."""
    return (extract_final_answer(d["text"]) in (None, "")) and d["completion_tokens"] >= cap - 8


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0, c - h), min(1, c + h)


def run_one(item, condition, model, cap, max_new):
    q = item["question"] + (DISTRACTOR if condition == "adversarial" else "")
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    d = call(model, [{"role": "user", "content": prompt}], cap, max_new)
    if d is None:
        return None
    return {"id": item["id"], "condition": condition,
            "completion_tokens": d["completion_tokens"],
            "reasoning_tokens": d["reasoning_tokens"],
            "finish_reason": d["finish_reason"],
            "nonterm": bool(is_nonterm(d, cap)), "cached": d.get("cached", False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="non-DeepSeek model id on the relay")
    ap.add_argument("--n", type=int, default=20, help="GPQA items per condition")
    ap.add_argument("--cap", type=int, default=8192, help="token budget (match main run)")
    ap.add_argument("--max-new-calls", type=int, default=60,
                    help="hard cap on NEW (uncached) API calls this run")
    args = ap.parse_args()

    items = load_gpqa_diamond(args.n)
    if not items:
        raise SystemExit("no GPQA data (expected code/gpqa_diamond.csv)")
    jobs = [(it, cond, args.model, args.cap, args.max_new_calls)
            for it in items for cond in ("clean", "adversarial")]
    print(f"=== cross-model probe: {args.model} | GPQA | cap={args.cap} | "
          f"{len(items)} items x 2 conditions = {len(jobs)} calls ===", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for i, f in enumerate(futs, 1):
            r = f.result()
            if r is not None:
                results.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done ({_new_calls} new calls)", flush=True)

    got = len(results)
    if got < len(jobs):
        print(f"  [note] {len(jobs)-got} calls skipped (budget guard or failure); "
              f"reporting on {got} completed.", flush=True)

    def rate(cond):
        sub = [x for x in results if x["condition"] == cond]
        k = sum(x["nonterm"] for x in sub)
        return wilson(k, len(sub)), k, len(sub)

    (pc, pcl, pch), kc, nc = rate("clean")
    (pa, pal, pah), ka, na = rate("adversarial")
    mult = (pa / pc) if pc > 0 else float("inf")
    out = {"model": args.model, "cap": args.cap, "n_per_condition": args.n,
           "clean": {"rate": pc, "lo": pcl, "hi": pch, "k": kc, "n": nc},
           "adversarial": {"rate": pa, "lo": pal, "hi": pah, "k": ka, "n": na},
           "multiplier": mult, "new_calls": _new_calls, "records": results}
    outp = ROOT / f"results/cross_model_{args.model.replace('/', '_')}.json"
    outp.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 60)
    print(f"MODEL: {args.model}  (GPQA-Diamond, cap={args.cap})")
    print(f"  clean       non-termination: {pc:.0%}  [{pcl:.0%},{pch:.0%}]  ({kc}/{nc})")
    print(f"  adversarial non-termination: {pa:.0%}  [{pal:.0%},{pah:.0%}]  ({ka}/{na})")
    print(f"  amplification: {mult:.1f}x   (DeepSeek v4-pro was 23%->77% ~3.3x on GPQA)")
    print(f"  NEW (billed) calls this run: {_new_calls}")
    print(f"  wrote {outp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
