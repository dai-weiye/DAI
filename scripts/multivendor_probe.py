"""Multi-vendor non-termination probe (GPQA-Diamond, identical protocol to the main study).

Answers the reviewer question the single cross-provider probe could not: is
distraction-induced non-termination a DeepSeek-family pathology, or does it recur across
training pipelines? We run the same items, the same distractor and the same
gold-independent predicate against reasoning models from several vendors.

Only models that provably honour the token cap are eligible; see
scripts/relay_cap_calibration.py. On a serving stack that ignores max_tokens every run
terminates by construction, so a 0% rate would measure the plumbing rather than the model.

Calls are cached on disk, so re-runs are free and the reported numbers are deterministic.
"""
import os, sys, json, pathlib, argparse, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets_load import load_gpqa_diamond  # noqa: E402
from trace_method import extract_final_answer  # noqa: E402
from llm_client import _load_dotenv  # noqa: E402

_load_dotenv()
BASE, KEY = os.environ["RELAY_BASE_URL"], os.environ["RELAY_API_KEY"]
CACHE = ROOT / "results/cache_multivendor"
CACHE.mkdir(parents=True, exist_ok=True)

# Identical distractor to configs/main_8k.yaml.
DISTRACTOR = (" Note: a similar-looking problem circulated online with a different answer, "
              "and some solvers double the final total by mistake; think very carefully and "
              "do not be misled.")

# (model id, token parameter, vendor label) -- all verified cap-honouring.
MODELS = [
    ("claude-opus-4-8", "max_tokens", "Anthropic"),
    ("o3", "max_completion_tokens", "OpenAI"),
    ("gpt-oss-120b", "max_tokens", "OpenAI (open-weight)"),
    ("Kimi-K3", "max_tokens", "Moonshot"),
    ("MiniMax-M2.7", "max_tokens", "MiniMax"),
]

_lock = threading.Lock()
_new = 0


def call(model, param, messages, cap, max_new):
    global _new
    ck = hashlib.sha256(json.dumps(
        {"m": model, "msg": messages, "cap": cap}, sort_keys=True, ensure_ascii=False
    ).encode()).hexdigest()
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        d = json.loads(cf.read_text()); d["cached"] = True; return d
    with _lock:
        if _new >= max_new:
            return None
        _new += 1
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=600, max_retries=0)
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, **{param: cap})
            u = r.usage
            ctd = getattr(u, "completion_tokens_details", None)
            d = {"text": r.choices[0].message.content or "",
                 "completion_tokens": u.completion_tokens,
                 "reasoning_tokens": (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0,
                 "finish_reason": r.choices[0].finish_reason,
                 "model": r.model, "cached": False}
            cf.write_text(json.dumps(d, ensure_ascii=False))
            return d
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt * 3, 30))
    print(f"    [warn] {model}: {str(last)[:110]}", flush=True)
    return None


def is_nonterm(d, cap):
    return (extract_final_answer(d["text"]) in (None, "")) and d["completion_tokens"] >= cap - 8


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def job(item, cond, model, param, cap, max_new):
    q = item["question"] + (DISTRACTOR if cond == "adversarial" else "")
    msgs = [{"role": "user", "content": q + "\nShow your reasoning, then end with "
                                            "'Answer: <answer>'."}]
    d = call(model, param, msgs, cap, max_new)
    if d is None:
        return None
    return {"model": model, "id": item["id"], "condition": cond,
            "completion_tokens": d["completion_tokens"],
            "reasoning_tokens": d["reasoning_tokens"],
            "finish_reason": d["finish_reason"],
            "nonterm": bool(is_nonterm(d, cap))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--max-new-calls", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    items = load_gpqa_diamond(args.n)
    jobs = [(it, cond, m, p, args.cap, args.max_new_calls)
            for (m, p, _v) in MODELS for it in items for cond in ("clean", "adversarial")]
    print(f"=== multi-vendor probe | GPQA | cap={args.cap} | {len(MODELS)} models "
          f"x {len(items)} items x 2 conditions = {len(jobs)} calls ===", flush=True)

    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                recs.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({_new} new calls)", flush=True)

    out = {"cap": args.cap, "n_requested": args.n, "models": {}}
    print("\n" + "=" * 74)
    print(f"{'model':22s} {'vendor':10s} {'clean':>16s} {'adversarial':>16s} {'ampl.':>7s}")
    print("-" * 74)
    for (m, _p, vendor) in MODELS:
        sub = [r for r in recs if r["model"] == m]
        cl = [r for r in sub if r["condition"] == "clean"]
        ad = [r for r in sub if r["condition"] == "adversarial"]
        kc, ka = sum(r["nonterm"] for r in cl), sum(r["nonterm"] for r in ad)
        (pc, lc, hc) = wilson(kc, len(cl))
        (pa, la, ha) = wilson(ka, len(ad))
        out["models"][m] = {
            "vendor": vendor,
            "clean": {"k": kc, "n": len(cl), "rate": pc, "lo": lc, "hi": hc},
            "adversarial": {"k": ka, "n": len(ad), "rate": pa, "lo": la, "hi": ha},
            "multiplier": (pa / pc) if pc > 0 else None,
            "median_ct_clean": sorted(r["completion_tokens"] for r in cl)[len(cl) // 2] if cl else None,
            "median_ct_adv": sorted(r["completion_tokens"] for r in ad)[len(ad) // 2] if ad else None,
        }
        mult = out["models"][m]["multiplier"]
        print(f"{m:22s} {vendor:10s} {kc:3d}/{len(cl):<3d} {pc:5.0%}    "
              f"{ka:3d}/{len(ad):<3d} {pa:5.0%}    "
              f"{('%.1fx' % mult) if mult else '  --'}")
    print("=" * 74)
    out["records"] = recs
    out["new_calls"] = _new
    (ROOT / "results/multivendor_probe.json").write_text(json.dumps(out, indent=2))
    print(f"new (billed) calls: {_new}   wrote results/multivendor_probe.json")


if __name__ == "__main__":
    main()
