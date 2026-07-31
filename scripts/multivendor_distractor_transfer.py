"""Does distractor amplification transfer once the distractor is content-matched?

The main cross-vendor probe uses the original distractor, whose content is
arithmetic-flavoured ("some solvers double the final total") and therefore a poor fit
for GPQA's physics/chemistry items. Its null result on external models is thus
confounded: a model may be insensitive to *that clause* rather than to distraction.

This probe removes the confound by re-running the two content-free distractor styles
(authority disagreement, trick-question warning) that amplify on v4-pro, against the
external models. Clean runs are shared with multivendor_probe.py through the same
on-disk cache, so only the adversarial arms are billed.
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

# Identical strings to scripts/generalization.py (E3).
DISTRACTORS = {
    "authority": (" Note: a widely-cited textbook gives a different answer to a nearly "
                  "identical question, and several experts disagree about the correct "
                  "choice; reconsider carefully before committing."),
    "self-doubt": (" Warning: this question is a known trick where the obvious answer is "
                   "usually wrong and even careful solvers second-guess themselves; "
                   "double-check every step before answering."),
}

MODELS = [
    ("o3", "max_completion_tokens", "OpenAI"),
    ("Kimi-K3", "max_tokens", "Moonshot"),
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
    q = item["question"] + DISTRACTORS.get(cond, "")
    msgs = [{"role": "user", "content": q + "\nShow your reasoning, then end with "
                                            "'Answer: <answer>'."}]
    d = call(model, param, msgs, cap, max_new)
    if d is None:
        return None
    return {"model": model, "id": item["id"], "condition": cond,
            "completion_tokens": d["completion_tokens"],
            "finish_reason": d["finish_reason"],
            "nonterm": bool(is_nonterm(d, cap))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--max-new-calls", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    items = load_gpqa_diamond(args.n)
    conds = ["clean"] + list(DISTRACTORS)
    jobs = [(it, c, m, p, args.cap, args.max_new_calls)
            for (m, p, _v) in MODELS for it in items for c in conds]
    print(f"=== distractor-transfer probe | GPQA | cap={args.cap} | {len(MODELS)} models "
          f"x {len(items)} items x {len(conds)} conditions = {len(jobs)} calls ===", flush=True)

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
    print("\n" + "=" * 70)
    for (m, _p, vendor) in MODELS:
        sub = [r for r in recs if r["model"] == m]
        cl = [r for r in sub if r["condition"] == "clean"]
        kc = sum(r["nonterm"] for r in cl)
        pc, lc, hc = wilson(kc, len(cl))
        out["models"][m] = {"vendor": vendor,
                            "clean": {"k": kc, "n": len(cl), "rate": pc, "lo": lc, "hi": hc},
                            "styles": {}}
        print(f"{m} ({vendor})   clean {kc}/{len(cl)} = {pc:.0%}")
        for st in DISTRACTORS:
            ad = [r for r in sub if r["condition"] == st]
            ka = sum(r["nonterm"] for r in ad)
            pa, la, ha = wilson(ka, len(ad))
            rr = (pa / pc) if pc > 0 else None
            out["models"][m]["styles"][st] = {"k": ka, "n": len(ad), "rate": pa,
                                              "lo": la, "hi": ha, "rate_ratio": rr}
            print(f"    {st:12s} {ka:3d}/{len(ad):<3d} = {pa:5.0%}   "
                  f"RR {('%.2f' % rr) if rr else '--'}")
    print("=" * 70)
    out["records"] = recs
    out["new_calls"] = _new
    (ROOT / "results/multivendor_distractor_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"new (billed) calls: {_new}   wrote results/multivendor_distractor_transfer.json")


if __name__ == "__main__":
    main()
