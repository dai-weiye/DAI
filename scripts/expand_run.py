#!/usr/bin/env python3
"""
Expanded-sample runner via the relay, kept INTERNALLY HOMOGENEOUS: every item here
(including a re-run of the original 60-per-cell items) is queried through the SAME
relay endpoint and SAME model string, so the expanded dataset is self-consistent and
not a mixture of the relay with the original api.deepseek.com run. The original 360
records remain the paper's primary result; this larger set is an independent
replication + extension reported alongside it.

Reads results/expand_prompts.jsonl (1196 item-conditions). For each, one greedy pass
at the 8192 budget. Non-termination is gold-independent: empty answer AND
completion_tokens >= cap-8. Resumes automatically (skips items already in the output).
All progress goes to a log FILE (this shell swallows stdout); read the file to monitor.

Run:
  cd code && python3 scripts/expand_run.py
"""
import sys, os, json, pathlib, time, re, hashlib, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from grade import grade

BASE = os.environ.get("OPENAI_BASE_URL", "https://api.v36.cm/v1")
KEY = os.environ["OPENAI_API_KEY"]
MODEL = "deepseek-v4-pro"
CAP = 8192
WORKERS = 8

IN = HERE / "results/expand_prompts.jsonl"
OUT = HERE / "results/expand_records.jsonl"
LOG = HERE / "results/expand_run.log"
CACHE = HERE / "results/cache_expand"
CACHE.mkdir(parents=True, exist_ok=True)


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def extract_answer(text):
    if not text:
        return None
    m = re.findall(r'[Aa]nswer:\s*([^\n]+)', text)
    return m[-1].strip() if m else None


def call(prompt):
    ck = hashlib.sha256((MODEL + "|" + str(CAP) + "|" + prompt).encode()).hexdigest()
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        d = json.loads(cf.read_text()); d["cached"] = True; return d
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": CAP}).encode()
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=240))
            u = r.get("usage", {})
            d = {"content": r["choices"][0]["message"].get("content") or "",
                 "finish_reason": r["choices"][0].get("finish_reason"),
                 "completion_tokens": u.get("completion_tokens", 0),
                 "model": r.get("model"), "cached": False}
            cf.write_text(json.dumps(d, ensure_ascii=False))
            return d
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 20))
    log(f"FAIL after retries: {last}")
    return None


def done_keys():
    if not OUT.exists():
        return set()
    ks = set()
    for line in open(OUT):
        try:
            r = json.loads(line); ks.add((r["dataset"], r["id"], r["condition"]))
        except Exception:
            pass
    return ks


def main():
    jobs = [json.loads(l) for l in open(IN) if l.strip()]
    have = done_keys()
    todo = [j for j in jobs if (j["dataset"], j["id"], j["condition"]) not in have]
    log(f"START total={len(jobs)} done={len(have)} todo={len(todo)} model={MODEL} base={BASE}")

    def work(j):
        d = call(j["prompt"])
        if d is None:
            return None
        ans = extract_answer(d["content"])
        ct = d["completion_tokens"]
        nonterm = (ans in (None, "")) and ct >= CAP - 8
        return {"id": j["id"], "dataset": j["dataset"], "condition": j["condition"],
                "gold": j["gold"], "full_ans": ans,
                "completion_tokens": ct, "finish_reason": d["finish_reason"],
                "correct_full": bool(grade(ans, j["gold"], j.get("gold_type", "number"))),
                "nonterm": bool(nonterm), "model": d.get("model")}

    n = 0
    with open(OUT, "a") as fout, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, j): j for j in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                n += 1
                if n % 25 == 0:
                    log(f"progress {n}/{len(todo)} (last {rec['dataset']}/{rec['condition']} nt={rec['nonterm']})")
    log(f"DONE wrote {n} new records to {OUT}")


if __name__ == "__main__":
    main()
