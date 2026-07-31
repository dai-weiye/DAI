"""Verify that a relay-served model actually honours the token cap.

The non-termination predicate is only meaningful if the serving stack enforces the
requested budget and reports finish_reason='length' when it is hit. Some relays accept
max_tokens and silently ignore it; on such a model every run terminates by construction
and a 0% non-termination rate would be an artifact of the plumbing, not a property of
the model. This script screens candidates before we spend anything on the real probe.

A model passes if, on a prompt long enough to overrun the cap, completion_tokens lands
at (or just under) the cap and finish_reason='length'.
"""
import os, sys, json, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import _load_dotenv  # noqa: E402

_load_dotenv()
BASE, KEY = os.environ["RELAY_BASE_URL"], os.environ["RELAY_API_KEY"]

# Deliberately overrunnable: a long multi-step derivation no model finishes in 256 tokens.
HARD = ("Derive, from first principles and showing every algebraic step, the partition "
        "function of a 3D quantum harmonic oscillator, then obtain the heat capacity in "
        "both the high- and low-temperature limits, then repeat the derivation for a "
        "2D oscillator and compare. Show all work, then end with 'Answer: <answer>'.")

CANDIDATES = [
    ("o4-mini", "max_completion_tokens"),
    ("o3", "max_completion_tokens"),
    ("gpt-5.4", "max_completion_tokens"),
    ("claude-opus-4-8", "max_tokens"),
    ("qwen3-235b-a22b-thinking-2507", "max_tokens"),
    ("glm-5", "max_tokens"),
    ("Kimi-K3", "max_tokens"),
    ("MiniMax-M2.7", "max_tokens"),
    ("gemini-3-pro-preview-thinking", "max_tokens"),
    # Second screening round: an open-weight non-Chinese reasoner is the one cell the
    # first round left empty (o3/claude are closed, Kimi/MiniMax are open but Chinese).
    ("gpt-oss-120b", "max_tokens"),
    ("grok-4", "max_tokens"),
    ("grok-4.2", "max_tokens"),
    ("grok-3-reasoner", "max_tokens"),
    ("gemini-3.1-pro-preview-thinking", "max_tokens"),
]
CAPS = [256, 1024]


def one(args):
    model, param = args
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=150, max_retries=0)
    rows = []
    for cap in CAPS:
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": HARD}],
                temperature=0.0, **{param: cap})
            rows.append({"cap": cap, "ct": r.usage.completion_tokens,
                         "finish": r.choices[0].finish_reason,
                         "textlen": len(r.choices[0].message.content or "")})
        except Exception as e:
            rows.append({"cap": cap, "err": str(e)[:120]})
    # honoured := every cap enforced within a small slack and signalled as 'length'
    ok = all("err" not in x and x["ct"] <= x["cap"] * 1.05 + 8 and x["finish"] == "length"
             for x in rows)
    return model, {"param": param, "honours_cap": ok, "probes": rows}


def main():
    out = {}
    with ThreadPoolExecutor(max_workers=len(CANDIDATES)) as ex:
        futs = {ex.submit(one, c): c[0] for c in CANDIDATES}
        for fut in as_completed(futs):
            model, res = fut.result()
            out[model] = res
            detail = "  ".join(
                f"cap{x['cap']}:ct={x.get('ct', 'ERR')},{x.get('finish', x.get('err', '')[:28])}"
                for x in res["probes"])
            print(f"[{'PASS' if res['honours_cap'] else 'FAIL'}] {model:32s} {detail}",
                  flush=True)
    pathlib.Path("results/relay_cap_calibration.json").write_text(json.dumps(out, indent=2))
    passed = [m for m, v in out.items() if v["honours_cap"]]
    print("\nusable for the probe:", passed)


if __name__ == "__main__":
    main()
