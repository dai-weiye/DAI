"""Smoke test for relay-served models before spending on the cross-vendor probe.

For each candidate model we check the two things the non-termination predicate needs:
  1. a normal call returns content and usage accounting;
  2. a deliberately tiny cap produces finish_reason='length' with the cap actually
     honoured, so a truncated run is distinguishable from a finished one.

A relay that silently ignores max_tokens, or that reports finish_reason='stop' on a
truncated completion, cannot support the probe and is excluded.
"""
import os, sys, json, pathlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import _load_dotenv  # noqa: E402  (reuses the repo's .env loader)

_load_dotenv()
BASE = os.environ["RELAY_BASE_URL"]
KEY = os.environ["RELAY_API_KEY"]

CANDIDATES = [
    "gemini-3-pro-preview-thinking",
    "gpt-5.4",
    "o4-mini",
    "qwen3-235b-a22b-thinking-2507",
    "grok-4",
    "glm-5",
    "claude-opus-4-8",
    "deepseek-v4-pro",
]

Q = ("A gas expands adiabatically and reversibly from 2.0 L to 6.0 L. "
     "For a monatomic ideal gas, by what factor does the temperature change? "
     "Show your reasoning, then end with 'Answer: <answer>'.")


def call(client, model, cap):
    for tok_param in ("max_tokens", "max_completion_tokens"):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": Q}],
                temperature=0.0, **{tok_param: cap})
            u = r.usage
            ctd = getattr(u, "completion_tokens_details", None)
            return {"ok": True, "param": tok_param, "model": r.model,
                    "ct": u.completion_tokens,
                    "rt": (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0,
                    "finish": r.choices[0].finish_reason,
                    "len_text": len(r.choices[0].message.content or "")}
        except Exception as e:
            err = str(e)
            if tok_param == "max_tokens" and "max_tokens" in err:
                continue
            return {"ok": False, "err": err[:160]}
    return {"ok": False, "err": "both token params rejected"}


def probe(model):
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=300, max_retries=1)
    return model, {"normal": call(client, model, 8192), "tiny": call(client, model, 64)}


def main():
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for model, res in ex.map(probe, CANDIDATES):
            out[model] = res
            n, t = res["normal"], res["tiny"]
            if not n["ok"]:
                print(f"[FAIL] {model:34s} {n['err']}", flush=True)
                continue
            honoured = t["ok"] and t["ct"] <= 96 and t["finish"] == "length"
            print(f"[{'OK ' if honoured else 'CHK'}] {model:34s} "
                  f"param={n['param']:22s} ct={n['ct']:<6} rt={n['rt']:<6} "
                  f"finish={n['finish']:<8} | tiny: ct={t.get('ct')} "
                  f"finish={t.get('finish')} textlen={t.get('len_text')}", flush=True)
    pathlib.Path("results/relay_smoke.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
