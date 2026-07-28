"""
Reproducible, cached LLM client for DeepSeek (OpenAI-compatible).

Design goals (AAAI reproducibility):
- Deterministic given a fixed cache: every (model, messages, params, seed, sample-index)
  maps to one on-disk record. Re-runs are free and identical.
- Full accounting: every call logs prompt/completion/reasoning tokens + latency.
- Captures reasoning_content and reasoning_tokens for reasoning models (v4-pro).
- No secret in code: key read from environment / .env only.

This file contains NO experimental logic — it is pure infrastructure.
"""
from __future__ import annotations
import os, json, time, hashlib, pathlib, threading
from dataclasses import dataclass, asdict, field
from typing import Any

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise RuntimeError("pip install openai") from e


def _load_dotenv(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        # also try repo root relative to this file
        p = pathlib.Path(__file__).resolve().parents[2] / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

CACHE_DIR = pathlib.Path(os.environ.get("LLM_CACHE_DIR", "results/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = pathlib.Path(os.environ.get("LLM_LOG_PATH", "logs/calls.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_client = None
_client_lock = threading.Lock()
_log_lock = threading.Lock()

# --- Pricing (USD per 1M tokens), from DeepSeek docs 2026-07 ---
# input=(cache_hit, cache_miss), output
PRICE = {
    "deepseek-v4-flash": {"in_hit": 0.0028, "in_miss": 0.14, "out": 0.28},
    "deepseek-v4-pro":   {"in_hit": 0.003625, "in_miss": 0.435, "out": 0.87},
}

# Budget guard: hard stop before exceeding this (USD of NEW spend this process).
MAX_SPEND_USD = float(os.environ.get("MAX_SPEND_USD", "3.5"))
_spend_lock = threading.Lock()
_spend_usd = 0.0  # cumulative NEW (uncached) spend this process


def _price_call(model, prompt_tok, completion_tok, cache_hit_tok=0):
    p = PRICE.get(model, PRICE["deepseek-v4-flash"])
    miss = max(prompt_tok - cache_hit_tok, 0)
    return (cache_hit_tok * p["in_hit"] + miss * p["in_miss"]
            + completion_tok * p["out"]) / 1e6


def get_spend() -> float:
    return _spend_usd


def _get_client() -> "OpenAI":
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                key = os.environ.get("DEEPSEEK_API_KEY")
                base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
                if not key:
                    raise RuntimeError("DEEPSEEK_API_KEY not set (check .env)")
                _client = OpenAI(api_key=key, base_url=base)
    return _client


@dataclass
class LLMResponse:
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cached: bool
    latency_s: float
    model: str
    finish_reason: str | None = None
    raw_key: str = ""


def _cache_key(model, messages, temperature, max_tokens, top_p, seed, sample_idx) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
            "sample_idx": sample_idx,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _log(record: dict) -> None:
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(
    messages: list[dict],
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    seed: int | None = 0,
    sample_idx: int = 0,
    max_retries: int = 5,
    use_cache: bool = True,
) -> LLMResponse:
    """One chat completion. sample_idx distinguishes repeated samples at temp>0
    so each draw is cached separately and reproducibly."""
    key = _cache_key(model, messages, temperature, max_tokens, top_p, seed, sample_idx)
    cache_file = CACHE_DIR / f"{key}.json"

    if use_cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        d["cached"] = True
        d["raw_key"] = key
        return LLMResponse(**d)

    client = _get_client()
    # Budget guard: refuse to start a new (uncached) call if we are already at cap.
    global _spend_usd
    with _spend_lock:
        if _spend_usd >= MAX_SPEND_USD:
            raise RuntimeError(
                f"Budget cap reached: spent ${_spend_usd:.4f} >= ${MAX_SPEND_USD} "
                f"(set MAX_SPEND_USD to raise). Cached calls still allowed.")
    kwargs: dict[str, Any] = dict(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, top_p=top_p,
    )
    if seed is not None:
        kwargs["seed"] = seed

    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            msg = r.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)
            usage = r.usage
            rt = 0
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                rt = getattr(ctd, "reasoning_tokens", 0) or 0
            resp = LLMResponse(
                text=msg.content or "",
                reasoning=reasoning,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                reasoning_tokens=rt,
                total_tokens=usage.total_tokens,
                cached=False,
                latency_s=round(latency, 3),
                model=r.model,
                finish_reason=r.choices[0].finish_reason,
                raw_key=key,
            )
            if use_cache:
                d = asdict(resp)
                cache_file.write_text(json.dumps(d, ensure_ascii=False))
            cache_hit_tok = getattr(getattr(usage, "prompt_tokens_details", None),
                                    "cached_tokens", 0) or 0
            cost = _price_call(r.model, usage.prompt_tokens, usage.completion_tokens,
                               cache_hit_tok)
            with _spend_lock:
                _spend_usd += cost
            _log({"key": key, "model": r.model, "pt": usage.prompt_tokens,
                  "ct": usage.completion_tokens, "rt": rt, "lat": resp.latency_s,
                  "temp": temperature, "sample_idx": sample_idx,
                  "cost_usd": round(cost, 6), "cum_spend_usd": round(_spend_usd, 4)})
            return resp
        except Exception as e:  # noqa
            last_err = e
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


if __name__ == "__main__":
    # smoke test (uses cache after first run -> free)
    for m in ["deepseek-v4-flash", "deepseek-v4-pro"]:
        r = chat([{"role": "user", "content": "What is 17*23? End with 'Answer: <n>'."}],
                 model=m, max_tokens=300)
        print(f"[{m}] cached={r.cached} ct={r.completion_tokens} rt={r.reasoning_tokens} "
              f"text={r.text[:60]!r}")
