"""
Probe a second (cross-provider) reasoning API to test whether the non-termination
phenomenon generalizes beyond DeepSeek. Reads creds from environment only:
    SECOND_API_KEY, SECOND_BASE_URL, SECOND_MODEL
No key is stored in this file. Sends ONE cheap call first to verify connectivity and
inspect the response shape (does it return usage + finish_reason? reasoning content?).

Usage (you pass the key inline so it never lands in the repo):
    SECOND_API_KEY=... SECOND_BASE_URL=... SECOND_MODEL=... python3 scripts/probe_second_model.py
"""
import os, sys, json, time

print("[probe] script started", flush=True)

try:
    from openai import OpenAI
    import openai as _openai_mod
    print(f"[probe] openai SDK version: {getattr(_openai_mod, '__version__', '?')}", flush=True)
except Exception as e:
    print(f"[probe] IMPORT FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)

# Read creds: prefer env vars, else fall back to ~/.second_key (KEY=VALUE lines).
def _load_creds():
    key = os.environ.get("SECOND_API_KEY")
    base = os.environ.get("SECOND_BASE_URL")
    model = os.environ.get("SECOND_MODEL")
    path = os.path.expanduser("~/.second_key")
    if (not key or not base) and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k == "SECOND_API_KEY" and not key:
                    key = v
                elif k == "SECOND_BASE_URL" and not base:
                    base = v
                elif k == "SECOND_MODEL" and not model:
                    model = v
    return key, base, (model or "gpt-5.5")

key, base, model = _load_creds()
print(f"[probe] env: key={'set('+str(len(key))+' chars)' if key else 'MISSING'} "
      f"base={base!r} model={model!r}", flush=True)
if not key or not base:
    print("[probe] ERROR: set SECOND_API_KEY/SECOND_BASE_URL or write ~/.second_key.", flush=True)
    sys.exit(1)

client = OpenAI(api_key=key, base_url=base)
prompt = ("What is 17*23? Show your reasoning step by step, then end with 'Answer: <n>'.")

print(f"[probe] calling model={model} base={base} ...", flush=True)
t0 = time.time()
try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
except Exception as e:
    print(f"[probe] CALL FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)
lat = time.time() - t0
msg = r.choices[0].message
usage = r.usage
print(f"[probe] OK in {lat:.1f}s", flush=True)
print(f"  model returned: {r.model}")
print(f"  finish_reason:  {r.choices[0].finish_reason}")
print(f"  prompt_tokens:  {getattr(usage,'prompt_tokens',None)}")
print(f"  completion_tokens: {getattr(usage,'completion_tokens',None)}")
ctd = getattr(usage, "completion_tokens_details", None)
print(f"  reasoning_tokens: {getattr(ctd,'reasoning_tokens',None) if ctd else 'n/a'}")
print(f"  has reasoning_content: {getattr(msg,'reasoning_content',None) is not None}")
print(f"  text[:120]: {(msg.content or '')[:120]!r}")
print("\n[probe] KEY QUESTION for non-termination: does this API expose finish_reason=='length' "
      "when max_tokens is hit? (that's all we need — reasoning_content is optional)")
