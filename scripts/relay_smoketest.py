"""
Minimal one-shot connectivity test for the relay API. NO retries, NO exception
swallowing, NO threads -- it prints exactly what happens on a single call so we can
see the real error (bad base_url, wrong path, model name, auth, etc.).

Run:
  OPENAI_BASE_URL=https://api.v36.cm/v1 OPENAI_API_KEY=sk-... \
    python3 scripts/relay_smoketest.py --model claude-opus-4-8
"""
import os, sys, argparse, traceback
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="claude-opus-4-8")
args = ap.parse_args()

base = os.environ.get("OPENAI_BASE_URL")
key = os.environ.get("OPENAI_API_KEY")
print(f"base_url = {base!r}")
print(f"key set  = {bool(key)} (len={len(key) if key else 0})")
print(f"model    = {args.model!r}")
print("-" * 50, flush=True)

if not base or not key:
    print("ERROR: set OPENAI_BASE_URL and OPENAI_API_KEY"); sys.exit(1)

client = OpenAI(api_key=key, base_url=base, timeout=30.0, max_retries=0)

# Try the simplest possible call, with max_tokens; on failure, show the FULL error.
for tok_param in ("max_tokens", "max_completion_tokens"):
    print(f"\n>>> trying with {tok_param}=64 ...", flush=True)
    try:
        kw = {"model": args.model,
              "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
              "temperature": 0.0, tok_param: 64}
        r = client.chat.completions.create(**kw)
        print("SUCCESS")
        print("  model returned:", r.model)
        print("  finish_reason :", r.choices[0].finish_reason)
        print("  content       :", repr(r.choices[0].message.content))
        print("  usage         :", r.usage)
        print("\n>>> API works. You can run the full probe now.")
        sys.exit(0)
    except Exception as e:
        print(f"  FAILED with {tok_param}:")
        print("  " + "".join(traceback.format_exception_only(type(e), e)).strip())
        # print HTTP body if present (relay error messages live here)
        for attr in ("status_code", "code", "message"):
            v = getattr(e, attr, None)
            if v is not None:
                print(f"    {attr} = {v!r}")
        body = getattr(e, "response", None)
        if body is not None:
            try:
                print("    body =", body.text[:400])
            except Exception:
                pass

print("\n>>> Both attempts failed. The error above tells us what to fix "
      "(base_url path, model id, or auth).")
