"""
Pilot (c'): verify the TWO-STAGE trace-anchored early-commit mechanism.
Take the distractor item that overthought (gold=70000, abandoned), grab its reasoning
trace, truncate at the point the answer first stabilized, and ask a committer to give
the final answer from that prefix. Does early-commit RECOVER the correct answer that
full reasoning abandoned?
"""
import sys, re, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import chat

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "gsm8k_test.jsonl"
NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")

def norm(s):
    s = s.replace("$","").replace(",","")
    try:
        f=float(s); return str(int(f)) if f==int(f) else str(f)
    except: return None

def all_numbers(t):
    out=[]
    for m in NUM.finditer(t):
        v=norm(m.group())
        if v is not None: out.append(v)
    return out

def load(n):
    items=[]
    with open(DATA) as f:
        for i,line in enumerate(f):
            if i>=n: break
            d=json.loads(line); gold=d["answer"].split("####")[-1].strip().replace(",","")
            items.append({"q":d["question"],"gold":gold})
    return items

def commit_from_prefix(question, reasoning_prefix, committer="deepseek-v4-flash", sidx=0):
    """Second-stage commit: given a problem + partial reasoning, force an immediate answer."""
    msg=[{"role":"user","content":
          f"Problem:\n{question}\n\nPartial reasoning so far:\n{reasoning_prefix}\n\n"
          "Based ONLY on the reasoning above, state the final answer now. Do not reconsider. "
          "Respond with exactly 'Answer: <number>'."}]
    r=chat(msg, model=committer, temperature=0.0, max_tokens=60, sample_idx=sidx)
    n=all_numbers(r.text); return (n[-1] if n else None), r

it = load(3)[2]  # the gold=70000 item
distractor=(" (Note: a similar problem last week had a different setup, and some people "
            "mistakenly double the final total, but consider carefully.)")
q = it["q"] + distractor + "\nEnd your answer with 'Answer: <number>'."
# reproduce the overthinking run (cached, sample_idx=1 as in pilot)
r = chat([{"role":"user","content": q}], model="deepseek-v4-pro",
         temperature=0.0, max_tokens=2000, sample_idx=1)
reasoning = r.reasoning or ""
final_full = (all_numbers(r.text)[-1] if all_numbers(r.text) else None)
print(f"gold={it['gold']}  FULL-reasoning final={final_full}  rt={r.reasoning_tokens}")

# find stabilization point: first index (by sentence) where gold appears, take prefix up to there +1 sentence
sents = re.split(r'(?<=[.!?\n])\s+', reasoning)
cum=""; commit_point=None
for i,s in enumerate(sents):
    cum += s + " "
    if norm(it["gold"]) in all_numbers(s):
        commit_point=i; break
print(f"gold first stabilizes at trace sentence #{commit_point}; prefix chars={len(cum)}")

for committer in ["deepseek-v4-flash","deepseek-v4-pro"]:
    ans,_ = commit_from_prefix(q, cum.strip(), committer=committer, sidx=hash(committer)%97)
    ok = ans==norm(it["gold"])
    print(f"  early-commit via {committer:18} -> {ans}  {'RECOVERED ✔' if ok else 'no'}")
