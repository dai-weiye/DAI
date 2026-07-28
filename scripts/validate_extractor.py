"""
Validate the final-answer extractor against GSM8K gold on real v4-pro outputs.
Reports extraction/grading accuracy so the paper can state it honestly.
Cheap: reuses cache where possible; caps new spend via MAX_SPEND_USD.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import chat, get_spend
from datasets_load import load_gsm8k
from trace_method import extract_final_answer
from grade import grade

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
items = load_gsm8k(N)
correct = extracted = 0
mism = []
for it in items:
    r = chat([{"role": "user", "content": it["question"] +
               "\nShow your work, then end with 'Answer: <number>'."}],
             model="deepseek-v4-pro", temperature=0.0, max_tokens=1200)
    pred = extract_final_answer(r.text)
    if pred is not None:
        extracted += 1
    ok = grade(pred, it["gold"], "number")
    correct += ok
    if not ok and pred is not None:
        mism.append((it["gold"], pred, r.text[-60:].replace("\n", " ")))
print(f"N={N}  extracted={extracted}/{N} ({extracted/N:.1%})  "
      f"graded_correct={correct}/{N} ({correct/N:.1%})")
print(f"new spend this run: ${get_spend():.4f}")
print("sample mismatches (gold, pred, tail):")
for g, p, t in mism[:6]:
    print(f"  gold={g} pred={p} | ...{t}")
