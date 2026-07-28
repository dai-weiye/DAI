"""
Answer grading: numeric (GSM8K) and light symbolic (MATH500) equivalence.
Uses sympy if available for expression equivalence; falls back to string/number match.
Conservative: only marks correct when confident, to avoid inflating accuracy.
"""
from __future__ import annotations
import re
from trace_method import norm_num

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    _HAVE_SYMPY = True
except Exception:
    _HAVE_SYMPY = False


def _clean_latex(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("^{\\circ}", "").replace("\\circ", "")
    s = s.strip().strip("$").strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[len("\\boxed{"):-1]
    return s.strip()


def grade(pred: str | None, gold: str, gold_type: str = "number") -> bool:
    if pred is None:
        return False
    pred = str(pred).strip()
    gold = str(gold).strip()
    # 0) multiple-choice: compare letters (tag-aware to avoid the 'A' in 'Answer')
    if gold_type == "mcq":
        m = re.search(r"answer\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", pred, re.I)
        if m:
            return m.group(1).upper() == gold.upper()
        # else: if pred is essentially a bare letter, use it; take the LAST A-D
        letters = re.findall(r"\b([A-D])\b", pred.upper())
        return bool(letters) and letters[-1] == gold.upper()
    # 1) numeric fast path
    pn, gn = norm_num(pred), norm_num(gold)
    if pn is not None and gn is not None:
        try:
            return abs(float(pn) - float(gn)) < 1e-4
        except ValueError:
            return pn == gn
    # 2) exact string after cleaning
    pc, gc = _clean_latex(pred), _clean_latex(gold)
    if pc == gc:
        return True
    pcn, gcn = norm_num(pc), norm_num(gc)
    if pcn is not None and gcn is not None:
        try:
            return abs(float(pcn) - float(gcn)) < 1e-4
        except ValueError:
            pass
    # 3) symbolic equivalence (MATH)
    if _HAVE_SYMPY and gold_type == "expr":
        for parser in (_try_parse_latex, _try_parse_expr):
            pe, ge = parser(pc), parser(gc)
            if pe is not None and ge is not None:
                try:
                    if sympy.simplify(pe - ge) == 0:
                        return True
                except Exception:
                    pass
    return pc == gc


def _try_parse_latex(s):
    try:
        return parse_latex(s)
    except Exception:
        return None


def _try_parse_expr(s):
    try:
        return sympy.sympify(s.replace("\\", ""))
    except Exception:
        return None
