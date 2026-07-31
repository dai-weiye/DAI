"""
Figure 1 overview / pipeline flowchart for the non-termination paper.

The figure is authored as vector TikZ (scripts/fig_pipeline.tex) and compiled
here with pdflatex; this script only drives the build and copies the result to
results/figures/fig_pipeline.{pdf,png}.  If pdflatex is unavailable the existing
PDF is left untouched (every other script in run_all.sh is pure Python).

Visualizes the study in one glance: a question (clean or adversarially distracted)
goes to a reasoning LLM with a fixed token budget; the model either TERMINATES
(emits an answer, which is graded) or NON-TERMINATES (exhausts the budget and emits
nothing). The non-termination branch is the paper's headline failure mode.
"""
import pathlib
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "fig_pipeline.tex"
FIG = ROOT / "results/figures"


def raster(pdf: pathlib.Path, png: pathlib.Path, width: int = 2100) -> None:
    """PDF -> PNG at ~300 dpi, using whichever converter this machine has."""
    if shutil.which("sips"):  # macOS
        subprocess.run(["sips", "-s", "format", "png", "--resampleWidth", str(width),
                        str(pdf), "--out", str(png)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                        str(pdf), str(png.with_suffix(""))], check=True)
    else:
        print("  (no sips/pdftoppm; skipping PNG)")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if not shutil.which("pdflatex"):
        print("pdflatex not found; keeping existing fig_pipeline.pdf")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shutil.copy(SRC, tmp / SRC.name)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            SRC.name], cwd=tmp, capture_output=True, text=True)
        out = tmp / "fig_pipeline.pdf"
        if r.returncode != 0 or not out.exists():
            print(r.stdout[-2000:])
            raise SystemExit("pdflatex failed on fig_pipeline.tex")
        shutil.copy(out, FIG / "fig_pipeline.pdf")
        raster(FIG / "fig_pipeline.pdf", FIG / "fig_pipeline.png")
    print("wrote fig_pipeline.pdf/.png")


if __name__ == "__main__":
    main()
