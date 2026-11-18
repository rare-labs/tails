"""Compare tails.idf with IDFCurves.jl reference values and print a Markdown report.

Usage: python validation/compare_idfcurves.py > docs/validation_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tails.datasets import load_idfcurves
from tails.idf import IDFData, IDFModel, fit_idf, loglikelihood, scaling_test

REF = json.loads((Path(__file__).parents[1] / "tests" / "data" / "idfcurves_reference.json").read_text())
SPECS = {
    "simple": ("simple", None, "gaussian"),
    "general": ("general", None, "gaussian"),
    "general_matern_gaussian": ("general", "matern", "gaussian"),
    "general_exponential_gaussian": ("general", "exponential", "gaussian"),
    "general_exponential_t5": ("general", "exponential", "t"),
}


def main():
    print("| Station | Model | max rel. diff params | loglik tails | loglik Julia | Δ loglik | q100(1h) tails | q100(1h) Julia | CI(1h) tails | CI(1h) Julia |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st, ref in REF.items():
        data = IDFData.from_dataframe(load_idfcurves(st).dropna(), "Year")
        for key, spec in SPECS.items():
            r = ref[key]
            fit = fit_idf(data, *spec, df=5)
            p = fit.params[0]
            pj = np.array(r["params"])
            rel = np.max(np.abs(p - pj) / np.maximum(np.abs(pj), 1e-2))
            e, lo, hi = fit.quantile_ci([1.0], [100])
            ci_j = r.get("ci100_1h")
            ci_j = f"[{ci_j[0]:.3f}, {ci_j[1]:.3f}]" if ci_j else "fails in IDFCurves 66a34d1"
            print(
                f"| {st} | {key} | {rel:.1e} | {fit.loglik[0]:.4f} | {r['loglik']:.4f} | {fit.loglik[0] - r['loglik']:+.1e} | "
                f"{e.ravel()[0]:.4f} | {r['q100_1h']:.4f} | [{lo.ravel()[0]:.3f}, {hi.ravel()[0]:.3f}] | {ci_j} |"
            )
    print()
    print("Likelihood implementation check (tails log-likelihood evaluated at the Julia MLE):")
    print()
    for st, ref in REF.items():
        data = IDFData.from_dataframe(load_idfcurves(st).dropna(), "Year")
        diffs = []
        for key, spec in SPECS.items():
            m = IDFModel(*spec, df=5)
            diffs.append(abs(loglikelihood(m, ref[key]["params"], data) - ref[key]["loglik"]))
        print(f"- {st}: max |Δ loglik| = {max(diffs):.2e}")
    print()
    print("Scaling-model test p-values (shortest duration held out):")
    print()
    for st, ref in REF.items():
        data = IDFData.from_dataframe(load_idfcurves(st).dropna(), "Year")
        print(
            f"- {st}: simple tails {scaling_test(data, 'simple'):.3e} vs Julia {ref['simple']['scalingtest']:.3e}; "
            f"general tails {scaling_test(data, 'general'):.3e} vs Julia {ref['general']['scalingtest']:.3e}"
        )


if __name__ == "__main__":
    main()
