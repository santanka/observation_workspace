#!/usr/bin/env python3
"""Empirical X0-kappa fits and fixed-q reparameterization diagnostics (saved CSV only)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import plot_x0_kappa_relationship as relation
import statistics_phases as phases
import statistics_selection as selection
import statistics_plotting as styling


PLOT_CONFIG = {"dpi": 180, "font_size": 12}


def fit_model(x, y, degree, robust=False, reference=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    reference = float(np.median(x)) if reference is None else reference
    design = np.polynomial.polynomial.polyvander(x - reference, degree)
    if len(x) <= degree + 1 or np.linalg.matrix_rank(design) != degree + 1:
        raise ValueError("Insufficient independent pairs for regression")
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if robust:
        residual = y - design @ coefficients
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
        result = least_squares(lambda c: design @ c - y, coefficients, loss="soft_l1", f_scale=scale)
        if not result.success:
            raise RuntimeError(result.message)
        coefficients = result.x
    return reference, coefficients


def predict(x, reference, coefficients):
    return np.polynomial.polynomial.polyval(np.asarray(x) - reference, coefficients)


def correlation(x, y):
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def compare_models(points, bootstrap=300, seed=42):
    """LOMO CV and month-cluster bootstrap; no covariance correction for fitted x/y."""
    summaries, predictions, folds = [], [], []
    rng = np.random.default_rng(seed)
    for component in relation.ae.COMPONENTS:
        group = points.loc[points.component.eq(component)].copy()
        if group.empty:
            continue
        x, y = group.log10_X0.to_numpy(), group.kappa.to_numpy()
        months = group.month.to_numpy()
        unique = np.unique(months)
        for degree in (1, 2):
            for robust in (False, True):
                name = f"{'linear' if degree == 1 else 'quadratic'}_{'robust' if robust else 'ols'}"
                try:
                    reference, coefficients = fit_model(x, y, degree, robust)
                except ValueError:
                    continue
                fitted = predict(x, reference, coefficients)
                cv = np.full(len(x), np.nan)
                for month in unique:
                    test = months == month
                    try:
                        ref, coef = fit_model(x[~test], y[~test], degree, robust)
                    except ValueError:
                        continue
                    cv[test] = predict(x[test], ref, coef)
                    folds.append({"component": component, "model": name, "month": month,
                                  "n": int(test.sum()), "rmse": float(np.sqrt(np.mean((y[test] - cv[test]) ** 2))),
                                  "mae": float(np.mean(np.abs(y[test] - cv[test])))})
                valid = np.isfinite(cv)
                row = {"component": component, "model": name, "degree": degree,
                       "n": len(x), "n_months": len(unique), "x_reference": reference,
                       "x_min": x.min(), "x_max": x.max(),
                       "a": coefficients[0], "b": coefficients[1], "c": coefficients[2] if degree == 2 else 0.,
                       "rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
                       "residual_q16": float(np.quantile(y - fitted, .16)),
                       "residual_q84": float(np.quantile(y - fitted, .84)),
                       "cv_n": int(valid.sum()),
                       "cv_rmse": float(np.sqrt(np.mean((y[valid] - cv[valid]) ** 2))) if valid.any() else np.nan,
                       "cv_mae": float(np.mean(np.abs(y[valid] - cv[valid]))) if valid.any() else np.nan}
                # Bootstrap OLS coefficients only; robust fit is an outlier sensitivity check.
                samples = []
                if not robust and len(unique) >= 2:
                    for _ in range(bootstrap):
                        drawn = rng.choice(unique, size=len(unique), replace=True)
                        indices = np.concatenate([np.flatnonzero(months == m) for m in drawn])
                        try:
                            _, coefs = fit_model(x[indices], y[indices], degree, reference=reference)
                            samples.append(coefs)
                        except ValueError:
                            pass
                row["bootstrap_success"] = len(samples)
                if samples:
                    for i, label in enumerate(("a", "b", "c")[:degree + 1]):
                        row[label + "_ci025"], row[label + "_ci975"] = np.quantile(samples, [.025, .975], axis=0)[:, i]
                summaries.append(row)
                saved = group.copy()
                saved["model"], saved["predicted"], saved["cv_predicted"] = name, fitted, cv
                saved["residual"], saved["cv_residual"] = y - fitted, y - cv
                predictions.append(saved)
    if not summaries:
        raise ValueError("No component has sufficient independent pairs")
    return pd.DataFrame(summaries), pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def pivot_diagnostics(points, pivots):
    rows, summaries = [], []
    for component, group in points.groupby("component", sort=False):
        for q in [1., *pivots]:
            transformed = group.log10_X0 - group.kappa * np.log10(q)
            saved = group.copy()
            saved["q_pivot"], saved["log10_X_pivot"] = q, transformed
            rows.append(saved)
            summaries.append({"component": component, "q_pivot": q, "n": len(group),
                              "spearman_rho": correlation(transformed, group.kappa)})
    return pd.DataFrame(summaries), pd.concat(rows, ignore_index=True)


def save_figure(fig, base):
    for suffix in (".png", ".pdf"):
        fig.savefig(base.with_suffix(suffix), dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_fits(points, summary, predictions, title, base):
    with plt.rc_context({"font.size": PLOT_CONFIG["font_size"]}):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for col, component in enumerate(relation.ae.COMPONENTS):
            group = points.loc[points.component.eq(component)]
            if group.empty:
                continue
            axes[0, col].scatter(group.log10_X0, group.kappa, s=15, alpha=.25, color=".4")
            x = np.linspace(group.log10_X0.min(), group.log10_X0.max(), 250)
            for degree, color in ((1, "tab:blue"), (2, "tab:orange")):
                for robust in (False, True):
                    model = f"{'linear' if degree == 1 else 'quadratic'}_{'robust' if robust else 'ols'}"
                    match = summary.loc[summary.component.eq(component) & summary.model.eq(model)]
                    if match.empty:
                        continue
                    row = match.iloc[0]
                    coef = [row.a, row.b] + ([row.c] if degree == 2 else [])
                    axes[0, col].plot(x, predict(x, row.x_reference, coef), color=color,
                                      ls="--" if robust else "-", lw=1.7,
                                      label=f"{model}; CV RMSE={row.cv_rmse:.3f}")
                    if not robust:
                        residuals = predictions.loc[predictions.component.eq(component) & predictions.model.eq(model)]
                        axes[1, col].scatter(residuals.log10_X0, residuals.cv_residual,
                                             color=color, s=13, alpha=.3, label=model)
            axes[0, col].set_title(f"{component}; N={len(group)}")
            axes[0, col].set_ylabel(r"$\kappa$")
            axes[1, col].set_ylabel(r"$\kappa-\widehat\kappa_{\rm held\ out}$")
            axes[1, col].axhline(0, color=".2", lw=1)
            for row in range(2):
                axes[row, col].set_xlabel(r"$\log_{10}X_0$")
                axes[row, col].legend(fontsize=8)
                styling.detail_grid(axes[row, col])
        fig.suptitle(title + "\nEqual range weights; leave-one-month-out CV; fit curves within observed range")
        fig.tight_layout(rect=(0, 0, 1, .92))
        save_figure(fig, base)


def plot_pivots(summary, transformed, title, base):
    pivots = summary.q_pivot.unique()
    with plt.rc_context({"font.size": PLOT_CONFIG["font_size"]}):
        fig, axes = plt.subplots(len(pivots), 3, figsize=(17, 4 * len(pivots)), squeeze=False)
        for row, q in enumerate(pivots):
            for col, component in enumerate(relation.ae.COMPONENTS):
                group = transformed.loc[transformed.component.eq(component) & transformed.q_pivot.eq(q)]
                ax = axes[row, col]
                ax.scatter(group.log10_X_pivot, group.kappa, alpha=.28, s=16, color=".4")
                rho = correlation(group.log10_X_pivot, group.kappa)
                ax.set_title(f"{component}; " + rf"$k_\perp\rho_{{\mathrm{{i}}}}={q:g}$"
                             + f"; N={len(group)}; Spearman rho={rho:.3f}")
                ax.set_xlabel(r"$\log_{10}X(k_\perp\rho_{\mathrm{i}})$")
                ax.set_ylabel(r"$\kappa$")
                styling.detail_grid(ax)
        fig.suptitle(title + "\nModel amplitudes at fixed " + r"$k_\perp\rho_{\mathrm{i}}$"
                     + "; individual fit-domain coverage not verified")
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_figure(fig, base)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    relation.add_period_arguments(parser)
    selection.add_selection_argument(parser)
    styling.add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Content directory (x0_kappa_fits)")
    parser.add_argument("--phase", choices=tuple(phases.PHASES), action="append")
    parser.add_argument("--pivot-q", type=float, action="append", help="Fixed q=k_perp*rho_i; default 3 and 10")
    parser.add_argument("--bootstrap-months", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pivots = list(dict.fromkeys(args.pivot_q or [3., 10.]))
    if any(not np.isfinite(q) or q <= 0 or q == 1 for q in pivots) or args.bootstrap_months < 0:
        raise ValueError("Require positive finite pivots other than baseline 1, and bootstrap-months >= 0")
    period = relation.resolve_period(args)
    root = args.statistics_dir or relation.batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks) / "x0_kappa_fits"
    master, provenance = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    output.mkdir(parents=True, exist_ok=True)
    for phase in dict.fromkeys(args.phase or phases.PHASES):
        points = relation.paired_points(selected, phase)
        if points.empty:
            continue
        dates = selected.loc[selected.phase_mode.eq(phase)].set_index("range_uid").range_start
        points["range_start"] = points.range_uid.map(dates)
        times = pd.to_datetime(points.range_start, errors="raise", utc=True)
        if times.isna().any():
            raise ValueError("Missing range_start prevents monthly validation")
        points["month"] = times.dt.strftime("%Y%m")
        base = output / f"arase_x0_kappa_fits_{phase}_{period.date_suffix}"
        extensions = (".png", ".pdf", ".csv", ".predictions.csv", ".folds.csv", ".pivots.csv",
                      ".pivot_points.csv", ".pivot.png", ".pivot.pdf", ".provenance.json")
        if not styling.should_write(base, overwrite=not args.skip_existing, extensions=extensions):
            continue
        summary, predictions, folds = compare_models(points, args.bootstrap_months, args.seed)
        pivot_summary, pivot_points = pivot_diagnostics(points, pivots)
        for suffix, table in ((".csv", summary), (".predictions.csv", predictions), (".folds.csv", folds),
                              (".pivots.csv", pivot_summary), (".pivot_points.csv", pivot_points)):
            table.insert(0, "requested_phase", phase)
            table.to_csv(base.with_suffix(suffix), index=False)
        title = (f"Arase X0-kappa: {phases.PHASES[phase]}; {period.key}; all MLT; "
                 f"occupied 10 s blocks >= {args.min_occupied_blocks}")
        plot_fits(points, summary, predictions, title, base)
        # with_suffix replaces an existing suffix, so use a suffix-free base for the pivot figures.
        pivot_base = Path(str(base) + "_pivot")
        plot_pivots(pivot_summary, pivot_points, title, pivot_base)
        for suffix in (".png", ".pdf"):
            pivot_base.with_suffix(suffix).replace(base.with_suffix(".pivot" + suffix))
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
                    "phase": phase, "selection": selection.selection_metadata(master, args.min_occupied_blocks),
                    "formula": "kappa=a+b*(log10_X0-x_reference)+c*(log10_X0-x_reference)^2",
                    "grid_config": styling.GRID_CONFIG, "master_sha256": provenance["annual_master_sha256"],
                    "source_provenance_sha256": relation.batch.file_sha256(root / "inputs/provenance.json"),
                    "validation": "leave one calendar month out; pooled range-weighted RMSE/MAE",
                    "bootstrap": {"unit": "calendar month", "draws": args.bootstrap_months, "seed": args.seed,
                                  "interval": "OLS coefficient 2.5-97.5 percentiles; conditional on saved fit pairs"},
                    "robust": "soft_l1; train-residual MAD scale; no robust coefficient intervals",
                    "pivot_q": pivots, "pivot_domain_verified": False,
                    "caveat": "shared-fit x/y covariance and measurement errors not corrected; no causal inference",
                    "valid_fraction_screening": False,
                    "code_sha256": {str(Path(p).resolve()): relation.batch.file_sha256(Path(p)) for p in
                                    (__file__, relation.__file__, phases.__file__, selection.__file__, styling.__file__)}}
        base.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Completed {phase}: {base}", flush=True)
        print(summary[["component", "model", "cv_rmse", "a", "b", "c"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
