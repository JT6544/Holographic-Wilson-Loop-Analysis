"""Numerically reconstruct a holographic Wilson-loop potential.

The module evaluates the boundary separation and renormalised energy as
functions of the string turning point. It then identifies ultraviolet and
infrared fit regions and compares their fitted coefficients with analytical
predictions.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import cycle, islice
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.integrate import IntegrationWarning
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.special import gamma
from scipy.stats import linregress


def default_config():
    """Return the default numerical and fit-selection configuration."""
    return {
        "R": 1.0,
        "U_max": 20.0,
        "N_scan": 1000,
        "epsilon": 1e-8,
        "quad_tol": 1e-8,
        "quad_lim": 250,
        "uv_exclude_frac": 0.65,
        "uv_skip_points": 10,
        "uv_alpha_tols": [0.01, 0.015, 0.02],
        "uv_min_points": 12,
        "ir_tension_tols": [0.002, 0.003, 0.005],
        "ir_min_points": 30,
        "ir_smooth_window": 11,
    }


_CFG = default_config()


def _validate_config_for_sigma(Sigma, cfg):
    """Validate settings that determine the numerical grid and selectors."""
    required_keys = set(default_config())
    missing = sorted(required_keys.difference(cfg))
    if missing:
        raise ValueError(f"Missing configuration entries: {', '.join(missing)}")

    if not np.isfinite(Sigma) or Sigma <= 0:
        raise ValueError("Sigma must be a positive finite number.")
    if not np.isfinite(cfg["R"]) or cfg["R"] <= 0:
        raise ValueError("R must be a positive finite number.")
    if not np.isfinite(cfg["U_max"]) or cfg["U_max"] <= np.sqrt(Sigma):
        raise ValueError("U_max must be finite and greater than sqrt(Sigma).")

    epsilon_max = cfg["U_max"] - np.sqrt(Sigma)
    if (
        not np.isfinite(cfg["epsilon"])
        or cfg["epsilon"] <= 0
        or cfg["epsilon"] >= epsilon_max
    ):
        raise ValueError("epsilon must lie between 0 and U_max - sqrt(Sigma).")

    integer_settings = {
        "N_scan": (2, None),
        "quad_lim": (1, None),
        "uv_skip_points": (0, cfg["N_scan"] - 1),
        "uv_min_points": (2, cfg["N_scan"]),
        "ir_min_points": (2, cfg["N_scan"]),
        "ir_smooth_window": (1, None),
    }
    for name, (minimum, maximum) in integer_settings.items():
        value = cfg[name]
        if not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be an integer <= {maximum}.")

    if not np.isfinite(cfg["quad_tol"]) or cfg["quad_tol"] <= 0:
        raise ValueError("quad_tol must be a positive finite number.")
    if not 0 < cfg["uv_exclude_frac"] <= 1:
        raise ValueError("uv_exclude_frac must lie in the interval (0, 1].")

    for name in ("uv_alpha_tols", "ir_tension_tols"):
        values = np.asarray(cfg[name], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain positive finite tolerances.")


def exact_slope(Sigma, R=1.0):
    """Return the analytical infrared string tension."""
    return Sigma / (np.pi * R**2)


def exact_alpha(R=1.0):
    """Return the analytical ultraviolet Coulomb coefficient."""
    return 4.0 * np.pi**2 * R**2 / gamma(0.25)**4


def F(U, Sigma):
    """Evaluate the deformed holographic warp factor."""
    return U**2 + (Sigma**2 / U**2)


def make_U0_grid(Sigma, cfg=None):
    """Construct the logarithmically spaced turning-point grid."""
    if cfg is None:
        cfg = _CFG

    _validate_config_for_sigma(Sigma, cfg)
    U_star = np.sqrt(Sigma)
    eps_max = cfg["U_max"] - U_star
    eps_grid = np.logspace(
        np.log10(cfg["epsilon"]),
        np.log10(eps_max),
        cfg["N_scan"],
    )
    return U_star + eps_grid


def _integrate_one(U0, Sigma, U_star, U_max, R, quad_lim, quad_tol):
    """Evaluate separation and energy for one turning point."""
    F0 = F(U0, Sigma)
    t_max = np.sqrt(max(U_max - U0, 0.0))

    def integrand_ell(t):
        FU = F(U0 + t * t, Sigma)
        arg = FU**2 - F0**2
        return 0.0 if arg <= 0.0 else 2.0 * t / (FU * np.sqrt(arg))

    def integrand_E(t):
        FU = F(U0 + t * t, Sigma)
        arg = FU**2 - F0**2
        return 0.0 if arg <= 0.0 else 2.0 * t * (FU / np.sqrt(arg) - 1.0)

    kw = dict(limit=quad_lim, epsabs=quad_tol, epsrel=quad_tol)

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", IntegrationWarning)
            val_l, _ = integrate.quad(integrand_ell, 0.0, t_max, **kw)
            val_E, _ = integrate.quad(integrand_E, 0.0, t_max, **kw)
    except Exception as error:
        return np.nan, np.nan, 0, type(error).__name__

    ell = 2.0 * R**2 * F0 * val_l
    E = (1.0 / np.pi) * (val_E - (U0 - U_star))
    warning_count = sum(
        issubclass(item.category, IntegrationWarning) for item in caught_warnings
    )
    return ell, E, warning_count, None


def compute_curves(Sigma, cfg=None):
    """Compute the parametric potential curve for one deformation value."""
    if cfg is None:
        cfg = _CFG

    U_star = np.sqrt(Sigma)
    U0_arr = make_U0_grid(Sigma, cfg)

    args = [
        (
            U0,
            Sigma,
            U_star,
            cfg["U_max"],
            cfg["R"],
            cfg["quad_lim"],
            cfg["quad_tol"],
        )
        for U0 in U0_arr
    ]

    with ProcessPoolExecutor() as pool:
        pairs = list(pool.map(_integrate_one, *zip(*args)))

    ell_arr = np.array([p[0] for p in pairs])
    E_arr = np.array([p[1] for p in pairs])
    integration_warning_count = sum(p[2] for p in pairs)
    warned_point_count = sum(p[2] > 0 for p in pairs)
    integration_errors = [p[3] for p in pairs if p[3] is not None]

    if integration_warning_count or integration_errors:
        error_types = ", ".join(sorted(set(integration_errors))) or "none"
        warnings.warn(
            "Integration diagnostics for "
            f"Sigma={Sigma}: {warned_point_count} warned point(s) "
            f"({integration_warning_count} IntegrationWarning event(s)), "
            f"{len(integration_errors)} failed point(s); error types: {error_types}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return U0_arr, ell_arr, E_arr


def clean_and_sort(U0_arr, ell_arr, E_arr):
    """Remove invalid points and sort the surviving data by separation."""
    mask = (
        np.isfinite(U0_arr)
        & np.isfinite(ell_arr)
        & np.isfinite(E_arr)
        & (ell_arr > 0)
    )
    idx = np.argsort(ell_arr[mask])
    return U0_arr[mask][idx], ell_arr[mask][idx], E_arr[mask][idx]


def _fit(x, y):
    """Fit a straight line and return its slope, intercept, and R-squared."""
    slope, intercept, r, *_ = linregress(x, y)
    return slope, intercept, r**2


def contiguous_blocks(mask):
    """Return contiguous index blocks selected by a Boolean mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)


def local_dE_dell(ell_sorted, E_sorted):
    """Estimate the local derivative of energy with respect to separation."""
    return np.gradient(E_sorted, ell_sorted)


def smooth_signal(arr, window):
    """Apply median and moving-average filters to a one-dimensional signal."""
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    median_smoothed = median_filter(arr, size=window, mode="nearest")
    return uniform_filter1d(median_smoothed, size=window, mode="nearest")


def find_coulomb_region(U0_sorted, ell_sorted, E_sorted, cfg=None):
    """Select and fit the ultraviolet Coulomb region."""
    if cfg is None:
        cfg = _CFG

    alpha_ex = exact_alpha(cfg["R"])
    alpha_eff = ell_sorted**2 * local_dE_dell(ell_sorted, E_sorted)

    base_mask = (
        (np.arange(len(ell_sorted)) >= cfg["uv_skip_points"])
        & (U0_sorted <= cfg["uv_exclude_frac"] * cfg["U_max"])
        & np.isfinite(alpha_eff)
    )

    chosen = None
    for tol in cfg["uv_alpha_tols"]:
        mask = base_mask & (np.abs(alpha_eff - alpha_ex) / alpha_ex <= tol)
        groups = [
            group
            for group in contiguous_blocks(mask)
            if len(group) >= cfg["uv_min_points"]
        ]
        if groups:
            chosen = max(groups, key=len)
            break

    if chosen is None:
        raise RuntimeError(
            "No UV Coulomb region satisfied the configured tolerances and "
            f"minimum length ({cfg['uv_min_points']} points)."
        )

    s, e = chosen[0], chosen[-1] + 1
    slope, intercept, r2 = _fit(1.0 / ell_sorted[s:e], E_sorted[s:e])
    alpha_fit = -slope
    rel_err = abs(alpha_fit - alpha_ex) / alpha_ex

    return s, e, alpha_fit, intercept, r2, rel_err, alpha_eff

def find_linear_region(ell_sorted, E_sorted, Sigma, cfg=None):
    """Select and fit the infrared linear region."""
    if cfg is None:
        cfg = _CFG

    T_ex = exact_slope(Sigma, cfg["R"])
    T_eff_raw = local_dE_dell(ell_sorted, E_sorted)
    T_eff_smooth = smooth_signal(T_eff_raw, cfg["ir_smooth_window"])

    chosen = None
    for tol in cfg["ir_tension_tols"]:
        mask = np.isfinite(T_eff_smooth) & (
            np.abs(T_eff_smooth - T_ex) / T_ex <= tol
        )
        groups = [
            group
            for group in contiguous_blocks(mask)
            if len(group) >= cfg["ir_min_points"]
        ]
        if groups:
            chosen = max(groups, key=lambda group: (len(group), group[-1]))
            break

    if chosen is None:
        raise RuntimeError(
            "No IR linear region satisfied the configured tolerances and "
            f"minimum length ({cfg['ir_min_points']} points)."
        )

    s, e = chosen[0], chosen[-1] + 1
    slope, intercept, r2 = _fit(ell_sorted[s:e], E_sorted[s:e])
    rel_err = abs(slope - T_ex) / T_ex

    return s, e, slope, intercept, r2, rel_err, T_eff_raw, T_eff_smooth

def _hline(col_widths, left="├", mid="┼", right="┤"):
    """Build a horizontal border for a Unicode box table."""
    return left + mid.join("─" * (width + 2) for width in col_widths) + right


def print_box_table(title, headers, rows, fmt=None):
    """Print a right-aligned Unicode table."""
    if fmt is None:
        fmt = [".6g"] * len(headers)

    str_rows = [
        [
            format(v, f) if isinstance(v, (int, float, np.floating)) else str(v)
            for v, f in zip(row, fmt)
        ]
        for row in rows
    ]

    cw = [
        max(len(h), max((len(r[j]) for r in str_rows), default=0))
        for j, h in enumerate(headers)
    ]

    def _row(cells):
        return "│" + "│".join(
            f" {cell:>{width}} " for cell, width in zip(cells, cw)
        ) + "│"

    top = "┌" + "┬".join("─" * (width + 2) for width in cw) + "┐"
    bottom = "└" + "┴".join("─" * (width + 2) for width in cw) + "┘"

    print(f"\n  {title}")
    print(top)
    print(_row(headers))
    print(_hline(cw))

    for row in str_rows:
        print(_row(row))

    print(bottom)


COLOURS = ["navy", "darkred", "darkgreen", "indigo"]


def _expanded_colours(colours, count):
    """Return enough colours for every requested dataset."""
    if not colours:
        raise ValueError("At least one plotting colour is required.")
    return list(islice(cycle(colours), count))


def _subplot_grid(item_count):
    """Create a compact plot grid with at most two columns."""
    if item_count < 1:
        raise ValueError("At least one dataset is required for plotting.")
    columns = min(2, item_count)
    rows = int(np.ceil(item_count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 3.5 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    return fig, flat_axes


def _plot_E_vs_ell(Sigma, d, colour):
    """Plot the full potential and its selected UV and IR fits."""
    x, y = d["ell_s"], d["E_s"]
    s_uv, e_uv = d["uv_start"], d["uv_end"]
    s_ir, e_ir = d["ir_start"], d["ir_end"]
    x_uv, y_uv = x[s_uv:e_uv], y[s_uv:e_uv]
    x_ir, y_ir = x[s_ir:e_ir], y[s_ir:e_ir]
    T_ex, alpha_ex = d["exact"], d["alpha_exact"]

    b_ir_exact = np.mean(y_ir - T_ex * x_ir)
    b_uv_exact = np.mean(y_uv + alpha_ex / x_uv)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(
        x,
        y,
        color=colour,
        linewidth=2.0,
        zorder=4,
        label=rf"$\Sigma = {Sigma}$",
    )
    ax.axvspan(
        x_uv[0],
        x_uv[-1],
        color="deepskyblue",
        alpha=0.16,
        zorder=1,
        label="Coulomb region",
    )
    ax.axvspan(
        x_ir[0],
        x_ir[-1],
        color="gold",
        alpha=0.16,
        zorder=1,
        label="linear region",
    )

    ax.plot(
        x_uv,
        -d["alpha_fit"] / x_uv + d["c_intercept"],
        "--",
        color="teal",
        linewidth=2.0,
        zorder=5,
        label=rf"UV fit: $\alpha={d['alpha_fit']:.5f}$, $R^2={d['c_r2']:.5f}$",
    )
    ax.plot(
        x_uv,
        -alpha_ex / x_uv + b_uv_exact,
        "-",
        color="deepskyblue",
        linewidth=2.2,
        zorder=6,
        label=rf"UV exact: $\alpha={alpha_ex:.5f}$",
    )
    ax.plot(
        x_ir,
        d["fit_slope"] * x_ir + d["intercept"],
        "--",
        color="grey",
        linewidth=1.8,
        zorder=5,
        label=rf"IR fit: $T={d['fit_slope']:.5f}$, $R^2={d['r2']:.5f}$",
    )
    ax.plot(
        x_ir,
        T_ex * x_ir + b_ir_exact,
        "-",
        color="black",
        linewidth=2.2,
        zorder=6,
        label=rf"IR exact: $T={T_ex:.5f}$",
    )

    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$E$", fontsize=13)
    ax.set_title(rf"$E$ vs $\ell$  —  $\Sigma = {Sigma}$", fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_uv_diagnostic(Sigma_values, results, colours):
    """Plot zoomed ultraviolet diagnostics around each selected fit region."""
    fig, axes = _subplot_grid(len(Sigma_values))

    for ax, Sigma, colour in zip(axes, Sigma_values, colours):
        d = results[Sigma]
        x = 1.0 / d["ell_s"]
        y = d["E_s"]
        s_uv, e_uv = d["uv_start"], d["uv_end"]
        padding = max(10, e_uv - s_uv)
        display_start = max(0, s_uv - padding)
        display_end = min(len(x), e_uv + padding)
        x_plot = x[display_start:display_end]
        y_plot = y[display_start:display_end]
        x_uv = x[s_uv:e_uv]
        y_uv = y[s_uv:e_uv]
        alpha_ex = d["alpha_exact"]
        b_uv_exact = np.mean(y_uv + alpha_ex * x_uv)

        ax.plot(
            x_plot,
            y_plot,
            color=colour,
            linewidth=1.9,
            label=rf"$\Sigma = {Sigma}$",
        )
        ax.axvspan(
            x_uv[0],
            x_uv[-1],
            color="deepskyblue",
            alpha=0.16,
            label="Coulomb region",
        )
        ax.plot(
            x_uv,
            -d["alpha_fit"] * x_uv + d["c_intercept"],
            "--",
            color="teal",
            linewidth=1.8,
            label=rf"fit: $\alpha={d['alpha_fit']:.5f}$",
        )
        ax.plot(
            x_uv,
            -alpha_ex * x_uv + b_uv_exact,
            "-",
            color="black",
            linewidth=2.0,
            label=rf"exact: $\alpha={alpha_ex:.5f}$",
        )
        ax.set_xlabel(r"$1/\ell$", fontsize=11)
        ax.set_ylabel(r"$E$", fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    for unused_ax in axes[len(Sigma_values) :]:
        unused_ax.set_visible(False)

    fig.suptitle(
        r"UV diagnostic: $E$ versus $1/\ell$ near the selected fit regions",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_ir_diagnostic(Sigma_values, results, colours):
    """Plot zoomed infrared diagnostics over each selected fit region."""
    fig, axes = _subplot_grid(len(Sigma_values))

    for ax, Sigma, colour in zip(axes, Sigma_values, colours):
        d = results[Sigma]
        x = d["ell_s"]
        T_ex = d["exact"]
        s_ir, e_ir = d["ir_start"], d["ir_end"]
        padding = max(10, (e_ir - s_ir) // 10)
        display_start = max(0, s_ir - padding)
        x_plot = x[display_start:e_ir]
        T_plot = d["T_eff_smooth"][display_start:e_ir]

        selected_tension = d["T_eff_smooth"][s_ir:e_ir]
        max_selected_deviation = np.max(np.abs(selected_tension - T_ex))
        vertical_margin = max(0.05 * abs(T_ex), 1.5 * max_selected_deviation)

        ax.plot(
            x_plot,
            T_plot,
            color=colour,
            linewidth=1.8,
            label=rf"$T_{{\rm eff}}$, $\Sigma={Sigma}$",
        )
        ax.axhline(
            T_ex,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=rf"exact $T={T_ex:.6f}$",
        )
        ax.axvspan(
            x[s_ir],
            x[e_ir - 1],
            color="gold",
            alpha=0.16,
            label="linear region",
        )
        ax.set_ylim(T_ex - vertical_margin, T_ex + vertical_margin)
        ax.set_xlabel(r"$\ell$", fontsize=11)
        ax.set_ylabel(r"$dE/d\ell$", fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    for unused_ax in axes[len(Sigma_values) :]:
        unused_ax.set_visible(False)

    fig.suptitle(
        r"IR diagnostic: $dE/d\ell$ over the selected fit regions",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_comparison(Sigma_values, results, colours):
    """Plot every numerical potential with its exact IR asymptote."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    for Sigma, colour in zip(Sigma_values, colours):
        d = results[Sigma]
        x = d["ell_s"]
        y = d["E_s"]
        s_ir, e_ir = d["ir_start"], d["ir_end"]
        x_ir, y_ir = x[s_ir:e_ir], y[s_ir:e_ir]
        T_ex = d["exact"]
        b_ir_exact = np.mean(y_ir - T_ex * x_ir)

        ax.plot(x, y, color=colour, linewidth=2.0, label=rf"$\Sigma={Sigma}$")
        ax.plot(
            x_ir,
            T_ex * x_ir + b_ir_exact,
            "--",
            color=colour,
            linewidth=1.6,
            alpha=0.9,
        )

    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$E$", fontsize=13)
    ax.set_title(
        r"Comparison plot: numerical $E(\ell)$ with exact IR asymptotes",
        fontsize=14,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def scanning_range_convergence_test(Sigma_values, cfg=None, U_max_values=None):
    """Repeat the IR extraction for several finite radial cutoffs."""
    if cfg is None:
        cfg = _CFG

    if U_max_values is None:
        U_max_values = (
            0.5 * cfg["U_max"],
            0.75 * cfg["U_max"],
            cfg["U_max"],
            1.25 * cfg["U_max"],
        )

    rows = []

    for Sigma in Sigma_values:
        T_ex = exact_slope(Sigma, cfg["R"])

        for U_max in U_max_values:
            cfg_test = dict(cfg)
            cfg_test["U_max"] = float(U_max)

            U0_raw, ell_raw, E_raw = compute_curves(Sigma, cfg_test)
            _, ell_s, E_s = clean_and_sort(U0_raw, ell_raw, E_raw)
            _, _, fit_slope, _, r2, _, _, _ = find_linear_region(
                ell_s,
                E_s,
                Sigma,
                cfg_test,
            )

            rows.append([
                Sigma,
                U_max,
                U0_raw[0],
                U0_raw[-1],
                ell_s[0],
                ell_s[-1],
                T_ex,
                fit_slope,
                100.0 * (fit_slope - T_ex) / T_ex,
                r2,
            ])

    return rows

def run(Sigma_values=(0.5, 1.0, 2.0, 4.0), cfg=None, colours=None):
    """Execute the full reconstruction, reporting, and plotting workflow."""
    if cfg is None:
        cfg = _CFG

    if colours is None:
        colours = COLOURS

    Sigma_values = tuple(Sigma_values)
    if not Sigma_values:
        raise ValueError("At least one Sigma value is required.")
    plot_colours = _expanded_colours(colours, len(Sigma_values))

    results = {}

    for Sigma in Sigma_values:
        U0_raw, ell_raw, E_raw = compute_curves(Sigma, cfg)
        U0_s, ell_s, E_s = clean_and_sort(U0_raw, ell_raw, E_raw)

        (
            uv_s,
            uv_e,
            alpha_fit,
            c_intercept,
            c_r2,
            alpha_rel_err,
            alpha_eff,
        ) = find_coulomb_region(U0_s, ell_s, E_s, cfg)

        (
            ir_s,
            ir_e,
            fit_slope,
            intercept,
            r2,
            ir_rel_err,
            T_eff_raw,
            T_eff_smooth,
        ) = find_linear_region(ell_s, E_s, Sigma, cfg)

        if uv_e > ir_s:
            raise RuntimeError(
                f"UV and IR windows overlap for Sigma={Sigma}: "
                f"UV [{uv_s},{uv_e}) vs IR [{ir_s},{ir_e})"
            )

        results[Sigma] = dict(
            U0_raw=U0_raw,
            ell_raw=ell_raw,
            E_raw=E_raw,
            U0_s=U0_s,
            ell_s=ell_s,
            E_s=E_s,
            uv_start=uv_s,
            uv_end=uv_e,
            ir_start=ir_s,
            ir_end=ir_e,
            fit_slope=fit_slope,
            intercept=intercept,
            r2=r2,
            alpha_fit=alpha_fit,
            c_intercept=c_intercept,
            c_r2=c_r2,
            alpha_exact=exact_alpha(cfg["R"]),
            alpha_rel_err=alpha_rel_err,
            ir_rel_err=ir_rel_err,
            T_eff_raw=T_eff_raw,
            T_eff_smooth=T_eff_smooth,
            alpha_eff=alpha_eff,
            exact=exact_slope(Sigma, cfg["R"]),
        )

    summary_rows = []

    for Sigma in Sigma_values:
        d = results[Sigma]
        ir_local = np.median(d["T_eff_smooth"][d["ir_start"] : d["ir_end"]])
        uv_local = np.median(d["alpha_eff"][d["uv_start"] : d["uv_end"]])

        summary_rows.append(
            [
                Sigma,
                d["exact"],
                d["fit_slope"],
                100.0 * (d["fit_slope"] - d["exact"]) / d["exact"],
                ir_local,
                100.0 * (ir_local - d["exact"]) / d["exact"],
                d["alpha_exact"],
                d["alpha_fit"],
                100.0
                * (d["alpha_fit"] - d["alpha_exact"])
                / d["alpha_exact"],
                uv_local,
                100.0 * (uv_local - d["alpha_exact"]) / d["alpha_exact"],
                d["r2"],
                d["c_r2"],
                d["ell_s"][d["uv_start"]],
                d["ell_s"][d["uv_end"] - 1],
                d["ell_s"][d["ir_start"]],
                d["ell_s"][d["ir_end"] - 1],
            ]
        )

    print_box_table(
        title="IR tension and UV Coulomb coefficient comparison "
        " (smoothed IR selector)",
        headers=[
            "Σ",
            "Exact T",
            "IR fit T",
            "IR % err",
            "IR local T",
            "IR local % err",
            "Exact α",
            "UV fit α",
            "UV % err",
            "UV local α",
            "UV local % err",
            "IR fit R²",
            "UV fit R²",
            "UV ℓ min",
            "UV ℓ max",
            "IR ℓ min",
            "IR ℓ max",
        ],
        rows=summary_rows,
        fmt=[
            ".2f",
            ".8f",
            ".8f",
            "+.4f",
            ".8f",
            "+.4f",
            ".8f",
            ".8f",
            "+.4f",
            ".8f",
            "+.4f",
            ".6f",
            ".6f",
            ".6f",
            ".6f",
            ".6f",
            ".6f",
        ],
    )

    scan_rows = []

    for Sigma in Sigma_values:
        d = results[Sigma]
        scan_rows.append(
            [
                Sigma,
                np.sqrt(Sigma),
                d["U0_raw"][0],
                d["U0_raw"][-1],
                d["ell_s"][0],
                d["ell_s"][-1],
                len(d["U0_raw"]),
                len(d["ell_s"]),
            ]
        )

    print_box_table(
        title="Scanning range",
        headers=[
            "Σ",
            "U★",
            "U₀ min",
            "U₀ max",
            "ℓ min",
            "ℓ max",
            "raw pts",
            "kept pts",
        ],
        rows=scan_rows,
        fmt=[".2f", ".6f", ".6f", ".6f", ".6f", ".6f", ".0f", ".0f"],
    )

    print()
    print(
        f"  Grid: log-spaced in ε = U0−U★  |  U_max={cfg['U_max']:g}"
        f"  |  N_scan={cfg['N_scan']}  |  quad tol={cfg['quad_tol']:g}"
    )
    print(
        f"  UV selector: α_eff within {cfg['uv_alpha_tols']} of exact α, "
        f"skipping first {cfg['uv_skip_points']} pts, "
        f"U0 ≤ {cfg['uv_exclude_frac']} U_max"
    )
    print(
        f"  IR selector: smoothed T_eff within {cfg['ir_tension_tols']} of exact T"
        f"  |  smooth window = {cfg['ir_smooth_window']}"
    )

    for Sigma in Sigma_values:
        d = results[Sigma]
        uv_idx = np.unique(
            np.round(
                np.linspace(d["uv_start"], d["uv_end"] - 1, 3)
            ).astype(int)
        )
        ir_idx = np.unique(
            np.round(
                np.linspace(d["ir_start"], d["ir_end"] - 1, 3)
            ).astype(int)
        )
        mid_start = d["uv_end"]
        mid_end = d["ir_start"] - 1

        if mid_end >= mid_start:
            mid_idx = np.unique(
                np.round(np.linspace(mid_start, mid_end, 4)).astype(int)
            )
        else:
            mid_idx = np.unique(
                np.round(np.linspace(0, len(d["ell_s"]) - 1, 4)).astype(int)
            )

        idx = np.unique(np.concatenate([uv_idx, mid_idx, ir_idx]))
        rows = [[d["U0_s"][i], d["ell_s"][i], d["E_s"][i]] for i in idx]

        print_box_table(
            title=f"Sample table  —  Σ = {Sigma}",
            headers=["U₀", "ℓ(U₀)", "E(U₀)"],
            rows=rows,
            fmt=[".6f", ".6f", ".6f"],
        )

    convergence_rows = scanning_range_convergence_test(Sigma_values, cfg)

    print_box_table(
        title="Scanning-range convergence test",
        headers=[
            "Σ",
            "U_max",
            "U₀ min",
            "U₀ max",
            "ℓ min",
            "ℓ max",
            "Exact T",
            "Fit slope",
            "Fit % err",
            "Fit R²",
        ],
        rows=convergence_rows,
        fmt=[
            ".2f",
            ".2f",
            ".6f",
            ".6f",
            ".6f",
            ".6f",
            ".8f",
            ".8f",
            "+.4f",
            ".8f",
        ],
    )

    _plot_comparison(list(Sigma_values), results, plot_colours)

    for Sigma, colour in zip(Sigma_values, plot_colours):
        _plot_E_vs_ell(Sigma, results[Sigma], colour)

    _plot_uv_diagnostic(list(Sigma_values), results, plot_colours)
    _plot_ir_diagnostic(list(Sigma_values), results, plot_colours)

    return results

if __name__ == "__main__":
    run()
