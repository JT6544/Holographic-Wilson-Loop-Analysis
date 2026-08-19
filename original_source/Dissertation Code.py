import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.integrate import IntegrationWarning
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.special import gamma
from scipy.stats import linregress

import warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore", category=IntegrationWarning)


def default_config():
    return {
        'R': 1.0,

        'U_max': 20.0,
        'N_scan': 1000,
        'epsilon': 1e-8,
        'quad_tol': 1e-8,
        'quad_lim': 250,

        'uv_exclude_frac': 0.65,
        'uv_skip_points': 10,
        'uv_alpha_tols': [0.01, 0.015, 0.02],
        'uv_min_points': 12,

        'ir_tension_tols': [0.002, 0.003, 0.005],
        'ir_min_points': 30,
        'ir_smooth_window': 11,
    }

_CFG = default_config()

def exact_slope(Sigma,R=1.0):
    return Sigma / (np.pi * R**2)

def exact_alpha(R=1.0):
    return 4.0 * np.pi**2 * R**2 / gamma(0.25)**4

def F(U,Sigma):
    return U**2 + (Sigma**2 / U**2)

def make_U0_grid(Sigma,cfg=None):
    if cfg is None:
        cfg = _CFG

    U_star = np.sqrt(Sigma)
    eps_max = cfg['U_max'] - U_star
    eps_grid = np.logspace(np.log10(cfg['epsilon']), np.log10(eps_max), cfg['N_scan'])
    return U_star + eps_grid

def _integrate_one(U0,Sigma,U_star,U_max,R,quad_lim,quad_tol):
    F0 = F(U0,Sigma)
    t_max = np.sqrt(max(U_max - U0, 0.0))

    def integrand_ell(t):
        FU = F(U0 + t * t,Sigma)
        arg = FU**2 - F0**2
        return 0.0 if arg <= 0.0 else 2.0 * t / (FU * np.sqrt(arg))

    def integrand_E(t):
        FU = F(U0 + t * t,Sigma)
        arg = FU**2 - F0**2
        return 0.0 if arg <= 0.0 else 2.0 * t * (FU / np.sqrt(arg) - 1.0)

    kw = dict(limit=quad_lim, epsabs=quad_tol, epsrel=quad_tol)

    try:
        val_l, _ = integrate.quad(integrand_ell, 0.0, t_max, **kw)
        val_E, _ = integrate.quad(integrand_E, 0.0, t_max, **kw)
    except Exception:
        return np.nan, np.nan

    ell = 2.0 * R**2 * F0 * val_l
    E = (1.0 / np.pi) * (val_E - (U0 - U_star))
    return ell, E

def compute_curves(Sigma,cfg=None):
    if cfg is None:
        cfg = _CFG

    U_star = np.sqrt(Sigma)
    U0_arr = make_U0_grid(Sigma,cfg)

    args = [(U0, Sigma, U_star, cfg['U_max'], cfg['R'], cfg['quad_lim'], cfg['quad_tol']) for U0 in U0_arr]

    with ProcessPoolExecutor() as pool: # Paralleises integration
        pairs = list(pool.map(_integrate_one, *zip(*args)))

    ell_arr = np.array([p[0] for p in pairs])
    E_arr = np.array([p[1] for p in pairs])

    return U0_arr, ell_arr, E_arr

def clean_and_sort(U0_arr,ell_arr,E_arr): # Remove NaN / non-positive-ell points and sort surviving data by ell.
    mask = (np.isfinite(U0_arr) & np.isfinite(ell_arr) & np.isfinite(E_arr) & (ell_arr > 0))
    idx = np.argsort(ell_arr[mask])
    return U0_arr[mask][idx], ell_arr[mask][idx], E_arr[mask][idx]

def _fit(x,y):
    slope, intercept, r, *_ = linregress(x,y)
    return slope, intercept, r**2

def contiguous_blocks(mask): # Builds region to test fits later
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

def local_dE_dell(ell_sorted,E_sorted):
    return np.gradient(E_sorted, ell_sorted)

def smooth_signal(arr,window): # Smooths by IR Region mostly 
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    return uniform_filter1d(median_filter(arr, size=window, mode='nearest'), size=window, mode='nearest',)

def find_coulomb_region(U0_sorted,ell_sorted,E_sorted,cfg=None): # Tries tolerances from cfg['uv_alpha_tols'] tightest-first, accepting the first one that yields a contiguous block of at least cfg['uv_min_points']
    if cfg is None:
        cfg = _CFG

    alpha_ex = exact_alpha(cfg['R'])
    alpha_eff = ell_sorted**2 * local_dE_dell(ell_sorted,E_sorted)

    base_mask = ((np.arange(len(ell_sorted)) >= cfg['uv_skip_points'] & (U0_sorted <= cfg['uv_exclude_frac'] * cfg['U_max']) & np.isfinite(alpha_eff))) # Eligible points

    for tol in cfg['uv_alpha_tols']:
        mask = base_mask & (np.abs(alpha_eff - alpha_ex) / alpha_ex <= tol)
        groups = [g for g in contiguous_blocks(mask) if len(g) >= cfg['uv_min_points']]
        if groups:
            chosen = max(groups, key=len)
            break

    s, e = chosen[0], chosen[-1] + 1
    slope, intercept, r2 = _fit(1.0 / ell_sorted[s:e], E_sorted[s:e])
    alpha_fit = -slope
    rel_err = abs(alpha_fit - alpha_ex) / alpha_ex

    return s, e, alpha_fit, intercept, r2, rel_err, alpha_eff

def find_linear_region(ell_sorted,E_sorted,Sigma,cfg=None):
    if cfg is None:
        cfg = _CFG

    T_ex = exact_slope(Sigma,cfg['R'])
    T_eff_raw = local_dE_dell(ell_sorted,E_sorted)
    T_eff_smooth = smooth_signal(T_eff_raw, cfg['ir_smooth_window'])

    for tol in cfg['ir_tension_tols']:
        mask = np.isfinite(T_eff_smooth) & (np.abs(T_eff_smooth - T_ex) / T_ex <= tol)
        groups = [g for g in contiguous_blocks(mask) if len(g) >= cfg['ir_min_points']]
        if groups:
            chosen = max(groups, key=lambda g: (len(g), g[-1]))
            break

    s, e = chosen[0], chosen[-1] + 1
    slope, intercept, r2 = _fit(ell_sorted[s:e], E_sorted[s:e])
    rel_err = abs(slope - T_ex) / T_ex

    return s, e, slope, intercept, r2, rel_err, T_eff_raw, T_eff_smooth

def _hline(col_widths,left='├',mid='┼',right='┤'):
    return left + mid.join('─' * (w + 2) for w in col_widths) + right

def print_box_table(title,headers,rows,fmt=None):
    if fmt is None:
        fmt = ['.6g'] * len(headers)

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
        return '│' + '│'.join(f' {c:>{w}} ' for c, w in zip(cells, cw)) + '│'

    top = '┌' + '┬'.join('─' * (w + 2) for w in cw) + '┐'
    bot = '└' + '┴'.join('─' * (w + 2) for w in cw) + '┘'

    print(f'\n  {title}')
    print(top)
    print(_row(headers))
    print(_hline(cw))

    for r in str_rows:
        print(_row(r))

    print(bot)

COLOURS = ['navy', 'darkred', 'darkgreen', 'indigo']

def _plot_E_vs_ell(Sigma,d,colour):
    x, y = d['ell_s'], d['E_s']
    s_uv, e_uv = d['uv_start'], d['uv_end']
    s_ir, e_ir = d['ir_start'], d['ir_end']
    x_uv, y_uv = x[s_uv:e_uv], y[s_uv:e_uv]
    x_ir, y_ir = x[s_ir:e_ir], y[s_ir:e_ir]
    T_ex, alpha_ex = d['exact'], d['alpha_exact']

    b_ir_exact = np.mean(y_ir - T_ex * x_ir)
    b_uv_exact = np.mean(y_uv + alpha_ex / x_uv)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, y, color=colour, linewidth=2.0, zorder=4, label=rf'$\Sigma = {Sigma}$')
    ax.axvspan(x_uv[0], x_uv[-1], color='deepskyblue', alpha=0.16, zorder=1, label='Coulomb region')
    ax.axvspan(x_ir[0], x_ir[-1], color='gold', alpha=0.16, zorder=1, label='linear region')

    ax.plot(x_uv, -d['alpha_fit'] / x_uv + d['c_intercept'], '--', color='teal', linewidth=2.0, zorder=5, label=rf"UV fit: $\alpha={d['alpha_fit']:.5f}$, $R^2={d['c_r2']:.5f}$")
    ax.plot(x_uv, -alpha_ex / x_uv + b_uv_exact, '-', color='deepskyblue', linewidth=2.2, zorder=6, label=rf'UV exact: $\alpha={alpha_ex:.5f}$')
    ax.plot(x_ir, d['fit_slope'] * x_ir + d['intercept'], '--', color='grey', linewidth=1.8, zorder=5, label=rf"IR fit: $T={d['fit_slope']:.5f}$, $R^2={d['r2']:.5f}$")
    ax.plot(x_ir, T_ex * x_ir + b_ir_exact,'-', color='black', linewidth=2.2, zorder=6,label=rf'IR exact: $T={T_ex:.5f}$')

    ax.set_xlabel(r'$\ell$', fontsize=13)
    ax.set_ylabel(r'$E$', fontsize=13)
    ax.set_title(rf'$E$ vs $\ell$  —  $\Sigma = {Sigma}$', fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_uv_diagnostic(Sigma_values,results,colours):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, Sigma, colour in zip(axes.flat, Sigma_values, colours):
        d = results[Sigma]
        x = 1.0 / d['ell_s']
        y = d['E_s']
        x_uv = x[d['uv_start']:d['uv_end']]
        y_uv = y[d['uv_start']:d['uv_end']]
        alpha_ex = d['alpha_exact']
        b_uv_exact = np.mean(y_uv + alpha_ex * x_uv)

        ax.plot(x, y, color=colour, linewidth=1.9, label=rf'$\Sigma = {Sigma}$')
        ax.axvspan(x_uv[0], x_uv[-1], color='deepskyblue', alpha=0.16, label='Coulomb region')
        ax.plot(x_uv, -d['alpha_fit'] * x_uv + d['c_intercept'],'--', color='teal', linewidth=1.8,label=rf"fit: $\alpha={d['alpha_fit']:.5f}$")
        ax.plot(x_uv, -alpha_ex * x_uv + b_uv_exact,'-', color='black', linewidth=2.0,label=rf'exact: $\alpha={alpha_ex:.5f}$')
        ax.set_xlabel(r'$1/\ell$', fontsize=11)
        ax.set_ylabel(r'$E$', fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(r'UV diagnostic: linearity of $E$ versus $1/\ell$', fontsize=13)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_ir_diagnostic(Sigma_values,results,colours):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, Sigma, colour in zip(axes.flat, Sigma_values, colours):
        d = results[Sigma]
        x = d['ell_s']
        T_ex = d['exact']
        s_ir, e_ir = d['ir_start'], d['ir_end']

        trim = 80 if np.isclose(Sigma, 4.0) else 0

        if trim > 0:
            x_plot = x[:-trim]
            T_plot = d['T_eff_smooth'][:-trim]
        else:
            x_plot = x
            T_plot = d['T_eff_smooth']

        ax.plot(x_plot, T_plot, color=colour, linewidth=1.8, label=rf'$T_{{\rm eff}}$, $\Sigma={Sigma}$')
        ax.axhline(T_ex, color='black', linestyle='--', linewidth=1.5, label=rf'exact $T={T_ex:.6f}$')
        ax.axvspan(x[s_ir], x[e_ir - 1], color='gold', alpha=0.16, label='linear region')
        ax.set_xlabel(r'$\ell$', fontsize=11)
        ax.set_ylabel(r'$dE/d\ell$', fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(r'IR diagnostic: smoothed convergence of $dE/d\ell$ to the exact tension', fontsize=13)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_comparison(Sigma_values,results,colours):
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    for Sigma, colour in zip(Sigma_values, colours):
        d = results[Sigma]
        x = d['ell_s']
        y = d['E_s']
        s_ir, e_ir = d['ir_start'], d['ir_end']
        x_ir, y_ir = x[s_ir:e_ir], y[s_ir:e_ir]
        T_ex = d['exact']
        b_ir_exact = np.mean(y_ir - T_ex * x_ir)

        ax.plot(x, y, color=colour, linewidth=2.0, label=rf'$\Sigma={Sigma}$')
        ax.plot(x_ir, T_ex * x_ir + b_ir_exact, '--', color=colour, linewidth=1.6, alpha=0.9)

    ax.set_xlabel(r'$\ell$', fontsize=13)
    ax.set_ylabel(r'$E$', fontsize=13)
    ax.set_title(r'Comparison plot: numerical $E(\ell)$ with exact IR asymptotes', fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def scanning_range_convergence_test(Sigma_values,cfg=None,U_max_values=None):
    if cfg is None:
        cfg = _CFG

    if U_max_values is None:
        U_max_values = (0.5 * cfg['U_max'], 0.75 * cfg['U_max'], cfg['U_max'], 1.25 * cfg['U_max'])

    rows = []

    for Sigma in Sigma_values:
        T_ex = exact_slope(Sigma,cfg['R'])

        for U_max in U_max_values:
            cfg_test = dict(cfg)
            cfg_test['U_max'] = float(U_max)

            U0_raw, ell_raw, E_raw = compute_curves(Sigma,cfg_test)
            U0_s, ell_s, E_s = clean_and_sort(U0_raw,ell_raw,E_raw)
            _, _, fit_slope, _, r2, _, _, _ = find_linear_region(ell_s,E_s,Sigma,cfg_test)

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

def run(Sigma_values=(0.5, 1.0, 2.0, 4.0),cfg=None,colours=None):
    if cfg is None:
        cfg = _CFG

    if colours is None:
        colours = COLOURS

    results = {}

    for Sigma in Sigma_values:
        U0_raw, ell_raw, E_raw = compute_curves(Sigma,cfg)
        U0_s, ell_s, E_s = clean_and_sort(U0_raw,ell_raw,E_raw)

        uv_s, uv_e, alpha_fit, c_intercept, c_r2, alpha_rel_err, alpha_eff = \
            find_coulomb_region(U0_s,ell_s,E_s,cfg)

        ir_s, ir_e, fit_slope, intercept, r2, ir_rel_err, T_eff_raw, T_eff_smooth = \
            find_linear_region(ell_s,E_s,Sigma,cfg)

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
            alpha_exact=exact_alpha(cfg['R']),
            alpha_rel_err=alpha_rel_err,
            T_eff_raw=T_eff_raw,
            T_eff_smooth=T_eff_smooth,
            alpha_eff=alpha_eff,
            exact=exact_slope(Sigma,cfg['R']),
        )

    summary_rows = []

    for Sigma in Sigma_values:
        d = results[Sigma]
        ir_local = np.median(d['T_eff_smooth'][d['ir_start']:d['ir_end']])
        uv_local = np.median(d['alpha_eff'][d['uv_start']:d['uv_end']])

        summary_rows.append([
            Sigma,
            d['exact'],       d['fit_slope'],  100.0 * (d['fit_slope']  - d['exact'])       / d['exact'],
            ir_local,         100.0 * (ir_local        - d['exact'])       / d['exact'],
            d['alpha_exact'], d['alpha_fit'],  100.0 * (d['alpha_fit']  - d['alpha_exact']) / d['alpha_exact'],
            uv_local,         100.0 * (uv_local        - d['alpha_exact']) / d['alpha_exact'],
            d['r2'], d['c_r2'],
            d['ell_s'][d['uv_start']], d['ell_s'][d['uv_end'] - 1],
            d['ell_s'][d['ir_start']], d['ell_s'][d['ir_end'] - 1],
        ])

    print_box_table(
        title='IR tension and UV Coulomb coefficient comparison  (smoothed IR selector)',
        headers=[
            'Σ',
            'Exact T',  'IR fit T',   'IR % err', 'IR local T', 'IR local % err',
            'Exact α',  'UV fit α',   'UV % err', 'UV local α', 'UV local % err',
            'IR fit R²', 'UV fit R²',
            'UV ℓ min', 'UV ℓ max',  'IR ℓ min', 'IR ℓ max',
        ],
        rows=summary_rows,
        fmt=[
            '.2f',
            '.8f', '.8f', '+.4f', '.8f', '+.4f',
            '.8f', '.8f', '+.4f', '.8f', '+.4f',
            '.6f', '.6f',
            '.6f', '.6f', '.6f', '.6f',
        ],
    )

    scan_rows = []

    for Sigma in Sigma_values:
        d = results[Sigma]
        scan_rows.append([
            Sigma,
            np.sqrt(Sigma),
            d['U0_raw'][0],
            d['U0_raw'][-1],
            d['ell_s'][0],
            d['ell_s'][-1],
            len(d['U0_raw']),
            len(d['ell_s']),
        ])

    print_box_table(
        title='Scanning range',
        headers=['Σ', 'U★', 'U₀ min', 'U₀ max', 'ℓ min', 'ℓ max', 'raw pts', 'kept pts'],
        rows=scan_rows,
        fmt=['.2f', '.6f', '.6f', '.6f', '.6f', '.6f', '.0f', '.0f'],
    )

    print()
    print(
        f"  Grid: log-spaced in ε = U0−U★  |  U_max={cfg['U_max']:g}"
        f"  |  N_scan={cfg['N_scan']}  |  quad tol={cfg['quad_tol']:g}"
    )
    print(
        f"  UV selector: α_eff within {cfg['uv_alpha_tols']} of exact α, "
        f"skipping first {cfg['uv_skip_points']} pts, U0 ≤ {cfg['uv_exclude_frac']} U_max"
    )
    print(
        f"  IR selector: smoothed T_eff within {cfg['ir_tension_tols']} of exact T"
        f"  |  smooth window = {cfg['ir_smooth_window']}"
    )

    for Sigma in Sigma_values:
        d = results[Sigma]
        uv_idx = np.unique(np.round(np.linspace(d['uv_start'], d['uv_end'] - 1, 3)).astype(int))
        ir_idx = np.unique(np.round(np.linspace(d['ir_start'], d['ir_end'] - 1, 3)).astype(int))
        mid_start = d['uv_end']
        mid_end = d['ir_start'] - 1

        if mid_end >= mid_start:
            mid_idx = np.unique(np.round(np.linspace(mid_start, mid_end, 4)).astype(int))
        else:
            mid_idx = np.unique(np.round(np.linspace(0, len(d['ell_s']) - 1, 4)).astype(int))

        idx = np.unique(np.concatenate([uv_idx, mid_idx, ir_idx]))
        rows = [[d['U0_s'][i], d['ell_s'][i], d['E_s'][i]] for i in idx]

        print_box_table(
            title=f'Sample table  —  Σ = {Sigma}',
            headers=['U₀', 'ℓ(U₀)', 'E(U₀)'],
            rows=rows,
            fmt=['.6f', '.6f', '.6f'],
        )

    convergence_rows = scanning_range_convergence_test(Sigma_values,cfg)

    print_box_table(
        title='Scanning-range convergence test',
        headers=['Σ', 'U_max', 'U₀ min', 'U₀ max', 'ℓ min', 'ℓ max', 'Exact T', 'Fit slope', 'Fit % err', 'Fit R²'],
        rows=convergence_rows,
        fmt=['.2f', '.2f', '.6f', '.6f', '.6f', '.6f', '.8f', '.8f', '+.4f', '.8f'],
    )

    _plot_comparison(list(Sigma_values), results, colours)

    for Sigma, colour in zip(Sigma_values, colours):
        _plot_E_vs_ell(Sigma, results[Sigma], colour)

    _plot_uv_diagnostic(list(Sigma_values), results, colours)
    _plot_ir_diagnostic(list(Sigma_values), results, colours)

    return results

if __name__ == '__main__':
    run()