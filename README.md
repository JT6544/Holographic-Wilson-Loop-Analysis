# Holographic Wilson-Loop Numerical Analysis

This project numerically reconstructs the static quark-antiquark potential $E(\ell)$ in a deformed asymptotically anti-de Sitter background.

The calculation is intended to verify that the same holographic geometry reproduces two different physical regimes:

- **Ultraviolet (UV):** a Coulomb-like potential at short quark-antiquark separation;
- **Infrared (IR):** a linearly rising potential at large separation, indicating confinement within the model.

The code evaluates the boundary separation and renormalised energy parametrically as functions of the string turning point $U_0$. It then identifies suitable UV and IR fitting regions, compares the fitted numerical coefficients with exact analytical predictions, performs a cutoff-convergence check, and produces diagnostic tables and plots.

## Repository Contents

```text
.
├── .gitignore
├── README.md
├── requirements.txt
└── wilson_loop_analysis.py
```

The repository deliberately uses a simple structure. The numerical study is self-contained in one Python module and does not require an external dataset.

## Requirements

The project has been verified using:

| Component | Tested version |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |

The tested package versions are recorded in `requirements.txt`.

## Installation

From the repository directory, create and activate a virtual environment.

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Analysis

Run the complete default study from the repository root:

```bash
python wilson_loop_analysis.py
```

No command-line arguments or input files are required. The default run:

- evaluates four deformation values;
- prints the numerical comparison and convergence tables;
- constructs seven Matplotlib figures;
- returns no saved files by default.

On the verified environment, a complete non-interactive run took approximately one minute. Runtime depends on the available CPU resources because the integrations are evaluated in parallel.

The script displays figures using `plt.show()`. When using an interactive Matplotlib backend, close each displayed figure to allow the program to continue.

## Physical Background

The model uses the deformed warp factor

$$
F(U)=U^2+\frac{\Sigma^2}{U^2},
$$

where:

- $U$ is the holographic radial coordinate;
- $\Sigma$ controls the strength of the infrared deformation;
- $R$ is the AdS radius.

At large $U$,

$$
F(U)\sim U^2,
$$

so the geometry remains asymptotically AdS in the UV.

The warp factor has a minimum at

$$
U_\star=\sqrt{\Sigma},
$$

with

$$
F(U_\star)=2\Sigma.
$$

This minimum provides a preferred radial position for the string worldsheet. At large boundary separation, the string approaches this position and develops a long approximately horizontal section, producing a linear contribution to the energy.

The analytical string tension is

$$
T_{\mathrm{string}} = \frac{\Sigma}{\pi R^2}.
$$

At short separation, the deformation becomes negligible and the potential approaches the AdS Coulomb form

$$
E(\ell)
\sim
-\frac{\alpha}{\ell}+c_{\mathrm{UV}},
$$

where

$$
\alpha_{\mathrm{exact}} = \frac{4\pi^2R^2}{\Gamma\!\left(\frac14\right)^4}.
$$

For the default value $R=1$,

$$
\alpha_{\mathrm{exact}}
\approx 0.22847329.
$$

The additive constants in the UV and IR expressions depend on the subtraction convention. The physically important quantities are therefore the Coulomb coefficient and the large-distance slope.

## Main Aim

The numerical analysis tests whether the reconstructed potential has the asymptotic forms

$$
E(\ell)
\sim
-\frac{\alpha_{\mathrm{exact}}}{\ell}
+c_{\mathrm{UV}}
\qquad
(\ell\to0)
$$

and

$$
E(\ell)
\sim
T_{\mathrm{string}}\ell
+c_{\mathrm{IR}}
\qquad
(\ell\to\infty).
$$

The default deformation values are

```python
Sigma_values = (0.5, 1.0, 2.0, 4.0)
```

This allows the code to test two expected features:

1. the UV coefficient should remain approximately independent of $\Sigma$;
2. the IR string tension should increase linearly with $\Sigma$.

## What the Code Does

For each value of $\Sigma$, the program:

1. calculates the preferred IR position $U_\star=\sqrt{\Sigma}$;
2. constructs a logarithmically spaced grid of turning points $U_0$;
3. evaluates the parametric functions $\ell(U_0)$ and $E(U_0)$;
4. removes invalid numerical values;
5. sorts the remaining data by increasing $\ell$;
6. calculates the local derivative $dE/d\ell$;
7. identifies a Coulombic UV region;
8. identifies a linear IR region;
9. performs linear regressions in the selected windows;
10. compares the fitted coefficients with exact analytical values;
11. checks that the UV and IR windows do not overlap;
12. repeats the IR calculation for several values of $U_{\max}$;
13. prints numerical summaries and displays plots;
14. returns the complete results in a dictionary.


## Numerical Reconstruction

### Turning-point grid

The turning point is parameterised as

$$
U_0=U_\star+\epsilon,
$$

where $\epsilon>0$.

The code samples $\epsilon$ logarithmically:

```python
eps_grid = np.logspace(
    np.log10(cfg["epsilon"]),
    np.log10(cfg["U_max"] - U_star),
    cfg["N_scan"],
)
```

This is useful because the large-$\ell$ behaviour occurs when

$$
U_0\to U_\star.
$$

A uniformly spaced grid would place relatively few points close to this sensitive IR region. The logarithmic grid concentrates points near the minimum while still extending into the UV.

The settings should satisfy

$$
U_{\max}>\sqrt{\Sigma}
$$

for every value of $\Sigma$.

### Parametric potential

The complete potential is not reconstructed from a closed analytical expression for $E(\ell)$. Instead, the code calculates

$$
\ell=\ell(U_0),
\qquad
E=E(U_0),
$$

and eliminates $U_0$ numerically by sorting the computed pairs according to $\ell$.

The resulting arrays provide a numerical representation of the potential across the UV-to-IR crossover.

### Endpoint regularisation

The radial integrals contain a square-root singularity at the lower limit $U=U_0$. The code introduces

$$
U=U_0+t^2,
\qquad
dU=2t\,dt.
$$

The integration range becomes

$$
0\leq t\leq\sqrt{U_{\max}-U_0}.
$$

This substitution removes the explicit endpoint singularity and improves the behaviour of the adaptive quadrature.

With

$$
F_0=F(U_0),
$$

the numerical separation is calculated from

$$
\ell(U_0)=2R^2F_0\int_0^{\sqrt{U_{\max}-U_0}}\frac{2t\,dt}{F(U_0+t^2)\sqrt{F(U_0+t^2)^2-F_0^2}}.
$$

The renormalised energy is calculated using

$$
E(U_0)=\frac{1}{\pi}\left[\int_0^{\sqrt{U_{\max}-U_0}}2t\left(\frac{F(U_0+t^2)}{\sqrt{F(U_0+t^2)^2-F_0^2}}-1\right)dt-(U_0-U_\star)\right].
$$

Both integrals are evaluated using `scipy.integrate.quad`.

### Parallel evaluation

Every value of $U_0$ can be integrated independently. The code therefore distributes the integrations using

```python
ProcessPoolExecutor
```

inside `compute_curves()`.

This reduces the total execution time for large scans, especially because each turning point requires two adaptive integrations.

If an individual integration fails, the helper function returns `NaN`. These values are later removed by `clean_and_sort()`.

SciPy `IntegrationWarning` events are captured inside each worker rather than suppressed globally. After each curve is calculated, the program reports:

- the number of turning points that generated warnings;
- the total number of warning events;
- the number of failed integrations;
- the exception types associated with failures, if any.

The default verification run reported warning events near numerically sensitive regions but no failed integrations. See [Integration warnings](#integration-warnings) for interpretation.

## Data Cleaning and Derivatives

After the integrations are complete, the code removes points where:

- $U_0$ is not finite;
- $\ell$ is not finite;
- $E$ is not finite;
- $\ell\leq0$.

The surviving data are sorted by increasing separation.

The local derivative is then estimated using

```python
np.gradient(E_sorted, ell_sorted)
```

which approximates

$$
\frac{dE}{d\ell}.
$$

This derivative is used in both the UV and IR diagnostics.

Because numerical gradients can be noisy, particularly near the ends of the dataset, the IR derivative is smoothed using:

1. a median filter;
2. a uniform moving-average filter.

The UV diagnostic uses the unsmoothed derivative.

## UV Coulomb Region

The effective Coulomb coefficient is defined as

$$
\alpha_{\mathrm{eff}}(\ell)=\ell^2\frac{dE}{d\ell}.
$$

If

$$
E(\ell)=-\frac{\alpha}{\ell}+c,
$$

then

$$
\frac{dE}{d\ell}=\frac{\alpha}{\ell^2}
$$

and therefore

$$
\alpha_{\mathrm{eff}}(\ell)\to\alpha.
$$

The code tests a series of relative tolerances:

```python
uv_alpha_tols = [0.01, 0.015, 0.02]
```

The tightest tolerance is tested first. Points are accepted when

$$
\frac{\left|\alpha_{\mathrm{eff}}-\alpha_{\mathrm{exact}}\right|}{\alpha_{\mathrm{exact}}}\leq\text{tolerance}.
$$

The accepted points are divided into contiguous blocks. A valid block must contain at least

```python
uv_min_points = 12
```

points.

The longest valid block is selected and fitted as a straight line in $E$ against $1/\ell$:

$$
E=m\left(\frac{1}{\ell}\right)+c.
$$

The numerical Coulomb coefficient is then

$$
\alpha_{\mathrm{fit}}=-m.
$$

The code reports:

- $\alpha_{\mathrm{exact}}$;
- $\alpha_{\mathrm{fit}}$;
- percentage error;
- local median $\alpha_{\mathrm{eff}}$;
- UV fit $R^2$;
- selected $\ell$ range.

## IR Linear Region

The local effective string tension is

$$
T_{\mathrm{eff}}(\ell)=\frac{dE}{d\ell}.
$$

The analytical target is

$$
T_{\mathrm{exact}}=\frac{\Sigma}{\pi R^2}.
$$

The code compares the smoothed derivative with this value using

```python
ir_tension_tols = [0.002, 0.003, 0.005]
```

and accepts points satisfying

$$
\frac{\left|T_{\mathrm{eff}}-T_{\mathrm{exact}}\right|}{T_{\mathrm{exact}}}\leq\text{tolerance}.
$$

A valid IR region must contain at least

```python
ir_min_points = 30
```

contiguous points.

The chosen region is fitted using

$$
E(\ell)=T_{\mathrm{fit}}\ell+c.
$$

The code reports:

- $T_{\mathrm{exact}}$;
- $T_{\mathrm{fit}}$;
- percentage error;
- local median $T_{\mathrm{eff}}$;
- IR fit $R^2$;
- selected $\ell$ range.

The selected UV and IR windows are checked for overlap. If they overlap, the program raises a `RuntimeError`.

## Default Configuration

The main settings are stored in `default_config()`.

| Setting | Default | Description |
|---|---:|---|
| `R` | `1.0` | AdS radius |
| `U_max` | `20.0` | Finite upper integration cutoff |
| `N_scan` | `1000` | Number of turning-point samples |
| `epsilon` | `1e-8` | Minimum value of $U_0-U_\star$ |
| `quad_tol` | `1e-8` | Absolute and relative quadrature tolerance |
| `quad_lim` | `250` | Maximum adaptive quadrature subdivisions |
| `uv_exclude_frac` | `0.65` | Upper turning-point fraction used in UV eligibility |
| `uv_skip_points` | `10` | Number of initial sorted points excluded from UV selection |
| `uv_alpha_tols` | `[0.01, 0.015, 0.02]` | UV relative tolerances |
| `uv_min_points` | `12` | Minimum UV block length |
| `ir_tension_tols` | `[0.002, 0.003, 0.005]` | IR relative tolerances |
| `ir_min_points` | `30` | Minimum IR block length |
| `ir_smooth_window` | `11` | Smoothing window used for the IR derivative |

A custom configuration can be created using

```python
from wilson_loop_analysis import default_config, run


def main():
    cfg = default_config()

    cfg.update(
        {
            "N_scan": 800,
            "U_max": 20.0,
            "quad_tol": 1e-8,
            "ir_smooth_window": 11,
        }
    )

    results = run(cfg=cfg)
    return results


if __name__ == "__main__":
    main()
```

The main guard is important on platforms that use process spawning for `ProcessPoolExecutor`.

Reducing `N_scan` or loosening `quad_tol` makes exploratory runs faster, but may reduce the stability of the derivative-based selectors. Configuration changes can also alter selected fit windows and numerical results.

## Convergence Test

The radial integrals formally extend to infinity, but the numerical implementation uses a finite cutoff $U_{\max}$.

The function

```python
scanning_range_convergence_test()
```

repeats the IR extraction using several cutoff values:

$$
0.5U_{\max},
\quad
0.75U_{\max},
\quad
U_{\max},
\quad
1.25U_{\max}.
$$

For each cutoff, it records:

- the scanned $U_0$ range;
- the resulting $\ell$ range;
- the exact string tension;
- the fitted IR slope;
- percentage error;
- fit $R^2$.

This provides a direct check that the large-distance slope is not strongly dependent on the chosen finite UV cutoff.

## Terminal Output

The program prints several formatted tables.

### UV and IR comparison table

This includes:

- exact and fitted IR tension;
- global-fit IR percentage error;
- median local IR tension;
- exact and fitted UV coefficient;
- global-fit UV percentage error;
- median local UV coefficient;
- UV and IR $R^2$ values;
- selected UV and IR $\ell$ ranges.

### Scanning-range table

This reports:

- $\Sigma$;
- $U_\star$;
- minimum and maximum $U_0$;
- minimum and maximum $\ell$;
- number of raw points;
- number of retained points.

### Sample data tables

For each value of $\Sigma$, representative points are selected from the UV, crossover, and IR sections of the reconstructed curve.

Each row contains

$$
U_0,
\qquad
\ell(U_0),
\qquad
E(U_0).
$$

### Cutoff-convergence table

This compares the fitted string tension across different values of $U_{\max}$.

## Plots

The program displays the following figures.

### Combined potential comparison

A single figure compares $E(\ell)$ for all values of $\Sigma$, together with the exact IR asymptotic lines.

This shows that increasing $\Sigma$ increases the large-distance slope.

### Individual potential plots

For each value of $\Sigma$, the code plots:

- the full reconstructed potential;
- the selected UV region;
- the selected IR region;
- the fitted Coulomb curve;
- the exact Coulomb curve;
- the fitted linear curve;
- the exact linear asymptote.

The UV and IR regions are shown using shaded intervals.

### UV diagnostic

The UV diagnostic plots $E$ against $1/\ell$ over a zoomed window surrounding the selected Coulomb region.

A Coulomb potential is linear in this representation:

$$
E=-\alpha\left(\frac{1}{\ell}\right)+c.
$$

The fitted and exact slopes can therefore be compared visually without the extremely small-$\ell$ endpoint values compressing the scientifically relevant window.

### IR diagnostic

The IR diagnostic plots the smoothed local derivative

$$
\frac{dE}{d\ell}
$$

against $\ell$, together with the exact tension. Each panel is displayed over the selected IR fit region with a small amount of preceding context.

The approach of the numerical derivative to the horizontal analytical line provides a local test of confinement. The zoom changes only the displayed range; it does not alter the derivative arrays, selected indices, fits, or returned results.

The figures are shown using `plt.show()` and are not saved automatically.

To save a figure, add a line such as

```python
fig.savefig(
    "potential.png",
    dpi=300,
    bbox_inches="tight",
)
```

before `plt.show()`.

## Returned Results

The `run()` function returns a dictionary keyed by $\Sigma$:

```python
results = run()
data = results[1.0]
```

Each dataset contains the following values.

| Key | Description |
|---|---|
| `U0_raw` | Original turning-point grid |
| `ell_raw` | Raw separation values |
| `E_raw` | Raw energy values |
| `U0_s` | Cleaned turning points sorted by $\ell$ |
| `ell_s` | Cleaned and sorted separation values |
| `E_s` | Cleaned and sorted energy values |
| `uv_start` | First index of the selected UV region |
| `uv_end` | Exclusive final index of the UV region |
| `ir_start` | First index of the selected IR region |
| `ir_end` | Exclusive final index of the IR region |
| `alpha_fit` | Fitted UV Coulomb coefficient |
| `alpha_exact` | Exact UV coefficient |
| `alpha_rel_err` | Relative UV fit error |
| `ir_rel_err` | Relative IR fit error |
| `c_intercept` | UV fit intercept |
| `c_r2` | UV fit $R^2$ |
| `fit_slope` | Fitted IR tension |
| `exact` | Exact IR tension |
| `intercept` | IR fit intercept |
| `r2` | IR fit $R^2$ |
| `alpha_eff` | Full effective Coulomb-coefficient array |
| `T_eff_raw` | Raw local derivative |
| `T_eff_smooth` | Smoothed local derivative |

Example:

```python
data = results[1.0]

print("Exact tension:", data["exact"])
print("Fitted tension:", data["fit_slope"])
print("IR R²:", data["r2"])

print("Exact alpha:", data["alpha_exact"])
print("Fitted alpha:", data["alpha_fit"])
print("UV R²:", data["c_r2"])
```

The selected windows can be extracted with

```python
uv_ell = data["ell_s"][data["uv_start"] : data["uv_end"]]
uv_E = data["E_s"][data["uv_start"] : data["uv_end"]]

ir_ell = data["ell_s"][data["ir_start"] : data["ir_end"]]
ir_E = data["E_s"][data["ir_start"] : data["ir_end"]]
```

## Main Functions

| Function | Description |
|---|---|
| `default_config()` | Returns the default numerical settings |
| `exact_slope()` | Calculates the analytical IR string tension |
| `exact_alpha()` | Calculates the analytical UV coefficient |
| `F()` | Evaluates the deformed warp factor |
| `make_U0_grid()` | Constructs the logarithmic turning-point grid |
| `_integrate_one()` | Evaluates $\ell$ and $E$ for one turning point |
| `compute_curves()` | Parallelises the integrations for one $\Sigma$ |
| `clean_and_sort()` | Removes invalid values and sorts by separation |
| `local_dE_dell()` | Calculates the numerical derivative |
| `smooth_signal()` | Smooths the IR derivative |
| `find_coulomb_region()` | Detects and fits the UV region |
| `find_linear_region()` | Detects and fits the IR region |
| `scanning_range_convergence_test()` | Tests sensitivity to $U_{\max}$ |
| `run()` | Executes the full analysis |

Functions beginning with an underscore are internal helper functions.

## GitHub Cleanup and Verification

The repository preserves the numerical model, formulas, parameters, fit settings, and conclusions of the original dissertation analysis. Later repository preparation was limited to traceable maintenance and documentation work, including:

- renaming the working-copy script to an importable filename;
- adding validation and descriptive error handling;
- adding docstrings and restrained PEP 8 formatting;
- declaring the tested dependencies;
- exposing previously suppressed integration diagnostics;
- improving diagnostic display ranges without changing plotted quantities;
- correcting the UV eligibility-mask parentheses.

The corrected mask is

```python
base_mask = (
    (np.arange(len(ell_sorted)) >= cfg["uv_skip_points"])
    & (U0_sorted <= cfg["uv_exclude_frac"] * cfg["U_max"])
    & np.isfinite(alpha_eff)
)
```

The original expression was affected by operator precedence. For the default configuration, the corrected mask selects the same UV windows and reproduces the same fitted coefficients at full floating-point precision. It prevents the eligibility rules from being ignored under other configurations.

The complete cleaned project was compared with an untouched copy of the uploaded research code. The default UV and IR selector indices, fitted coefficients, retained point counts, and printed numerical results were unchanged.

## Numerical Considerations

### Fit-window dependence

The UV and IR regions are selected using agreement with known analytical values. This makes the method suitable for numerical verification, but it is not a fully independent extraction of unknown asymptotic coefficients.

### Derivative noise

Numerical differentiation can be sensitive to:

- non-uniform grid spacing;
- endpoint behaviour;
- integration error;
- insufficient scan resolution.

The IR smoothing reduces this noise, but the chosen smoothing window can affect the selected region.

### Integration warnings

The adaptive quadrature reports `IntegrationWarning` events for many turning points, especially near numerically sensitive endpoint regions. In the verified default run, the four primary scans reported:

| $\Sigma$ | Warned points | Warning events | Failed points |
|---:|---:|---:|---:|
| $0.5$ | 423 | 842 | 0 |
| $1.0$ | 438 | 873 | 0 |
| $2.0$ | 451 | 885 | 0 |
| $4.0$ | 484 | 909 | 0 |

The warning summaries were previously hidden by a module-wide filter. They are now reported without changing the integration results or discarding warned points.

No integration failed in the verified run, all 1,000 points were retained for every primary scan, and the cutoff-convergence results remained stable. Nevertheless, the warnings should be treated as a numerical limitation. Changing endpoint handling, quadrature tolerances, scan limits, or the integration method could change the fitted results and should therefore be revalidated against the reported baseline.

### Finite cutoff

The formal upper integration limit is replaced by $U_{\max}$. The convergence test checks the stability of the IR slope, but the full potential and additive constants can retain some cutoff dependence.

### Additive energy constant

The subtraction scheme fixes the vertical offset of the energy. Changing the subtraction convention can shift the potential without changing the UV coefficient or IR slope.

### Automatic window selection

The selector requires at least one sufficiently long contiguous block to satisfy the configured tolerance. If no suitable block exists, the program raises a descriptive `RuntimeError` rather than continuing with an undefined fit window.

### Diagnostic display windows

The UV and IR diagnostic plots show zoomed views around the automatically selected fit regions. These displays are intended to make the local asymptotic behaviour legible. The complete numerical arrays remain available in the returned results, and the full $E(\ell)$ curves are plotted separately.

The subplot layout adapts to the number of supplied $\Sigma$ values. The default four-value analysis uses a $2\times2$ layout.

## Expected Behaviour

A successful run should show:

- a smooth potential across the full scanned range;
- approximately Coulombic behaviour at small $\ell$;
- approximately linear behaviour at large $\ell$;
- UV coefficients close to the same exact value for all $\Sigma$;
- IR slopes increasing in proportion to $\Sigma$;
- high $R^2$ values in both selected regions;
- stable IR slopes under moderate changes in $U_{\max}$;
- non-overlapping UV and IR fit windows.

These checks provide numerical support for the analytical interpretation of the deformed geometry.

## Related Dissertation

This code accompanies:

> Jack Turner, *String Theory, Holography, and Confinement in Quantum Chromodynamics*, Swansea University, 2026.

The dissertation contains the analytical derivation of the Wilson-loop expressions, the deformed holographic background, the asymptotic UV and IR calculations, and the interpretation of the numerical results.

This repository presents the dissertation numerical analysis as an academic and portfolio project. It is not a general-purpose phenomenology package, and its fitted windows use known analytical expectations as verification targets.

## Licence

No open-source licence has currently been applied.

Copyright remains with Jack Turner. The absence of a licence means that permission to copy, modify, or redistribute the code should not be assumed.
