# Improvement Record: Holographic Wilson-Loop Analysis — Gate 3

## Archive represented

This document describes the changes made when producing:

```text
Holographic-Wilson-Loop-Analysis-Gate3.zip
```

The accompanying `Dissertation Code.py` is the untouched original source used as the baseline. Its SHA-256 checksum is:

```text
2f8c1a9755d29ad870ffa91c1f196b5a8ccd6b11c153c46787c1faa07bd50bf5  Dissertation Code.py
```

## Executive summary

The original dissertation code numerically reconstructed the static quark-antiquark potential in a deformed holographic geometry. The repository preparation deliberately preserved the model, integration formulas, scan settings, fit tolerances, selected default windows, and numerical conclusions.

The improvements focused on correctness outside the default configuration, transparent numerical diagnostics, reusable structure, safer parallel execution, adaptive plotting, dependency declaration, and a complete technical README.

The most important code correction was the ultraviolet eligibility mask. Parentheses were added so that all Boolean conditions are applied as intended. The default results were unchanged at full floating-point precision, but custom configurations are no longer vulnerable to Python operator-precedence behaviour that could bypass eligibility rules.

## Physical calculation retained

The deformed warp factor remains

$$
F(U)=U^2+\frac{\Sigma^2}{U^2}.
$$

Its minimum occurs at

$$
U_\star=\sqrt{\Sigma},
$$

and the analytical infrared string tension is

$$
T_{\mathrm{string}}
=
\frac{\Sigma}{\pi R^2}.
$$

The ultraviolet potential is expected to approach

$$
E(\ell)
\sim
-\frac{\alpha}{\ell}+c_{\mathrm{UV}},
$$

where

$$
\alpha_{\mathrm{exact}}
=
\frac{4\pi^2R^2}{\Gamma\!\left(\frac14\right)^4}.
$$

These formulas and their numerical verification were not changed.

## Improvement summary

| Area | Improvement | Why it was made | Impact |
|---|---|---|---|
| File structure | Renamed the repository copy to `wilson_loop_analysis.py` | Spaces and generic naming reduce importability | The analysis can be imported as a normal Python module |
| Entrypoint | Added a protected main entrypoint | `ProcessPoolExecutor` can recursively spawn workers without a main guard on spawn-based platforms | Direct execution is safer across operating systems |
| Validation | Added explicit configuration and per-$\Sigma$ checks | Invalid cutoffs or scan settings previously failed deep inside numerical routines | Users receive early, descriptive errors |
| Error handling | Added descriptive failures for empty data and missing fit regions | Undefined fits could otherwise fail indirectly | Invalid configurations stop at the actual cause |
| Integration diagnostics | Captured `IntegrationWarning` events inside workers | A module-wide warning filter hid numerical sensitivity | Warning counts and failures are now visible without discarding results |
| UV selection | Corrected Boolean-mask parentheses | Python operator precedence could ignore intended conditions | Default output is preserved and custom configurations apply every eligibility rule correctly |
| Plotting | Added adaptive subplot layouts and expanded colour handling | The original display assumed the default four $\Sigma$ values | Custom parameter lists plot without layout failure |
| Diagnostics | Zoomed UV and IR diagnostic displays around selected windows | Full-range axes made the fitted asymptotic regions difficult to inspect | The same numerical data are easier to evaluate visually |
| Style | Added docstrings, clearer naming, and restrained PEP 8 formatting | The research script was difficult to navigate | The numerical workflow is more maintainable without changing its mathematics |
| Packaging | Added dependencies, `.gitignore`, and a comprehensive README | The dissertation script was not a self-contained repository | Installation, equations, results, caveats, and usage are documented |

## Detailed technical improvements

### Endpoint regularisation retained and documented

The radial integrals contain a square-root singularity at $U=U_0$. The original substitution

$$
U=U_0+t^2,
\qquad
dU=2t\,dt
$$

was retained. It converts the separation integral to

$$
\ell(U_0)
=
2R^2F_0
\int_0^{\sqrt{U_{\max}-U_0}}
\frac{2t\,dt}
{F(U_0+t^2)\sqrt{F(U_0+t^2)^2-F_0^2}},
$$

where $F_0=F(U_0)$.

The rebuild did not alter the integration expression. It added validation and surfaced the adaptive-quadrature warnings that were previously suppressed.

### Transparent integration warnings

The original module disabled SciPy integration warnings globally. This made a clean terminal display but concealed how frequently the adaptive quadrature encountered difficult regions.

The improved workers capture warnings locally and return summary information. In the verified default run, many turning points generated warning events near sensitive endpoints, but no integration failed and all 1,000 primary scan points were retained for every $\Sigma$.

The impact is interpretive rather than numerical: users can distinguish a successful result from a numerically effortless one.

### Correct ultraviolet eligibility mask

The effective Coulomb coefficient is

$$
\alpha_{\mathrm{eff}}(\ell)
=
\ell^2\frac{dE}{d\ell}.
$$

The corrected eligibility expression is logically

$$
M
=
M_{\mathrm{skip}}
\land
M_{U_0}
\land
M_{\mathrm{finite}}.
$$

In code, every comparison is parenthesized before applying `&`. This removes ambiguity between comparison and bitwise operators.

For the default configuration, the selected UV windows and fitted coefficients were identical before and after the correction. The impact appears when settings change: the skip count, $U_0$ limit, and finite-value rule can no longer be bypassed by precedence.

### Explicit asymptotic selection failures

The UV region is accepted when

$$
\frac{|\alpha_{\mathrm{eff}}-\alpha_{\mathrm{exact}}|}
{\alpha_{\mathrm{exact}}}
\leq
\varepsilon_{\mathrm{UV}},
$$

and the IR region is accepted when

$$
\frac{|T_{\mathrm{eff}}-T_{\mathrm{exact}}|}
{T_{\mathrm{exact}}}
\leq
\varepsilon_{\mathrm{IR}},
$$

where

$$
T_{\mathrm{eff}}=\frac{dE}{d\ell}.
$$

If no sufficiently long contiguous block satisfies the configured tolerances, the rebuild raises a descriptive `RuntimeError`. This is safer than allowing later indexing or regression operations to fail with little context.

### Safer parallel execution

Each $U_0$ integral is independent, so the use of `ProcessPoolExecutor` was retained. The repository version places direct execution behind

```python
if __name__ == "__main__":
```

This prevents recursive process creation on platforms that import the main module when starting a worker.

### Adaptive plots

The original plotting layout assumed four deformation values. The improved layout calculates a suitable subplot grid and expands colours as necessary. This does not change the plotted quantities, selected windows, or fits. It removes an unnecessary restriction on exploratory parameter lists.

## Verification and measured impact

The cleaned repository was compared with an untouched baseline copy. For the default configuration, the following remained unchanged:

- retained point counts;
- UV and IR selector indices;
- fitted Coulomb coefficients;
- fitted string tensions;
- $R^2$ values;
- cutoff-convergence results;
- printed numerical conclusions.

This matters because the repository work was intended as maintenance and transparency work, not a redefinition of the dissertation analysis.

The corrected mask improves correctness for non-default settings. Warning summaries improve visibility of integration sensitivity. Adaptive plotting and validation improve usability without affecting the verified baseline.

## Repository and documentation impact

The archive contains:

- the importable analysis module;
- compatible NumPy, SciPy, and Matplotlib requirements;
- a Python-focused `.gitignore`;
- a detailed README with the physical derivation, numerical method, configuration, results, plots, warnings, limitations, and dissertation context.

The README explains that the automatic fit windows use known analytical values as verification targets. This is important: the fits confirm expected asymptotic behaviour but are not fully independent discoveries of unknown coefficients.

## Remaining limitations

- The radial integral uses a finite cutoff $U_{\max}$.
- Numerical differentiation remains sensitive to grid resolution and endpoint behaviour.
- IR smoothing can influence the selected region.
- Many default integrations raise captured warnings even though none fails.
- UV and IR window selection uses the known analytical targets.
- The additive energy constant depends on the subtraction convention.
- No open-source licence was applied to the produced archive.
