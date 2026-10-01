# Kicked Top: AGP Norm as a Chaos Diagnostic

This code reproduces the kicked-top figures. The first three compare chaos indicators as functions of the kick strength *k*:

| Figure | Quantity |
|---|---|
| `Figures/kicked_top_agp_exponent.pdf` | Finite-size scaling exponent α of the squared regularized AGP norm, ‖A_k‖² ~ j^α, with an inset of ‖A_k‖² vs. j |
| `Figures/kicked_top_lyapunov.pdf` | Classical Lyapunov exponent λ |
| `Figures/kicked_top_level_spacing_ratio.pdf` | Mean level-spacing ratio ⟨r̃⟩, with the COE and Poisson reference values |

The other three show how the Krylov calculation of the AGP converges, at j = 200:

| Figure | Quantity |
|---|---|
| `Figures/kicked_top_agp_convergence_vs_M.pdf` | Squared regularized AGP norm ‖A_k^(R)‖² vs. Krylov truncation order M for k = 0.5, 2.5 and 6.0. Dotted lines mark the exact-diagonalization values. |
| `Figures/kicked_top_agp_Mconv_vs_k.pdf` | Krylov depth M needed to converge vs. k, compared with the maximal Krylov dimension K = d² − d + 1 |
| `Figures/kicked_top_combined_inset.pdf` | The two figures above combined: M-to-converge as the main panel, with the convergence curves as an inset |

## Files

```
Kicked Top /
├── core.jl                                    # all calculations + sweeps that write the CSVs
├── generate_kicked_top_figures_split.jl       # AGP exponent, Lyapunov, level-spacing figures
├── generate_kicked_top_convergence_split.jl   # the two convergence figures (+ combined)
├── kicked_top_near_zero_k_summary.csv         # AGP norm per (k, j)        -> AGP exponent
├── kicked_top_near_zero_k_detail.csv          # AGP norm per Krylov step   -> convergence figures
├── kicked_top_exact_agp_reference.csv         # exact AGP norms, j = 200   -> convergence figures
├── kicked_top_lyapunov_sweep.csv              # λ(k)                       -> Lyapunov
├── level_stats_sweep.csv                      # ⟨r⟩(k)                     -> level spacing
└── Figures/                                   # output PDFs
```

## Requirements

- Julia 1.12. Earlier 1.x versions should also work.
- `core.jl` uses only the standard libraries (`LinearAlgebra`, `Statistics`, `Printf`).
- The figure scripts need `Plots` and `LaTeXStrings`:
  ```julia
  using Pkg; Pkg.add(["Plots", "LaTeXStrings"])
  ```

## Quick start

Run these commands from inside `Kicked Top /` (the folder name ends in a space).

**Make the figures from the included data (a few seconds):**
```bash
mkdir -p Figures
julia generate_kicked_top_figures_split.jl
julia generate_kicked_top_convergence_split.jl
```

**Regenerate all the data from scratch, then make the figures:**
```bash
rm kicked_top_near_zero_k_summary.csv kicked_top_near_zero_k_detail.csv   # see note below
julia core.jl            # all sweeps
julia core.jl level      # or run just one: level | lyapunov | agp | exact
mkdir -p Figures
julia generate_kicked_top_figures_split.jl
julia generate_kicked_top_convergence_split.jl
```

The `rm` is needed because `run_agp_sweep` appends to the two AGP CSVs and skips every (k, j) point already in the summary file, so an interrupted run can be resumed. With the included CSVs in place, `julia core.jl agp` does nothing. The other sweeps overwrite their CSVs.

## The model

Quantum kicked top with spin j and Hilbert-space dimension d = 2j+1:

    U(p, k) = exp(-i (k/2j) J_z²) · exp(-i p J_y)

Here k is the twist (chaos) parameter and p is the precession angle. All quantum calculations are done in the +1 sector of the parity Π_y = exp(iπJ_y), because U commutes with Π_y.

## How each quantity is computed

### AGP norm and the exponent α (`run_agp_sweep`)

`agp_norm_cell(k, j)` computes one (k, j) point:

1. **Build the operators.** U is built at p = 0.9, together with the seed G_k = −i(∂_k U)U† = −J_z²/(2j). Both are projected into the parity sector.
2. **Krylov basis (Szegő recursion, `szego_extend!`).** Starting from G, the map X ↦ U X U† is applied repeatedly with the inner product ⟨A,B⟩ = tr(A†B)/d. Each step gives a pair of Verblunsky coefficients (αₙ, ρₙ). Memory use is O(d²), independent of the Krylov depth.
3. **Regularized AGP (`agp_norm2_cmv_cg`).** In the Krylov basis, conjugation by U is represented by the CMV matrix, which is never formed explicitly. At a truncation depth M below the full Krylov dimension, the last coefficient is set to α_{M−1} = 1, which keeps the truncated matrix exactly unitary. The code solves (Δ†Δ + η²)c = Δ†e₁ with Δ = U_CMV − I by matrix-free conjugate gradient. It uses η = ½·(2π/d), which is half the mean level spacing, and returns ‖A_k‖² = ‖G‖²·Σ|cₙ|².
4. **Adaptive depth (`adaptive_converge_cmv_stepwise`).** The Krylov depth M is increased one vector at a time. At each M the norm is computed twice, closing the truncation with α_{M−1} = +1 and with α_{M−1} = −1. The loop stops once the two agree to a relative difference (the closure gap) below 10⁻⁴, meaning the result no longer depends on the unexplored part of the Krylov space. The α = +1 value is reported. At j = 200, the converged norms agree with exact diagonalization to a relative error below 10⁻⁴ for every k (largest 5.4 × 10⁻⁵, at k = 1.75).

The figure script fits log‖A_k‖² against log j over j ≥ 50 to get α(k).

### Exact reference and convergence figures (`run_exact_reference`)

`exact_agp_norm2(U, G, η)` computes the same regularized AGP norm by fully diagonalizing U:

    [A^(R)]_mn = (e^{i(θm−θn)} − 1)* G_mn / (|e^{i(θm−θn)} − 1|² + η²),    ‖A^(R)‖² = (1/d) Σ_{m≠n} |[A^(R)]_mn|²

It uses the same seed, parity sector and η as the Krylov calculation, so the two should agree once the Krylov calculation has converged. `run_exact_reference` evaluates it at j = 200 for every k on the grid.

Both convergence figures are built from `kicked_top_near_zero_k_detail.csv` at j = 200:
- **Convergence vs. M** plots norm2 = ‖A_k‖² against M on log-log axes, with the exact values drawn as dotted lines.
- **M to converge vs. k** plots the first even M at which the Krylov norm is within a relative error of 10⁻⁴ of the exact value. The dashed line is K = d² − d + 1, where d = 201 is the parity-sector dimension at j = 200.

### Lyapunov exponent (`run_lyapunov_sweep`)

This uses the classical (j → ∞) map on the unit sphere: a rotation by p about y, then a twist by angle k·Z about z (`classical_step`). λ is computed with the Benettin two-trajectory method (`lyapunov_exponent`).

| Parameter | Value |
|---|---|
| Precession angle | p = 0.9 |
| Initial conditions | 50, uniform on the sphere |
| Transient steps | 200 |
| Measured steps | 1000 |
| Initial separation | d0 = 1e-8 |
| Random number generator | Seeded LCG (seed = 1), so runs are reproducible |

### Level statistics (`run_level_stats_sweep`)

The code computes the ratio r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) of consecutive quasi-energy spacings. Settings:

- j = 150, in a single parity sector.
- Averaged over 9 precession angles p ∈ [π/2 − 0.15, π/2 + 0.15].
- The error bars are standard errors of the mean.
- Reference values: COE ≈ 0.5307 (chaotic) and Poisson ≈ 0.3863 (integrable).

## Using the functions directly

```julia
include("core.jl")
using .KickedTopCore

# one AGP data point
d, η, res = agp_norm_cell(3.0, 50.0)          # k = 3.0, j = 50
res.norm2, res.M, res.status                  # ‖A‖², Krylov depth, "converged"

# one Lyapunov exponent
lyapunov_exponent(0.9, 3.0; n_ic=50, n_steps=1000, n_transient=200)

# exact AGP norm² for one (k, j), by full diagonalization
d, η, n2 = exact_agp_cell(6.0, 200.0)         # sqrt(n2) ≈ 207.84

# ⟨r⟩ for a single unitary
P_plus, _ = parity_sectors(150.0)
U, _, _ = kicked_top_floquet(π/2, 150.0; k=5.0)
mean_r_statistic(project_to_sector(U, P_plus))

# custom sweeps (all keyword arguments are optional)
run_agp_sweep("my_summary.csv", "my_detail.csv"; ks=[1.0, 5.0], js=[20.0, 50.0])
run_lyapunov_sweep("my_lyap.csv"; ks=0.1:0.1:8.0)
run_level_stats_sweep("my_levels.csv"; j=100.0)
run_exact_reference("my_exact.csv"; ks=[1.0, 3.0], j=100.0)
```

### Default parameters (defined at the top of `core.jl`)

| Name | Value | Used by |
|---|---|---|
| `K_VALUES` | 90 values of k, from 0.05 to 8.0 | AGP and Lyapunov sweeps |
| `J_VALUES` | 10, 20, 30, 40, 50, 70, 90, 110, 140, 170, 200 | AGP sweep |
| `P0` | 0.9 | AGP and Lyapunov sweeps |

## CSV formats

| File | Columns |
|---|---|
| `kicked_top_near_zero_k_summary.csv` | `k, j, d, eta, K_full, K_reached, M_conv, status, closure_gap_final, norm2_cmv` |
| `kicked_top_near_zero_k_detail.csv` | `k, j, d, M, norm2, closure_gap, cg_iters` (one row per Krylov step M) |
| `kicked_top_lyapunov_sweep.csv` | `k, lyapunov` |
| `level_stats_sweep.csv` | `k, r_mean, r_std, n_samples` |
| `kicked_top_exact_agp_reference.csv` | `k, j, d, eta, exact_norm2, exact_norm` |

In the AGP files, `d` is the dimension of the parity sector, roughly j + 1, not 2j + 1. `norm2_cmv`, `norm2` and `exact_norm2` are ‖A_k‖².
