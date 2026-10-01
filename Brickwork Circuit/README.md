# XXZ Brickwork (L = 12): AGP Norm and CD Infidelity

This code reproduces two figures for the L = 12 XXZ brickwork circuit. Both compute the approximate AGP the same way: a Szegő recursion, followed by regularized AGP coefficients in the CMV basis.

| Figure | Quantity |
|---|---|
| `Figures/AGP_norm_Brickwork.pdf` | Regularized AGP norm ‖A_θ^(R)‖² vs. θ at η_reg = 0.1, lines for log-spaced truncation orders M = 2, 4, 6, 10, 18, 30, 50 (legend), with the exact (eigendecomposition) curve in solid red |
| `Figures/AGP_Brickwork_Infidelity.pdf` | Counterdiabatic-driving infidelity 1 − F vs. circuit depth S = 2, 4, …, 64 (log–log), lines for M = 10, 30, 50 (the CSV also has 3, 5, 8, 15, 20, 25, 40), with the exact regularized AGP in solid red and the uncorrected ramp in gray dashed |

## Files

```
Brickwork Circuit/
├── core.jl                   # all physics (module BrickworkCore)
├── run_agp_norm_sweep.jl     # data for the AGP-norm figure   -> agp_norm_L12.csv
├── run_infidelity_ramp.jl    # data for the infidelity figure -> infidelity_L12.csv
├── generate_figures.jl       # both figures                   -> Figures/*.pdf
├── agp_norm_L12.csv
├── infidelity_L12.csv
└── Figures/
```

`core.jl` contains:
1. The XXZ gate Ř(θ,γ), its θ-derivative, and the two-site seed generator.
2. The brickwork circuit U and its seed G_θ = −i(∂_θU)U†. These are built by applying two-site gates as tensor contractions, with no dense d×d matrix products.
3. The symmetry sectors: S^z = 0 ∩ reflection ∩ spin flip. The ground-state block has dimension 252.
4. A resumable Szegő recursion for the Verblunsky coefficients α_k, ρ_k.
5. The regularized AGP in the CMV basis. The coefficients come from a matrix-free CG solve. The norm follows from the coefficients alone; the operator is rebuilt in one forward sweep.
6. The exact regularized AGP from the eigendecomposition, used as the reference curve.


## Requirements

- Julia 1.12 (tested with 1.12.2).
- `core.jl` and the two data scripts use only the standard libraries (`LinearAlgebra`, `Printf`, `DelimitedFiles`).
- `generate_figures.jl` needs `Plots` and `LaTeXStrings`:
  ```julia
  using Pkg; Pkg.add(["Plots", "LaTeXStrings"])
  ```

## Quick start

Run these commands from inside `Brickwork Circuit/`.

**Make the figures from the included data (a few seconds):**
```bash
julia generate_figures.jl
```

**Regenerate all the data from scratch, then make the figures:**
```bash
julia run_agp_norm_sweep.jl      # ~5 min (60 θ points)
julia run_infidelity_ramp.jl     # ~5.5 min (65 θ points)
julia generate_figures.jl
```

## Parameters

Both figures use L = 12, gate anisotropy angle γ = 1.0 (Δ = cos γ ≈ 0.54), and the S^z = 0, R = +1, F = +1 sector (dimension 252). The sector is chosen as the one holding the lowest eigenphase of U at θ = 0.05.

**AGP norm:**
- η_reg = 0.1 (fixed).
- θ ∈ [0.05, 2.5], 60 points.
- M = 2:2:50 in the CSV, from one Szegő recursion to depth 50 per θ. The figure plots a log-spaced subset.
- At small θ the Krylov space is exhausted before depth 50, so larger M reuse the exhausted depth. The `K_reached` column records where this happens.

**Infidelity:**
- η_reg = 0.05 · 2π / 252, 1/20 of the mean level spacing. At half the spacing, regularization left a ~1e-6 floor in 1 − F for the exact AGP.
- The ramp runs from θ = 1.5 to 0.2 in S = 2, 4, 8, 16, 32, 64 steps.
- The step θ_i → θ_{i+1} applies the trapezoidal kick exp(−i δθ (A(θ_i) + A(θ_{i+1}))/2) and then U(θ_{i+1}). The kick maps eigenstates of U(θ_i) to those of U(θ_{i+1}), so the following U must be at θ_{i+1}.
- The start and end states are ground states tracked by lowest ⟨G_θ⟩.
- Every S grid is a subset of the S = 64 grid, so the per-θ work is done once on those 65 points and reused for every S.
