"""
run_agp_norm_sweep.jl

Data for Figures/AGP_norm_Brickwork.pdf: regularized AGP norm ‖A_θ^(R)‖²
vs. θ for the L = 12 XXZ brickwork, at Szegő/CMV truncation orders
M = 2, 4, …, 50, plus the exact (eigendecomposition) value.

Output: agp_norm_L12.csv
    columns: theta, norm_exact_reg, M=2, M=4, …, M=50, K_reached
    (K_reached < 50: the Krylov space was exhausted before M = 50, and
     larger M reuse the exhausted depth)

Usage (from this directory):  julia run_agp_norm_sweep.jl
"""

using LinearAlgebra, Printf, DelimitedFiles
include("core.jl")
using .BrickworkCore

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))

# ── Parameters ──
L         = 12
γ         = 1.0                     # gate anisotropy angle, Δ = cos γ
η_reg     = 0.1                     # fixed regularization
theta_min = 0.05
theta_max = 2.50
n_theta   = 60
M_values  = collect(2:2:50)
M_max     = maximum(M_values)

P = half_filling_projector(L)
P_comb, sector_name, dim_sub = ground_sector_projector(L, γ, P; θ_probe=theta_min)
@printf("L=%d  γ=%.4f  η_reg=%.3g  sector %s (dim %d)  BLAS threads %d\n",
        L, γ, η_reg, sector_name, dim_sub, BLAS.get_num_threads())

thetas     = collect(range(theta_min, theta_max, length=n_theta))
norm_exact = zeros(n_theta)
norm_cmv   = fill(NaN, n_theta, length(M_values))
K_reached  = zeros(Int, n_theta)

t0 = time()
for (i, θ) in enumerate(thetas)
    U, G, dU = sector_seed(θ, γ, L, P_comb)

    norm_exact[i] = agp_norm(U, exact_agp_regulated_A(U, dU, η_reg))

    state = szego_init(G)
    szego_extend!(U, state, M_max)
    K_reached[i] = length(state.alphas)
    for (j, M) in enumerate(M_values)
        norm_cmv[i, j], _, _ = agp_norm2_cmv_cg(state.alphas, state.rhos, state.init_norm,
                                                η_reg, min(M, K_reached[i]))
    end

    @printf("θ %2d/%d  θ=%.4f  exact=%.4g  K_reached=%d  (%.1f s)\n",
            i, n_theta, θ, norm_exact[i], K_reached[i], time() - t0)
    flush(stdout)
end

open("agp_norm_L12.csv", "w") do f
    println(f, join(vcat(["theta", "norm_exact_reg"], ["M=$M" for M in M_values], ["K_reached"]), ","))
    writedlm(f, hcat(thetas, norm_exact, norm_cmv, K_reached), ',')
end
println("Saved agp_norm_L12.csv")
