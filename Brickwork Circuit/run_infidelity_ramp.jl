"""
run_infidelity_ramp.jl

Data for Figures/AGP_Brickwork_Infidelity.pdf: counterdiabatic-driving
infidelity 1 − F vs. circuit depth S for the L = 12 XXZ brickwork.

The ramp takes θ from 1.5 to 0.2 in S steps (S = 2, 4, …, 64). The step
θ_i → θ_{i+1} applies the counterdiabatic kick exp(−i δθ (A(θ_i) + A(θ_{i+1}))/2)
(trapezoidal, second-order Magnus), then U(θ_{i+1}): the kick carries
eigenstates of U(θ_i) to those of U(θ_{i+1}), so the U that follows must be
the one at the new θ.
The state starts in the tracked ground state at θ = 1.5 and F is its
overlap with the tracked ground state at θ = 0.2. A is the Szegő/CMV AGP
at truncation order M, or the exact regularized AGP; the uncorrected
ramp (no A) is the baseline.

Every S grid is a subset of the S = 64 grid, so the per-θ work is done
once on those 65 points and reused for every S.

Output: infidelity_L12.csv
    columns: S, leak_uncorrected, leak_exact, M=3, M=5, M=8, M=10, M=15, M=20, M=25, M=30, M=40, M=50

Usage (from this directory):  julia run_infidelity_ramp.jl
"""

using LinearAlgebra, Printf
include("core.jl")
using .BrickworkCore


BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))

# ── Parameters ──
L          = 12
γ          = 1.0                    # gate anisotropy angle, Δ = cos γ
θ_max      = 1.5
θ_min      = 0.2
S_values   = [2, 4, 8, 16, 32, 64]
M_list     = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
M_max      = maximum(M_list)

P = half_filling_projector(L)
P_comb, sector_name, dim_sub = ground_sector_projector(L, γ, P; θ_probe=0.05)
η_reg = 0.05 * 2π / dim_sub         # 1/20 of the mean level spacing (1/2 left a ~1e-6 floor)
@printf("L=%d  γ=%.4f  η_reg=%.6f  sector %s (dim %d)  BLAS threads %d\n",
        L, γ, η_reg, sector_name, dim_sub, BLAS.get_num_threads())

# ── Per-θ work on the finest grid ──
S_max = maximum(S_values)
@assert all(S_max % S == 0 for S in S_values)
θ_all = collect(range(θ_max, θ_min, length=S_max + 1))
n_all = length(θ_all)

Us    = Vector{Matrix{ComplexF64}}(undef, n_all)
track = Vector{Vector{ComplexF64}}(undef, n_all)     # tracked ground state
As    = Vector{Dict{Any,Matrix{ComplexF64}}}(undef, n_all)   # M or "exact" => A

t0 = time()
for (i, θ) in enumerate(θ_all)
    U, G, dU = sector_seed(θ, γ, L, P_comb)
    G = Hermitian(G)
    Us[i] = U

    # Ground state: eigenvector of U with the lowest ⟨G⟩ (Kato ordering)
    _, vecs = eigen(U)
    track[i] = vecs[:, argmin(real.(diag(vecs' * G * vecs)))]

    state = szego_init(G)
    szego_extend!(U, state, M_max)
    recon = reconstruct_multi_M_cmv(U, G, state, η_reg, M_list)
    As[i] = Dict{Any,Matrix{ComplexF64}}(M => recon[M][1] for M in M_list)
    As[i]["exact"] = exact_agp_regulated_A(U, dU, η_reg)

    @printf("θ %2d/%d  θ=%.5f  (%.1f s)\n", i, n_all, θ, time() - t0)
    flush(stdout)
end

# ── Ramps ──
"""
    ramp_infidelity(idx, label) -> 1 − F

Ramp over θ_all[idx]; `label` is an M, "exact", or nothing (uncorrected).
"""
function ramp_infidelity(idx, label)
    ψ = copy(track[idx[1]])
    for (i, i_next) in zip(idx[1:end-1], idx[2:end])
        if label !== nothing
            A_mid = (As[i][label] + As[i_next][label]) / 2
            ψ = exp(-im * (θ_all[i_next] - θ_all[i]) * A_mid) * ψ
        end
        ψ = Us[i_next] * ψ
        ψ /= norm(ψ)
    end
    return 1 - abs2(dot(track[idx[end]], ψ))
end

labels = vcat([nothing, "exact"], M_list)
open("infidelity_L12.csv", "w") do f
    println(f, join(vcat(["S", "leak_uncorrected", "leak_exact"], ["M=$M" for M in M_list]), ","))
    for S in S_values
        idx  = 1:(S_max ÷ S):n_all
        leak = [ramp_infidelity(idx, lab) for lab in labels]
        println(f, join(vcat(S, [@sprintf("%.17g", x) for x in leak]), ","))
        @printf("S=%3d  uncorrected=%.3e  exact=%.3e  M=50=%.3e\n", S, leak[1], leak[2], leak[end])
    end
end
println("Saved infidelity_L12.csv")
