"""
core.jl

Self-contained core for the kicked-top chaos-diagnostic figures:

  kicked_top_agp_exponent         <- run_agp_sweep         -> kicked_top_near_zero_k_summary.csv
  kicked_top_lyapunov             <- run_lyapunov_sweep    -> kicked_top_lyapunov_sweep.csv
  kicked_top_level_spacing_ratio  <- run_level_stats_sweep -> level_stats_sweep.csv
  kicked_top_agp_convergence_vs_M,
  kicked_top_agp_Mconv_vs_k,
  kicked_top_combined_inset       <- run_agp_sweep         -> kicked_top_near_zero_k_detail.csv
                                   + run_exact_reference   -> kicked_top_exact_agp_reference.csv

The figures themselves are made by generate_kicked_top_figures_split.jl
and generate_kicked_top_convergence_split.jl, which read the CSVs above
from the working directory.

Usage:
  julia core.jl                          # run all sweeps with production parameters
  julia core.jl agp|lyapunov|level|exact # run only one sweep

or from Julia:
  include("core.jl"); using .KickedTopCore
  run_level_stats_sweep("level_stats_sweep.csv")

Model (Haake, Kus & Scharf, Z. Phys. B 65, 381 (1987)):
    U(p, k) = exp(-i (k/2j) J_z²) · exp(-i p J_y),   d = 2j+1
k is the twist (chaos) parameter, p the precession angle.
"""
module KickedTopCore

using LinearAlgebra, Statistics, Printf

export spin_operators, kicked_top_floquet, kicked_top_floquet_dk
export parity_sectors, project_to_sector, kicked_top_seed_sector, kicked_top_seed_sector_dk
export level_spacing_ratio, mean_r_statistic
export classical_step, lyapunov_exponent
export SzegoState, szego_init, szego_extend!
export cmv_matvec, cmv_matvec_adjoint, cg_normal_equations, agp_norm2_cmv_cg
export adaptive_converge_cmv_stepwise, cutoff_eta, agp_norm_cell
export exact_agp_norm2, exact_agp_cell
export K_VALUES, J_VALUES, P0
export run_agp_sweep, run_lyapunov_sweep, run_level_stats_sweep, run_exact_reference

# ═══════════════════════════════════════════════════════════════════════
#  Production parameters (as used for the published data)
# ═══════════════════════════════════════════════════════════════════════

# k grid shared by the AGP and Lyapunov sweeps
const K_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
                  0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.75, 1.8, 1.9, 2.0, 2.1, 2.2, 2.25, 2.3,
                  2.4, 2.5, 2.6, 2.7, 2.75, 2.8, 2.9, 3.0, 3.1, 3.2, 3.25,
                  3.3, 3.4, 3.5, 3.6, 3.7, 3.75, 3.8, 3.9, 4.0, 4.1, 4.2,
                  4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3,
                  5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4,
                  6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5,
                  7.6, 7.7, 7.8, 7.9, 8.0]
const J_VALUES = [10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 90.0, 110.0, 140.0, 170.0, 200.0]
const P0 = 0.9   # precession angle for the AGP and Lyapunov sweeps

# ═══════════════════════════════════════════════════════════════════════
#  1. Quantum kicked top
# ═══════════════════════════════════════════════════════════════════════

"""
    spin_operators(j) → (Jx, Jy, Jz)

d×d (d = 2j+1) spin operators in the |j,m⟩ basis ordered m = j, j-1, …, -j.
"""
function spin_operators(j::Real)
    d = Int(round(2j + 1))
    @assert abs(2j + 1 - d) < 1e-9 "j must be a non-negative half-integer"
    m = collect(j:-1:-j)
    Jz = Matrix(Diagonal(ComplexF64.(m)))
    Jp = zeros(ComplexF64, d, d)
    for n in 1:d-1
        mm = m[n+1]
        Jp[n, n+1] = sqrt(j*(j+1) - mm*(mm+1))
    end
    Jminus = adjoint(Jp)
    Jx = Matrix((Jp + Jminus) / 2)
    Jy = Matrix((Jp - Jminus) / (2im))
    return Jx, Jy, Jz
end

"""
    kicked_top_floquet(p, j; k=3.0) → (U, G, dU)

Floquet unitary U(p) and AGP seed w.r.t. the precession angle p:
    G_p = -U_twist · J_y · U_twist†
Used for the level-statistics sweep (only U is needed there).
"""
function kicked_top_floquet(p::Real, j::Real; k::Real=3.0)
    Jx, Jy, Jz = spin_operators(j)
    U_twist = exp(-im * (k / (2j)) * Jz^2)
    U_kick  = exp(-im * p * Jy)
    U  = U_twist * U_kick
    dU = U_twist * (-im * Jy) * U_kick
    G  = -Matrix(U_twist * Jy * U_twist')
    G  = (G + G') / 2   # enforce exact Hermiticity
    return U, G, dU
end

"""
    kicked_top_floquet_dk(p, j; k=3.0) → (U, G, dU)

Floquet unitary and AGP seed w.r.t. the twist/chaos parameter k:
    ∂_k U = (-i/2j) J_z² U,    G_k = -i(∂_k U)U† = -(1/2j) J_z²
G_k is constant, but the AGP built from it via the Szegő recursion still
depends on k through U. This is the seed for all the AGP figures (exponent
and convergence).
"""
function kicked_top_floquet_dk(p::Real, j::Real; k::Real=3.0)
    Jx, Jy, Jz = spin_operators(j)
    U_twist = exp(-im * (k / (2j)) * Jz^2)
    U_kick  = exp(-im * p * Jy)
    U  = U_twist * U_kick
    dU = (-im/(2j)) * Jz^2 * U
    G  = -(1/(2j)) * Matrix(Jz^2)
    G  = (G + G') / 2   # enforce exact Hermiticity
    return U, G, dU
end

"""
    parity_sectors(j) → (P_plus, P_minus)

Orthonormal bases for the two sectors of Π_y = exp(iπJ_y), which commutes
with U(p,k). Built from the eigenvectors of the Hermitian J_y (not from
eigen(Π_y), whose degenerate eigenvectors are not orthonormal).
"""
function parity_sectors(j::Real)
    _, Jy, _ = spin_operators(j)
    Jy_herm = Hermitian((Jy + Jy') / 2)
    F = eigen(Jy_herm)
    phases = exp.(im * π .* F.values)
    vecs   = F.vectors
    P_plus  = vecs[:, findall(v -> real(v) > 0.5 || imag(v) > 0.5,  phases)]
    P_minus = vecs[:, findall(v -> real(v) < -0.5 || imag(v) < -0.5, phases)]
    return P_plus, P_minus
end

"""
    project_to_sector(M, P) = P' * M * P
"""
project_to_sector(M, P) = P' * M * P

function kicked_top_seed_sector(p::Real, j::Real, P::AbstractMatrix; k::Real=3.0)
    U, G, dU = kicked_top_floquet(p, j; k=k)
    return project_to_sector(U, P), project_to_sector(G, P), project_to_sector(dU, P)
end

function kicked_top_seed_sector_dk(p::Real, j::Real, P::AbstractMatrix; k::Real=3.0)
    U, G, dU = kicked_top_floquet_dk(p, j; k=k)
    return project_to_sector(U, P), project_to_sector(G, P), project_to_sector(dU, P)
end

# ═══════════════════════════════════════════════════════════════════════
#  2. Level statistics
# ═══════════════════════════════════════════════════════════════════════

"""
    level_spacing_ratio(U) → r_values

r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) for sorted quasi-energies of U.
⟨r⟩ ≈ 0.5307 (COE), ≈ 0.3863 (Poisson). Pass U projected into a SINGLE
parity sector; mixing sectors corrupts the statistic.
"""
function level_spacing_ratio(U::AbstractMatrix)
    evals = eigvals(U)
    θ = sort(mod2pi.(angle.(evals) .+ 2π))
    s = diff(θ)
    n = length(s)
    n < 2 && return Float64[]
    r = [min(s[i], s[i+1]) / max(s[i], s[i+1]) for i in 1:n-1]
    return r
end

mean_r_statistic(U::AbstractMatrix) = sum(level_spacing_ratio(U)) / length(level_spacing_ratio(U))

# ═══════════════════════════════════════════════════════════════════════
#  3. Classical kicked top and Lyapunov exponent
# ═══════════════════════════════════════════════════════════════════════

"""
    classical_step(X, Y, Z, p, k) -> (X', Y', Z')

One period of the classical (j → ∞) map on the unit sphere: rotation by p
about y, then twist by angle k·Z about z (same convention as
kicked_top_floquet).
"""
function classical_step(X::Float64, Y::Float64, Z::Float64, p::Float64, k::Float64)
    X1 =  X*cos(p) + Z*sin(p)
    Y1 =  Y
    Z1 = -X*sin(p) + Z*cos(p)
    kZ1 = k*Z1
    ck, sk = cos(kZ1), sin(kZ1)
    Xp = X1*ck - Y1*sk
    Yp = X1*sk + Y1*ck
    Zp = Z1
    return Xp, Yp, Zp
end

"""
    lyapunov_exponent(p, k; n_ic=20, n_steps=400, n_transient=50, d0=1e-8, seed=1) -> λ

Largest Lyapunov exponent by the Benettin two-trajectory method, averaged
over `n_ic` random initial conditions (deterministic LCG, seeded).
"""
function lyapunov_exponent(p::Float64, k::Float64; n_ic::Int=20, n_steps::Int=400,
                             n_transient::Int=50, d0::Float64=1e-8, seed::Int=1)
    rng_state = UInt64(seed)
    function next_rand()
        rng_state = rng_state * 6364136223846793005 + 1442695040888963407
        return Float64(rng_state >> 11) / Float64(UInt64(1) << 53)
    end

    lyap_sum = 0.0
    for ic in 1:n_ic
        theta = acos(2*next_rand() - 1)
        phi   = 2π*next_rand()
        X, Y, Z = sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)

        # random tangent perturbation of length d0
        vx, vy, vz = next_rand()-0.5, next_rand()-0.5, next_rand()-0.5
        dot = vx*X + vy*Y + vz*Z
        vx, vy, vz = vx - dot*X, vy - dot*Y, vz - dot*Z
        vnorm = sqrt(vx^2+vy^2+vz^2)
        vx, vy, vz = (vx/vnorm)*d0, (vy/vnorm)*d0, (vz/vnorm)*d0
        X2, Y2, Z2 = X+vx, Y+vy, Z+vz

        for step in 1:n_transient
            X, Y, Z   = classical_step(X, Y, Z, p, k)
            nrm = sqrt(X^2+Y^2+Z^2); X/=nrm; Y/=nrm; Z/=nrm
            X2,Y2,Z2  = classical_step(X2, Y2, Z2, p, k)
            nrm2 = sqrt(X2^2+Y2^2+Z2^2); X2/=nrm2; Y2/=nrm2; Z2/=nrm2
            dx,dy,dz = X2-X, Y2-Y, Z2-Z
            dnorm = sqrt(dx^2+dy^2+dz^2)
            dnorm = dnorm < 1e-300 ? d0 : dnorm
            sx,sy,sz = dx/dnorm*d0, dy/dnorm*d0, dz/dnorm*d0
            X2,Y2,Z2 = X+sx, Y+sy, Z+sz
        end

        acc = 0.0
        for step in 1:n_steps
            X, Y, Z   = classical_step(X, Y, Z, p, k)
            nrm = sqrt(X^2+Y^2+Z^2); X/=nrm; Y/=nrm; Z/=nrm
            X2,Y2,Z2  = classical_step(X2, Y2, Z2, p, k)
            nrm2 = sqrt(X2^2+Y2^2+Z2^2); X2/=nrm2; Y2/=nrm2; Z2/=nrm2
            dx,dy,dz = X2-X, Y2-Y, Z2-Z
            dnorm = sqrt(dx^2+dy^2+dz^2)
            dnorm = dnorm < 1e-300 ? d0 : dnorm
            acc += log(dnorm/d0)
            sx,sy,sz = dx/dnorm*d0, dy/dnorm*d0, dz/dnorm*d0
            X2,Y2,Z2 = X+sx, Y+sy, Z+sz
        end
        lyap_sum += acc / n_steps
    end
    return lyap_sum / n_ic
end

# ═══════════════════════════════════════════════════════════════════════
#  4. AGP norm: Szegő recursion + CMV/CG solve + adaptive convergence
# ═══════════════════════════════════════════════════════════════════════

"""
Resumable Szegő recursion state: only the two running operators (Φ, Φ̃)
and the Verblunsky coefficients are stored (O(d²) + O(K) memory).
"""
mutable struct SzegoState
    Phi::Matrix{ComplexF64}
    Phi_tilde::Matrix{ComplexF64}
    alphas::Vector{Float64}
    rhos::Vector{Float64}
    init_norm::Float64
    exhausted::Bool
end

"""
    szego_init(G_lambda)

Initialise the recursion from the Hermitian seed operator.
"""
function szego_init(G_lambda::AbstractMatrix; tol::Float64=1e-12)
    d = size(G_lambda, 1)
    init_norm = sqrt(abs(real(tr(G_lambda' * G_lambda))) / d)
    @assert init_norm > tol "Seed operator has zero norm — check G_lambda."
    Phi = G_lambda ./ init_norm
    return SzegoState(Phi, copy(Phi), Float64[], Float64[], init_norm, false)
end

"""
    szego_extend!(U, state, K_target; tol=1e-12)

Run the recursion forward (operator map X ↦ U X U†) until
`length(state.alphas) >= K_target` or the Krylov space is exhausted.
"""
function szego_extend!(U::AbstractMatrix, state::SzegoState, K_target::Int;
                        tol::Float64=1e-12)
    d = size(U, 1)
    ip(A, B) = tr(A' * B) / d
    U_op(X)  = U * X * U'

    while length(state.alphas) < K_target && !state.exhausted
        U_Phi     = U_op(state.Phi)
        alpha_bar = real(ip(state.Phi_tilde, U_Phi))
        push!(state.alphas, alpha_bar)

        rho_sq = 1.0 - alpha_bar^2
        if rho_sq < tol^2
            state.exhausted = true
        else
            rho = sqrt(rho_sq)
            push!(state.rhos, rho)
            Phi_new       = (U_Phi     - alpha_bar * state.Phi_tilde) / rho
            Phi_tilde_new = (state.Phi_tilde - alpha_bar * U_Phi)     / rho
            state.Phi       = Phi_new
            state.Phi_tilde = Phi_tilde_new
        end
    end
    return state
end

"""
    apply_blockdiag!(out, v, alphas, rhos, offset)

Apply a direct sum of 2×2 blocks Θ_n = [α_n ρ_n; ρ_n -α_n] to `v`.
offset = 0 → M = Θ₀⊕Θ₂⊕…;  offset = 1 → L = I₁⊕Θ₁⊕Θ₃⊕…
The trailing 1×1 block of either is the closing coefficient α_{K-1}.
"""
function apply_blockdiag!(out::AbstractVector, v::AbstractVector,
                           alphas::Vector{Float64}, rhos::Vector{Float64}, offset::Int)
    K = length(v)
    if offset == 1
        K >= 1 && (out[1] = v[1])
    end
    pos = 1 + offset
    n = offset
    while pos + 1 <= K
        α = alphas[n + 1]
        ρ = (n < length(rhos)) ? rhos[n + 1] : 0.0
        v1, v2 = v[pos], v[pos+1]
        out[pos]   = α * v1 + ρ * v2
        out[pos+1] = ρ * v1 - α * v2
        pos += 2; n += 2
    end
    if pos == K   # trailing 1×1 block
        out[K] = alphas[K] * v[K]
    end
    return out
end

"""
    cmv_matvec(v, alphas, rhos) -> U_CMV * v      (U_CMV = L·M, O(K))

Mirror of the paper's (S11) convention (U = M·L); same Δ'Δ and AGP norm, RHS Δ'e₁ = (α₀−1, ρ₀, 0, …).
"""
function cmv_matvec(v::AbstractVector, alphas::Vector{Float64}, rhos::Vector{Float64})
    tmp = similar(v, Float64)
    out = similar(v, Float64)
    apply_blockdiag!(tmp, v, alphas, rhos, 0)
    apply_blockdiag!(out, tmp, alphas, rhos, 1)
    return out
end

"""
    cmv_matvec_adjoint(v, alphas, rhos) -> U_CMV' * v   (= M·L, O(K))
"""
function cmv_matvec_adjoint(v::AbstractVector, alphas::Vector{Float64}, rhos::Vector{Float64})
    tmp = similar(v, Float64)
    out = similar(v, Float64)
    apply_blockdiag!(tmp, v, alphas, rhos, 1)
    apply_blockdiag!(out, tmp, alphas, rhos, 0)
    return out
end

delta_matvec(v, alphas, rhos) = cmv_matvec(v, alphas, rhos) .- v
delta_adjoint_matvec(v, alphas, rhos) = cmv_matvec_adjoint(v, alphas, rhos) .- v

"""
    cg_normal_equations(alphas, rhos, η; tol=1e-10, maxit=nothing) -> (c, iters)

Solve (Δ'Δ + η²I) c = Δ'e₁, Δ = U_CMV - I, by matrix-free CG.
"""
function cg_normal_equations(alphas::Vector{Float64}, rhos::Vector{Float64}, η::Real;
                              tol::Float64=1e-10, maxit::Union{Int,Nothing}=nothing)
    K = length(alphas)
    maxit_use = maxit === nothing ? K : maxit
    e1 = zeros(Float64, K); e1[1] = 1.0
    b = delta_adjoint_matvec(e1, alphas, rhos)

    Afun(x) = delta_adjoint_matvec(delta_matvec(x, alphas, rhos), alphas, rhos) .+ (η^2) .* x

    x = zeros(Float64, K)
    r = b .- Afun(x)
    p = copy(r)
    rs_old = dot(r, r)
    bnorm = norm(b)
    bnorm == 0 && return x, 0
    for it in 1:maxit_use
        Ap = Afun(p)
        α = rs_old / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rs_new = dot(r, r)
        if sqrt(rs_new) < tol * bnorm
            return x, it
        end
        p .= r .+ (rs_new / rs_old) .* p
        rs_old = rs_new
    end
    return x, maxit_use
end

"""
    agp_norm2_cmv_cg(alphas, rhos, init_norm, η, M; tol=1e-10, closure=1.0)
        -> (norm2, M_use, iters)

Squared regularized AGP norm at Krylov truncation depth M. Unless the Krylov
space is exhausted at M, the truncation is closed with α_{M-1} = closure
(±1), so that U_CMV stays unitary.
"""
function agp_norm2_cmv_cg(alphas::Vector{Float64}, rhos::Vector{Float64},
                           init_norm::Float64, η::Real, M::Int; tol::Float64=1e-10,
                           closure::Float64=1.0)
    M_use = min(M, length(alphas))
    rho_len = min(M_use - 1, length(rhos))
    a = alphas[1:M_use]; r = rhos[1:rho_len]
    M_use <= length(rhos) && (a[end] = closure)   # not the terminal α (|α_{K-1}| = 1 already)
    c, iters = cg_normal_equations(a, r, η; tol=tol)
    return init_norm^2 * sum(abs2, c), M_use, iters
end

"""
    adaptive_converge_cmv_stepwise(step_extend!, state, η;
                                    conv_tol=1e-4, cg_tol=1e-10,
                                    K_hard_cap, patience=1) -> NamedTuple

Grow M one Krylov vector at a time. At each M the norm² is computed with
both unitary closures α_{M-1} = ±1; the closure gap |N₊ − N₋|/N₊ measures
how much the result still depends on the unexplored part of the Krylov
space. Converged once `patience` consecutive gaps fall below `conv_tol`.
N₊ is the reported norm². Returns
(status, M, norm2, metric, K_reached, cg_iters, history), metric = gap.
"""
function adaptive_converge_cmv_stepwise(step_extend!::Function, state, η::Real;
                                         conv_tol::Float64=1e-4, cg_tol::Float64=1e-10,
                                         K_hard_cap::Int, patience::Int=1)
    history = Tuple{Int,Float64,Float64,Int}[]   # (M, norm2, closure gap, cg_iters)
    M = 0
    stable = 0
    norm2_M = NaN
    cg_iters = 0
    K_reached = 0

    while true
        M += 1
        step_extend!(state, M)
        K_reached = length(state.alphas)
        M_eff = min(M, K_reached)

        norm2_M, _, cg_iters = agp_norm2_cmv_cg(state.alphas, state.rhos,
                                                  state.init_norm, η, M_eff; tol=cg_tol)
        norm2_minus, _, _ = agp_norm2_cmv_cg(state.alphas, state.rhos, state.init_norm,
                                             η, M_eff; tol=cg_tol, closure=-1.0)
        metric = abs(norm2_M - norm2_minus) / norm2_M
        push!(history, (M_eff, norm2_M, metric, cg_iters))

        if metric < conv_tol
            stable += 1
            if stable >= patience
                return (status="converged", M=M_eff, norm2=norm2_M, metric=metric,
                        K_reached=K_reached, cg_iters=cg_iters, history=history)
            end
        else
            stable = 0
        end

        if state.exhausted
            return (status="exhausted", M=M_eff, norm2=norm2_M, metric=metric,
                    K_reached=K_reached, cg_iters=cg_iters, history=history)
        end
        if M_eff >= K_hard_cap
            return (status="capped", M=M_eff, norm2=norm2_M, metric=metric,
                    K_reached=K_reached, cg_iters=cg_iters, history=history)
        end
    end
end

"""
    cutoff_eta(j, d) = 0.5 * (2π/d)

Regularization matched to the mean quasi-energy spacing.
"""
cutoff_eta(j::Float64, d::Int) = 0.5 * (2π / d)

"""
    agp_norm_cell(k, j; p=P0, conv_tol=1e-4, cg_tol=1e-10, patience=1)
        -> (d, η, res)

Converged regularized AGP norm² for one (k, j): ∂_k seed, P_plus sector,
η = cutoff_eta(j, d), adaptive Krylov convergence on the closure gap.
"""
function agp_norm_cell(k::Float64, j::Float64; p::Float64=P0, conv_tol::Float64=1e-4,
                       cg_tol::Float64=1e-10, patience::Int=1)
    P_plus, _ = parity_sectors(j)
    U, G, _ = kicked_top_seed_sector_dk(p, j, P_plus; k=k)
    d = size(U, 1)
    η = cutoff_eta(j, d)

    state = szego_init(G)
    step_extend! = (st, M) -> szego_extend!(U, st, M)

    res = adaptive_converge_cmv_stepwise(step_extend!, state, η;
                                          conv_tol=conv_tol, cg_tol=cg_tol,
                                          K_hard_cap=d^2, patience=patience)
    return d, η, res
end

"""
    exact_agp_norm2(U, G, η) -> ‖A^(R)‖²

Exact regularized AGP norm² by full diagonalization of the (sector-
projected) U, as a reference for the Krylov result:
    [A^(R)]_mn = (e^{i(θm-θn)} - 1)^* G_mn / (|e^{i(θm-θn)} - 1|² + η²)
    ‖A^(R)‖²  = (1/d) Σ_{m≠n} |[A^(R)]_mn|²
with G in the eigenbasis of U, and the same tr(A'B)/d normalization as
the Szegő recursion.
"""
function exact_agp_norm2(U::AbstractMatrix, G::AbstractMatrix, η::Float64)
    d = size(U,1)
    F = eigen(U)
    evals = F.values
    V = F.vectors
    theta = angle.(evals)
    Xb = V' * G * V
    norm2 = 0.0
    for m in 1:d, n in 1:d
        m == n && continue
        denom_phase = exp(im*(theta[m]-theta[n])) - 1
        num = conj(denom_phase) * Xb[m,n]
        val = num / (abs2(denom_phase) + η^2)
        norm2 += abs2(val)
    end
    return norm2 / d
end

"""
    exact_agp_cell(k, j; p=P0) -> (d, η, norm2)

Exact AGP norm² for one (k, j), with the same seed, sector and η as
agp_norm_cell.
"""
function exact_agp_cell(k::Float64, j::Float64; p::Float64=P0)
    P_plus, _ = parity_sectors(j)
    U, G, _ = kicked_top_seed_sector_dk(p, j, P_plus; k=k)
    d = size(U, 1)
    η = cutoff_eta(j, d)
    return d, η, exact_agp_norm2(U, G, η)
end

# ═══════════════════════════════════════════════════════════════════════
#  5. Sweep drivers (compute + write CSV)
# ═══════════════════════════════════════════════════════════════════════

function load_done(summary_file::String)
    done = Set{Tuple{Float64,Float64}}()
    isfile(summary_file) || return done
    for line in eachline(summary_file)
        startswith(line, "k,") && continue
        parts = split(line, ",")
        length(parts) >= 2 || continue
        push!(done, (parse(Float64, parts[1]), parse(Float64, parts[2])))
    end
    return done
end

"""
    run_agp_sweep(summary_file="kicked_top_near_zero_k_summary.csv",
                  detail_file="kicked_top_near_zero_k_detail.csv";
                  ks=K_VALUES, js=J_VALUES, p=P0)

AGP-norm sweep over (k, j). Appends to existing files and skips (k, j)
cells already in `summary_file`, so an interrupted run can be resumed.
  summary: k,j,d,eta,K_full,K_reached,M_conv,status,closure_gap_final,norm2_cmv
  detail:  k,j,d,M,norm2,closure_gap,cg_iters   (full convergence history)
"""
function run_agp_sweep(summary_file::String="kicked_top_near_zero_k_summary.csv",
                       detail_file::String="kicked_top_near_zero_k_detail.csv";
                       ks=K_VALUES, js=J_VALUES, p::Float64=P0)
    done = load_done(summary_file)
    summary_exists = isfile(summary_file)
    detail_exists = isfile(detail_file)
    summary_io = open(summary_file, "a")
    detail_io = open(detail_file, "a")
    if !summary_exists
        println(summary_io, "k,j,d,eta,K_full,K_reached,M_conv,status,closure_gap_final,norm2_cmv")
        flush(summary_io)
    end
    if !detail_exists
        println(detail_io, "k,j,d,M,norm2,closure_gap,cg_iters")
        flush(detail_io)
    end

    for k in ks, j in js
        if (k, j) in done
            @printf("skip k=%-6.2f j=%-6.1f (already done)\n", k, j)
            continue
        end
        t0 = time()
        d, η, res = agp_norm_cell(k, j; p=p)
        for (M_h, norm2_h, metric_h, cg_iters_h) in res.history
            println(detail_io, "$k,$j,$d,$M_h,$norm2_h,$metric_h,$cg_iters_h")
        end
        flush(detail_io)
        println(summary_io, "$k,$j,$d,$η,$(d^2),$(res.K_reached),$(res.M),$(res.status),$(res.metric),$(res.norm2)")
        flush(summary_io)
        @printf("k=%-6.2f j=%-6.1f d=%-6d  norm2_cmv=%.4e  M(status=%s)=%d  [%.2fs]\n",
                k, j, d, res.norm2, res.status, res.M, time() - t0)
    end
    close(summary_io)
    close(detail_io)
end

"""
    run_lyapunov_sweep(out_file="kicked_top_lyapunov_sweep.csv";
                       ks=K_VALUES, p=P0, n_ic=50, n_steps=1000, n_transient=200,
                       d0=1e-8, seed=1)

Classical Lyapunov exponent λ(k). Writes: k,lyapunov
"""
function run_lyapunov_sweep(out_file::String="kicked_top_lyapunov_sweep.csv";
                            ks=K_VALUES, p::Float64=P0, n_ic::Int=50, n_steps::Int=1000,
                            n_transient::Int=200, d0::Float64=1e-8, seed::Int=1)
    open(out_file, "w") do io
        println(io, "k,lyapunov")
        for k in ks
            λ = lyapunov_exponent(p, k; n_ic=n_ic, n_steps=n_steps,
                                  n_transient=n_transient, d0=d0, seed=seed)
            println(io, "$k,$λ")
            @printf("  k=%-5.2f  λ=%.5f\n", k, λ)
        end
    end
    println("Saved $out_file")
end

"""
    run_level_stats_sweep(out_file="level_stats_sweep.csv"; j=150.0,
                          ks=vcat(0.0:0.25:2.0, 2.25:0.15:4.0, 4.25:0.25:10.0))

Mean level-spacing ratio ⟨r⟩(k) in the P_plus sector, averaged over 9
precession angles p ∈ [π/2 - 0.15, π/2 + 0.15].
Writes: k,r_mean,r_std,n_samples   (r_std = standard error of the mean)
"""
function run_level_stats_sweep(out_file::String="level_stats_sweep.csv"; j::Float64=150.0,
                               ks=vcat(0.0:0.25:2.0, 2.25:0.15:4.0, 4.25:0.25:10.0))
    P_plus, _ = parity_sectors(j)
    println("j=$j, d=$(Int(2j+1)), sector dim=$(size(P_plus,2))")
    p_window = range(π/2 - 0.15, π/2 + 0.15, length=9)

    open(out_file, "w") do io
        println(io, "k,r_mean,r_std,n_samples")
        for k in ks
            rvals = Float64[]
            for p in p_window
                U, _, _ = kicked_top_floquet(p, j; k=k)
                append!(rvals, level_spacing_ratio(project_to_sector(U, P_plus)))
            end
            rvals = filter(isfinite, rvals)
            rm = mean(rvals)
            rs = std(rvals) / sqrt(length(rvals))
            println(io, "$k,$rm,$rs,$(length(rvals))")
            @printf("  k=%-5.2f  <r>=%.4f ± %.4f  (n=%d)\n", k, rm, rs, length(rvals))
        end
    end
    println("Saved $out_file")
end

"""
    run_exact_reference(out_file="kicked_top_exact_agp_reference.csv";
                        ks=K_VALUES, j=200.0, p=P0)

Exact-diagonalization AGP norms at j = 200: reference lines in the
convergence-vs-M figure, and the target that defines M-to-converge. Writes: k,j,d,eta,exact_norm2,exact_norm
"""
function run_exact_reference(out_file::String="kicked_top_exact_agp_reference.csv";
                             ks=K_VALUES, j::Float64=200.0, p::Float64=P0)
    open(out_file, "w") do io
        println(io, "k,j,d,eta,exact_norm2,exact_norm")
        for k in ks
            d, η, n2 = exact_agp_cell(k, j; p=p)
            println(io, "$k,$j,$d,$η,$n2,$(sqrt(n2))")
            @printf("  k=%-5.2f j=%-6.1f  exact ‖A‖=%.6f\n", k, j, sqrt(n2))
        end
    end
    println("Saved $out_file")
end

end # module KickedTopCore

# ═══════════════════════════════════════════════════════════════════════
#  Command-line entry point
# ═══════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    using .KickedTopCore
    which = isempty(ARGS) ? "all" : ARGS[1]
    which in ("all", "level")    && run_level_stats_sweep()
    which in ("all", "lyapunov") && run_lyapunov_sweep()
    which in ("all", "agp")      && run_agp_sweep()
    which in ("all", "exact")    && run_exact_reference()
end
