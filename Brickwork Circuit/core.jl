"""
core.jl

Physics core for the L = 12 XXZ brickwork figures:

  Figures/AGP_norm_Brickwork.pdf        <- run_agp_norm_sweep.jl   -> agp_norm_L12.csv
  Figures/AGP_Brickwork_Infidelity.pdf  <- run_infidelity_ramp.jl  -> infidelity_L12.csv

(plots made by generate_figures.jl)

Contents:
  1. XXZ gate Ř(θ,γ) (γ: anisotropy angle, Δ = cos γ), its θ-derivative, and the local seed generator
  2. Brickwork circuit U = U_odd·U_even and seed G_θ = -i(∂_θU)U†, built by
     applying two-site gates as tensor contractions (never forming a
     dense 2^L×2^L gate, no dense d×d matmuls)
  3. Symmetry sectors: S^z = 0 (half filling) ∩ reflection R ∩ spin flip F
  4. Szegő recursion for the Verblunsky coefficients (α_k, ρ_k) of
     𝒰(X) = U X U† seeded by G_θ
  5. Regularized AGP in the CMV basis: coefficients by matrix-free CG on
     (Δ'Δ + η²I)c = Δ'e₁, Δ = U_CMV − I; norm from the coefficients alone,
     operator by one forward Szegő sweep
  6. Exact (eigendecomposition) regularized AGP, the reference curve
"""
module BrickworkCore

using LinearAlgebra

export build_xxz_gate, build_xxz_gate_derivative, local_seed_generator,
       circuit_seed, half_filling_projector, ground_sector_projector, sector_seed,
       SzegoState, szego_init, szego_extend!,
       cmv_coeffs, agp_norm2_cmv_cg, reconstruct_multi_M_cmv,
       exact_agp_regulated_A, agp_norm

# ═══════════════════════════════════════════════════════════════════════
#  1. Gate primitives
# ═══════════════════════════════════════════════════════════════════════

"""
    build_xxz_gate(θ, γ)

4×4 XXZ Yang–Baxter gate Ř(θ,γ) in the basis {|00⟩,|01⟩,|10⟩,|11⟩}:

  N = sin(γ)cosh(θ) + i cos(γ)sinh(θ),  a = sin(γ)/N,  b = i sinh(θ)/N
  Ř = [1 0 0 0; 0 a b 0; 0 b a 0; 0 0 0 1]

Unitary (|a|² + |b|² = 1); Ř(0) = 𝕀.
"""
function build_xxz_gate(θ::Real, γ::Real)
    N = sin(γ)*cosh(θ) + im*cos(γ)*sinh(θ)
    a = sin(γ) / N
    b = im*sinh(θ) / N
    return ComplexF64[1 0 0 0;
                      0 a b 0;
                      0 b a 0;
                      0 0 0 1]
end

"""
    build_xxz_gate_derivative(θ, γ)

∂_θŘ:  N′ = sin(γ)sinh(θ) + i cos(γ)cosh(θ),
       a′ = −sin(γ)N′/N²,  b′ = i(cosh(θ)N − sinh(θ)N′)/N².
"""
function build_xxz_gate_derivative(θ::Real, γ::Real)
    N  = sin(γ)*cosh(θ) + im*cos(γ)*sinh(θ)
    N′ = sin(γ)*sinh(θ) + im*cos(γ)*cosh(θ)
    a′ = -sin(γ)*N′ / N^2
    b′ = im*(cosh(θ)*N - sinh(θ)*N′) / N^2
    return ComplexF64[0  0  0  0;
                      0 a′ b′  0;
                      0 b′ a′  0;
                      0  0  0  0]
end

"""
    local_seed_generator(θ, γ)

Two-site seed g = −i(∂_θŘ)Ř†, so that ∂_θŘ = i g Ř.
"""
local_seed_generator(θ::Real, γ::Real) =
    -im * build_xxz_gate_derivative(θ, γ) * build_xxz_gate(θ, γ)'

# ═══════════════════════════════════════════════════════════════════════
#  2. Brickwork circuit by local two-site gate application
# ═══════════════════════════════════════════════════════════════════════
#
#  Site 1 is the most significant tensor factor: a gate on bond j (sites
#  j, j+1) is 𝕀_{2^(j-1)} ⊗ gate ⊗ 𝕀_{2^(L-j-1)}. In Julia's column-major
#  order that is reshape(X, right_dim, 4, left_dim, ncols). Applying one
#  gate costs O(d·ncols), a full layer O(L·d·ncols).

"""
    apply_two_site_left(X, gate, j, L) -> (𝕀 ⊗ gate ⊗ 𝕀) · X
"""
function apply_two_site_left(X::AbstractMatrix, gate::AbstractMatrix, j::Int, L::Int)
    d         = size(X, 1)
    left_dim  = 2^(j-1)
    right_dim = 2^(L-j-1)
    ncols     = size(X, 2)
    @assert left_dim * 4 * right_dim == d "bond j=$j inconsistent with L=$L"

    Xp  = permutedims(reshape(X, right_dim, 4, left_dim, ncols), (2, 1, 3, 4))
    Yp  = reshape(gate * reshape(Xp, 4, :), 4, right_dim, left_dim, ncols)
    return reshape(permutedims(Yp, (2, 1, 3, 4)), d, ncols)
end

"""
    apply_two_site_right(X, gate, j, L) -> X · (𝕀 ⊗ gate ⊗ 𝕀)

Uses embed(gate)ᵀ = embed(gateᵀ).
"""
function apply_two_site_right(X::AbstractMatrix, gate::AbstractMatrix, j::Int, L::Int)
    Yt = apply_two_site_left(collect(transpose(X)), collect(transpose(gate)), j, L)
    return collect(transpose(Yt))
end

"""
    bond_list(L, parity)

Left sites of the bonds in one layer: :even → (1,2),(3,4),…;
:odd → (2,3),(4,5),… (open boundary).
"""
bond_list(L::Int, parity::Symbol) = collect((parity == :even ? 1 : 2):2:L-1)

"""
    apply_layer_left(X, gate, L, parity) -> U_layer · X
    apply_layer_right(X, gate, L, parity) -> X · U_layer

U_layer = product of `gate` on every bond of the given parity (the bonds
do not overlap, so the order does not matter).
"""
function apply_layer_left(X::AbstractMatrix, gate::AbstractMatrix, L::Int, parity::Symbol)
    for j in bond_list(L, parity)
        X = apply_two_site_left(X, gate, j, L)
    end
    return X
end

function apply_layer_right(X::AbstractMatrix, gate::AbstractMatrix, L::Int, parity::Symbol)
    for j in bond_list(L, parity)
        X = apply_two_site_right(X, gate, j, L)
    end
    return X
end

"""
    circuit_seed(θ, γ, L) -> (U, G_θ, ∂_θU)

Full-space (2^L × 2^L) brickwork unitary U = U_odd·U_even, its seed
G_θ = G_odd + U_odd G_even U_odd†, and ∂_θU = dU_odd·U_even + U_odd·dU_even,
where G_layer = Σ_bonds g and dU_layer = Σ_bonds (i g)·U_layer.
"""
function circuit_seed(θ::Real, γ::Real, L::Int)
    d      = 2^L
    R      = build_xxz_gate(θ, γ)
    g_loc  = local_seed_generator(θ, γ)
    ig_loc = im * g_loc
    Id     = Matrix{ComplexF64}(I, d, d)

    U_even = apply_layer_left(Id, R, L, :even)
    U_odd  = apply_layer_left(Id, R, L, :odd)
    U      = apply_layer_left(U_even, R, L, :odd)          # U_odd · U_even

    G_even = zeros(ComplexF64, d, d)
    for j in bond_list(L, :even)
        G_even += apply_two_site_left(Id, g_loc, j, L)
    end
    G_odd = zeros(ComplexF64, d, d)
    for j in bond_list(L, :odd)
        G_odd += apply_two_site_left(Id, g_loc, j, L)
    end
    G = G_odd + apply_layer_right(apply_layer_left(G_even, R, L, :odd), R', L, :odd)

    dU_odd = zeros(ComplexF64, d, d)
    for j in bond_list(L, :odd)
        dU_odd += apply_two_site_left(U_odd, ig_loc, j, L)
    end
    dU_even = zeros(ComplexF64, d, d)
    for j in bond_list(L, :even)
        dU_even += apply_two_site_left(U_even, ig_loc, j, L)
    end
    dU = apply_layer_right(dU_odd, R, L, :even) + apply_layer_left(dU_even, R, L, :odd)

    return U, G, dU
end

# ═══════════════════════════════════════════════════════════════════════
#  3. Symmetry sectors
# ═══════════════════════════════════════════════════════════════════════

"""
    half_filling_projector(L)

2^L × C(L, L/2) isometry onto the S^z = 0 sector (basis states with L/2
set bits).
"""
function half_filling_projector(L::Int)
    @assert iseven(L) "L must be even for the S^z=0 half-filling sector"
    indices = [k + 1 for k in 0:2^L-1 if count_ones(k) == L ÷ 2]
    P = zeros(ComplexF64, 2^L, length(indices))
    for (col, row) in enumerate(indices)
        P[row, col] = 1.0
    end
    return P
end

"""
    four_sectors(L, P) -> (Qpp, Qpm, Qmp, Qmm)

Joint eigenbases of the commuting reflection R (site j ↔ L+1−j) and
global spin flip F within the sector spanned by P, for
(R,F) = (+1,+1), (+1,−1), (−1,+1), (−1,−1). Each block is U-invariant.
"""
function four_sectors(L::Int, P::AbstractMatrix)
    dim = 2^L
    R_full = zeros(ComplexF64, dim, dim)
    F_full = zeros(ComplexF64, dim, dim)
    mask = (1 << L) - 1
    for state in 0:dim-1
        ref = 0
        for bit in 0:L-1
            ref |= ((state >> bit) & 1) << (L-1-bit)
        end
        R_full[ref+1, state+1] = 1.0
        F_full[xor(state, mask)+1, state+1] = 1.0
    end
    R_sec = P' * R_full * P
    F_sec = P' * F_full * P

    valR, vecR = eigen(Hermitian(R_sec))
    Qp = vecR[:, findall(v -> real(v) >  0.5, valR)]
    Qm = vecR[:, findall(v -> real(v) < -0.5, valR)]

    function split_F(Q)
        valF, vecF = eigen(Hermitian(Q' * F_sec * Q))
        return Q * vecF[:, findall(v -> real(v) >  0.5, valF)],
               Q * vecF[:, findall(v -> real(v) < -0.5, valF)]
    end

    Qpp, Qpm = split_F(Qp)
    Qmp, Qmm = split_F(Qm)
    return Qpp, Qpm, Qmp, Qmm
end

"""
    ground_sector_projector(L, γ, P; θ_probe=0.05) -> (P_comb, sector_name, dim_sub)

Of the four R×F blocks inside P, pick the one holding the lowest
eigenphase of U(θ_probe) (small θ, where the assignment is unambiguous),
and return the combined isometry P_comb = P·Q onto that block.
"""
function ground_sector_projector(L::Int, γ::Real, P::AbstractMatrix; θ_probe::Float64=0.05)
    Qs    = four_sectors(L, P)
    names = ["R=+1,F=+1", "R=+1,F=-1", "R=-1,F=+1", "R=-1,F=-1"]

    R      = build_xxz_gate(θ_probe, γ)
    U_full = apply_layer_left(apply_layer_left(Matrix{ComplexF64}(I, 2^L, 2^L), R, L, :even),
                              R, L, :odd)
    U_sec  = P' * U_full * P
    phases = [minimum(angle.(eigvals(Q' * U_sec * Q))) for Q in Qs]
    k = argmin(phases)
    return P * Qs[k], names[k], size(Qs[k], 2)
end

"""
    sector_seed(θ, γ, L, P_comb) -> (U, G, dU)

circuit_seed restricted to the sector: P'XP for each of U, G_θ, ∂_θU.
"""
function sector_seed(θ::Real, γ::Real, L::Int, P_comb::AbstractMatrix)
    U, G, dU = circuit_seed(θ, γ, L)
    return P_comb' * U * P_comb, P_comb' * G * P_comb, P_comb' * dU * P_comb
end

# ═══════════════════════════════════════════════════════════════════════
#  4. Szegő recursion (resumable)
# ═══════════════════════════════════════════════════════════════════════
#
#  Inner product ⟨A,B⟩ = tr(A†B)/d. Starting from Φ₀ = Φ̃₀ = G/‖G‖:
#     α_k   = Re⟨Φ̃_k, 𝒰Φ_k⟩,   ρ_k = √(1 − α_k²)
#     Φ_{k+1} = (𝒰Φ_k − α_k Φ̃_k)/ρ_k,   Φ̃_{k+1} = (Φ̃_k − α_k 𝒰Φ_k)/ρ_k
#  Only the running pair (Φ, Φ̃) is kept, so memory is O(d²) at any depth.

mutable struct SzegoState
    Phi::Matrix{ComplexF64}
    Phi_tilde::Matrix{ComplexF64}
    alphas::Vector{Float64}
    rhos::Vector{Float64}
    init_norm::Float64
    exhausted::Bool
end

"""
    szego_init(G) -> SzegoState
"""
function szego_init(G::AbstractMatrix; tol::Float64=1e-12)
    d = size(G, 1)
    init_norm = sqrt(abs(real(tr(G' * G))) / d)
    @assert init_norm > tol "Seed operator has zero norm."
    Phi = G ./ init_norm
    return SzegoState(Phi, copy(Phi), Float64[], Float64[], init_norm, false)
end

"""
    szego_extend!(U, state, K_target; tol=1e-12)

Run the recursion until `length(state.alphas) ≥ K_target`, or until
ρ² < tol², when the Krylov space is exhausted (`state.exhausted = true`).
"""
function szego_extend!(U::AbstractMatrix, state::SzegoState, K_target::Int;
                        tol::Float64=1e-12)
    d = size(U, 1)
    while length(state.alphas) < K_target && !state.exhausted
        U_Phi = U * state.Phi * U'
        α = real(tr(state.Phi_tilde' * U_Phi) / d)
        push!(state.alphas, α)

        ρ² = 1.0 - α^2
        if ρ² < tol^2
            state.exhausted = true
        else
            ρ = sqrt(ρ²)
            push!(state.rhos, ρ)
            Phi_new         = (U_Phi - α * state.Phi_tilde) / ρ
            state.Phi_tilde = (state.Phi_tilde - α * U_Phi) / ρ
            state.Phi       = Phi_new
        end
    end
    return state
end

# ═══════════════════════════════════════════════════════════════════════
#  5. Regularized AGP in the CMV basis
# ═══════════════════════════════════════════════════════════════════════
#
#  U_CMV = 𝓛·𝓜 with 𝓜 = Θ₀⊕Θ₂⊕…, 𝓛 = 𝕀₁⊕Θ₁⊕Θ₃⊕…, Θ_n = [α_n ρ_n; ρ_n −α_n].
#  Mirror of the paper's (S11) convention (U = 𝓜𝓛, basis {G, 𝒰G, 𝒰⁻¹G, …}); same Δ'Δ and AGP norm, RHS Δ'e₁ = (α₀−1, ρ₀, 0, …).
#  Each Θ_n is a symmetric orthogonal involution, so with the closure
#  α_{M−1} = 1 (see cmv_coeffs) U_CMV is exactly unitary at every
#  truncation depth and U_CMVᵀ = 𝓜𝓛. Both products cost
#  O(M); the K×K matrix is never built. (α, ρ are real for this model.)

"""
    apply_blockdiag!(out, v, alphas, rhos, offset)

offset = 0: 𝓜 v (blocks from position 1; trailing 1×1 block α_{K−1}).
offset = 1: 𝓛 v (leading 1×1 identity). Either trailing 1×1 block is α_{K−1}.
"""
function apply_blockdiag!(out::AbstractVector, v::AbstractVector,
                           alphas::Vector{Float64}, rhos::Vector{Float64}, offset::Int)
    K = length(v)
    offset == 1 && K >= 1 && (out[1] = v[1])
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
    if pos == K
        out[K] = alphas[K] * v[K]
    end
    return out
end

"""
    cmv_matvec(v, alphas, rhos)          -> U_CMV v  = 𝓛(𝓜 v)
    cmv_matvec_adjoint(v, alphas, rhos)  -> U_CMV' v = 𝓜(𝓛 v)
"""
function cmv_matvec(v::AbstractVector, alphas::Vector{Float64}, rhos::Vector{Float64})
    tmp = similar(v, Float64); out = similar(v, Float64)
    apply_blockdiag!(tmp, v, alphas, rhos, 0)
    return apply_blockdiag!(out, tmp, alphas, rhos, 1)
end

function cmv_matvec_adjoint(v::AbstractVector, alphas::Vector{Float64}, rhos::Vector{Float64})
    tmp = similar(v, Float64); out = similar(v, Float64)
    apply_blockdiag!(tmp, v, alphas, rhos, 1)
    return apply_blockdiag!(out, tmp, alphas, rhos, 0)
end

"""
    cg_normal_equations(alphas, rhos, η; tol=1e-10) -> (c, iters)

CG solve of (Δ'Δ + η²I)c = Δ'e₁ with Δ = U_CMV − I, using only the O(K)
matvecs above.
"""
function cg_normal_equations(alphas::Vector{Float64}, rhos::Vector{Float64}, η::Real;
                              tol::Float64=1e-10)
    K = length(alphas)
    Δ(x)  = cmv_matvec(x, alphas, rhos) .- x
    Δt(x) = cmv_matvec_adjoint(x, alphas, rhos) .- x
    A(x)  = Δt(Δ(x)) .+ (η^2) .* x

    e1 = zeros(Float64, K); e1[1] = 1.0
    b = Δt(e1)
    x = zeros(Float64, K)
    bnorm = norm(b)
    bnorm == 0 && return x, 0
    r = b .- A(x)
    p = copy(r)
    rs_old = dot(r, r)
    for it in 1:K
        Ap = A(p)
        a = rs_old / dot(p, Ap)
        x .+= a .* p
        r .-= a .* Ap
        rs_new = dot(r, r)
        sqrt(rs_new) < tol * bnorm && return x, it
        p .= r .+ (rs_new / rs_old) .* p
        rs_old = rs_new
    end
    return x, K
end

"""
    cmv_coeffs(alphas, rhos, η, M; tol=1e-10) -> (c, M_use, iters)

AGP coefficients in the CMV basis at truncation depth M (capped at the
number of available α's). Unless the Krylov space is exhausted at M, the
truncation is closed with α_{M−1} = 1 so that U_CMV stays unitary.
"""
function cmv_coeffs(alphas::Vector{Float64}, rhos::Vector{Float64}, η::Real, M::Int;
                    tol::Float64=1e-10)
    M_use = min(M, length(alphas))
    a = alphas[1:M_use]
    M_use <= length(rhos) && (a[end] = 1.0)   # not the terminal α (|α_{K−1}| = 1 already)
    c, iters = cg_normal_equations(a, rhos[1:min(M_use - 1, length(rhos))], η;
                                   tol=tol)
    return c, M_use, iters
end

"""
    agp_norm2_cmv_cg(alphas, rhos, init_norm, η, M; tol=1e-10) -> (‖A_M‖², M_use, iters)

Squared regularized AGP norm, ‖A_M‖² = init_norm² Σ_k |c_k|² (orthonormal
basis), with no d×d operator built.
"""
function agp_norm2_cmv_cg(alphas::Vector{Float64}, rhos::Vector{Float64},
                           init_norm::Float64, η::Real, M::Int; tol::Float64=1e-10)
    c, M_use, iters = cmv_coeffs(alphas, rhos, η, M; tol=tol)
    return init_norm^2 * sum(abs2, c), M_use, iters
end

"""
    reconstruct_multi_M_cmv(U, G, state, η, M_list; tol=1e-10)
        -> Dict M => (A_M, M_use)

AGP operator A_M = init_norm · Σ_{k<M} c_k x_k for every M in M_list, in
one forward sweep. The CMV basis is the orthonormalization of
{G, 𝒰⁻¹G, 𝒰G, 𝒰⁻²G, …}:
    x_k = 𝒰^{−⌈k/2⌉} Φ_k  (k even),   x_k = 𝒰^{−⌈k/2⌉} Φ̃_k  (k odd).
The recursion is linear and commutes with 𝒰⁻¹, so it is run on the
shifted pair (𝒰^{−s}Φ_k, 𝒰^{−s}Φ̃_k), with s raised by applying 𝒰⁻¹ in
place. `state` must come from szego_init(G) + szego_extend!.
"""
function reconstruct_multi_M_cmv(U::AbstractMatrix, G::AbstractMatrix,
                                  state::SzegoState, η::Real, M_list::Vector{Int};
                                  tol::Float64=1e-10)
    alphas, rhos, init_norm = state.alphas, state.rhos, state.init_norm
    K_reached   = length(alphas)
    M_used_list = sort(unique(min.(M_list, K_reached)))
    c_per_M     = Dict(M => cmv_coeffs(alphas, rhos, η, M; tol=tol)[1] for M in M_used_list)
    max_M       = maximum(M_used_list)

    d = size(U, 1)
    U_op(X)  = U * X * U'
    U_inv(X) = U' * X * U
    acc = Dict(M => zeros(ComplexF64, d, d) for M in M_used_list)

    P  = Matrix{ComplexF64}(G) ./ init_norm   # 𝒰^{−s} Φ_k
    Pt = copy(P)                              # 𝒰^{−s} Φ̃_k
    s  = 0
    for k in 0:max_M-1
        while s < cld(k, 2)
            P, Pt = U_inv(P), U_inv(Pt)
            s += 1
        end
        x_k = iseven(k) ? P : Pt
        for M in M_used_list
            k < M && (acc[M] .+= c_per_M[M][k+1] .* x_k)
        end

        k == max_M - 1 && break
        α, ρ = alphas[k+1], rhos[k+1]
        UP = U_op(P)
        P, Pt = (UP .- α .* Pt) ./ ρ, (Pt .- α .* UP) ./ ρ
    end

    return Dict(M => (init_norm .* acc[min(M, K_reached)], min(M, K_reached)) for M in M_list)
end

# ═══════════════════════════════════════════════════════════════════════
#  6. Exact regularized AGP (reference)
# ═══════════════════════════════════════════════════════════════════════

"""
    exact_agp_regulated_A(U, dU, η)

In the eigenbasis of U (eigenphases φ_m), with G = −i dU U†:
    A_mn = conj(z_mn) / (|z_mn|² + η²) · G_mn,   z_mn = e^{i(φ_m − φ_n)} − 1,
with A_mm = 0. Reduces to the exact AGP as η → 0.
"""
function exact_agp_regulated_A(U, dU, η::Real)
    n = size(U, 1)
    G = -im * dU * U'
    evals, P = eigen(U)
    phases = angle.(evals)
    G_eig  = P' * G * P
    A_eig  = zeros(ComplexF64, n, n)
    for m in 1:n, k in 1:n
        m == k && continue
        z = exp(im * (phases[m] - phases[k])) - 1
        A_eig[m, k] = conj(z) / (abs2(z) + η^2) * G_eig[m, k]
    end
    return P * A_eig * P'
end

"""
    agp_norm(U, A) = tr(A†A)/d
"""
agp_norm(U, A) = real(tr(A' * A) / size(U, 1))

end # module BrickworkCore
