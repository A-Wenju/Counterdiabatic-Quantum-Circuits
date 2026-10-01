"""
generate_kicked_top_convergence_split.jl

Krylov-convergence figures for the AGP norm at j=200, for three
representative regimes: integrable k=0.5, transition k=2.5, chaotic k=6.0.

Outputs (in Figures/):

  1. kicked_top_agp_convergence_vs_M.pdf
     Squared regularized AGP norm ‖A_k^(R)‖² vs. Krylov truncation order M
     (log scale), with dotted horizontal lines marking the exact-diagonalization norm² for
     each k.

  2. kicked_top_agp_Mconv_vs_k.pdf
     Number of Krylov steps M needed to converge (first even M whose norm²
     is within 1e-4 of the exact value) vs. kick strength k, over the full k grid,
     referenced against the maximal Krylov dimension K = d^2 - d + 1 for
     the parity-sector dimension d=201 (j=200).

  3. kicked_top_combined_inset.pdf
     Figure 2 as the main panel with figure 1 as an inset.

Required input data files (current directory):
  - kicked_top_near_zero_k_detail.csv     (per-Krylov-step convergence history,
                                            from core.jl run_agp_sweep)
  - kicked_top_exact_agp_reference.csv    (exact-diagonalization reference norms
                                            at j=200 for every k on the grid,
                                            from core.jl run_exact_reference)

Requires: DelimitedFiles, Plots, LaTeXStrings
"""

using DelimitedFiles
using Plots
using LaTeXStrings


# ─────────────────────────────────────────────────────────────────────────
# CSV loading (simple header + Float64 body, no CSV.jl dependency)
# ─────────────────────────────────────────────────────────────────────────
function read_csv_cols(path)
    raw, header = readdlm(path, ',', header=true)
    cols = Dict{String,Any}()
    for (i, name) in enumerate(vec(header))
        col = raw[:, i]
        cols[strip(String(name))] = try
            Float64.(col)
        catch
            String.(col)
        end
    end
    return cols
end


# ─────────────────────────────────────────────────────────────────────────
# Publication figure style
# ─────────────────────────────────────────────────────────────────────────
function apply_figure_style()
    default(
        titlefontsize=18,
        guidefontsize=24,
        tickfontsize=22,
        legendfontsize=18,
        frame=:box,
        grid=false,
        fontfamily="Computer Modern",
        legend=false,
        left_margin=3Plots.mm, bottom_margin=6Plots.mm
    )
end

# canvas, text and line sizes shared with the brickwork figures
# (Brickwork Circuit/generate_figures.jl), so they print at the same scale
const BRICKWORK_SIZES = (size=(700, 500), guidefontsize=30, tickfontsize=26, legendfontsize=22)
const BRICKWORK_LINEWIDTH = 3.2


"""
    convergence_curve(detail, j_target, k_target) -> (M, norm2)
"""
function convergence_curve(detail, j_target, k_target)
    j_col, k_col, M_col, norm2_col = detail["j"], detail["k"], detail["M"], detail["norm2"]
    mask = isapprox.(j_col, j_target; atol=1e-9) .& isapprox.(k_col, k_target; atol=1e-9) .& (mod.(M_col, 2) .== 0)
    M = M_col[mask]
    norm2 = norm2_col[mask]
    order = sortperm(M)
    return M[order], norm2[order]
end


"""
    mconv_curve(detail, exact_norm2, j_target; rel_err_tol=1e-4) -> (k_all, Mconv)

For every k on the near-zero-k grid at j=j_target, the smallest even M at
which |‖A_M‖² − ‖A‖²_exact| / ‖A‖²_exact drops below rel_err_tol (NaN if
never reached). `exact_norm2` maps k => exact ‖A‖² at j_target.
"""
function mconv_curve(detail, exact_norm2, j_target; rel_err_tol=1e-4)
    j_col, k_col, M_col, norm2_col = detail["j"], detail["k"], detail["M"], detail["norm2"]
    jmask = isapprox.(j_col, j_target; atol=1e-9)
    k_all = sort(unique(k_col[jmask]))
    Mconv = Float64[]
    for k in k_all
        mask = jmask .& isapprox.(k_col, k; atol=1e-9) .& (mod.(M_col, 2) .== 0)
        Mk = M_col[mask]
        relk = abs.(norm2_col[mask] .- exact_norm2[k]) ./ exact_norm2[k]
        order = sortperm(Mk)
        Mk, relk = Mk[order], relk[order]
        conv_idx = findfirst(relk .< rel_err_tol)
        push!(Mconv, conv_idx === nothing ? NaN : Mk[conv_idx])
    end
    return k_all, Mconv
end


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: regularized AGP norm vs. Krylov truncation order M
# ─────────────────────────────────────────────────────────────────────────
function make_convergence_vs_M_figure(detail, exact_norm2)
    j_target = 200.0
    k_reps = [(0.5, "#3A6FB3", "integrable"), (2.5, "#D08A2B", "transition"), (6.0, "#B33A3A", "chaotic")]


    apply_figure_style()
    p = plot(; xscale=:log10, yscale=:log10, legend=:topleft, right_margin=6Plots.mm, BRICKWORK_SIZES...)
    for (k, color, label) in k_reps
        M, norm2 = convergence_curve(detail, j_target, k)
        curve_label = latexstring("(k=$k)")
        plot!(p, M, norm2, color=color, linewidth=BRICKWORK_LINEWIDTH, label=curve_label)
        hline!(p, [exact_norm2[k]], color=color, linestyle=:dot, linewidth=2.6, alpha=0.8, label=false)
    end
    # redraw the k=0.5 curve on top: at small M it lies under the other two
    k, color, _ = k_reps[1]
    plot!(p, convergence_curve(detail, j_target, k)..., color=color, linewidth=BRICKWORK_LINEWIDTH, label=false)
    ylims!(p, 60, 2e5)
    xlabel!(p, L"Krylov truncation order $M$")
    ylabel!(p, L"$‖A_k^{(R)}‖^2$")

    savefig(p, "Figures/kicked_top_agp_convergence_vs_M.pdf")
    println("saved kicked_top_agp_convergence_vs_M.pdf")
end


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: number of Krylov steps to converge vs. twist strength k
# ─────────────────────────────────────────────────────────────────────────
function make_Mconv_vs_k_figure(detail, exact_norm2)
    j_target = 200.0
    k_all, Mconv = mconv_curve(detail, exact_norm2, j_target)

    # K_max uses the PARITY-SECTOR dimension d=201 at j=200 (not the full
    # 2j+1=401 spin dimension) -- confirmed against exact diagonalization.
    d_correct = 201
    K_max = d_correct^2 - d_correct + 1

    apply_figure_style()
    p = scatter(k_all, Mconv, markershape=:square, markersize=6, markercolor="#2E8B57",
                markerstrokewidth=0, yscale=:log10, label=false,
                left_margin=6Plots.mm; BRICKWORK_SIZES...)
    hline!(p, [K_max], color=RGB(0.5, 0.5, 0.5), linestyle=:dash, linewidth=BRICKWORK_LINEWIDTH,
           label=L"K", legend=:topleft)
    xlabel!(p, L"kick strength $k$")
    ylabel!(p, L"$M$ to converge ") #$(\mathrm{rel\_err}<10^{-4})$

    savefig(p, "Figures/kicked_top_agp_Mconv_vs_k.pdf")
    println("saved kicked_top_agp_Mconv_vs_k.pdf")
end


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Figure 2 (Mconv vs k) as the main panel, with Figure 1
# (regularized AGP norm vs Krylov order M) shown as an inset inside it.
# ─────────────────────────────────────────────────────────────────────────
"""
    make_combined_inset_figure(detail, exact_norm2; inset_bbox=bbox(0.08, 0.225, 0.35, 0.25, :bottom, :right))

Combine the two split figures into one: the M-to-converge-vs-k scatter
(figure 2) fills the full axes, and the regularized-AGP-norm-vs-M curves
(figure 1) are drawn as a smaller inset panel positioned within it.

`inset_bbox` controls the inset's size/position -- see Plots.jl's `bbox`
(x-offset, y-offset, width, height, corner-anchor...), all as fractions of
the parent axes. The default sits in the lower right, below the
plateau of converged M values and clear of the main x-axis.
"""
function make_combined_inset_figure(detail, exact_norm2; inset_bbox=bbox(0.08, 0.225, 0.35, 0.25, :bottom, :right))
    j_target = 200.0

    # ---- main panel: number of Krylov steps to converge vs. k (figure 2) ----
    k_all, Mconv = mconv_curve(detail, exact_norm2, j_target)
    d_correct = 201
    K_max = d_correct^2 - d_correct + 1

    apply_figure_style()
    p = scatter(k_all, Mconv, markershape=:square, markersize=5, markercolor="#2E8B57",
                markerstrokewidth=0, yscale=:log10, label=false)
    hline!(p, [K_max], color=RGB(0.5, 0.5, 0.5), linestyle=:dash, linewidth=2.8,
           label=L"K=d^2-d+1", legend=:topleft)
    xlabel!(p, L"kick strength $k$")
    ylabel!(p, L"$M$ to converge ")

    # ---- inset: regularized AGP norm vs. Krylov truncation order M (figure 1) ----
    plot!(p, inset=(1, inset_bbox))
    sp = p[2]

    k_reps = [(0.5, "#3A6FB3", "k=0.5"), (2.5, "#D08A2B", "k=2.5"), (6.0, "#B33A3A", "k=6.0")]

    plot!(sp, xscale=:log10, yscale=:log10, frame=:box, grid=false,
          titlefontsize=9, guidefontsize=16, tickfontsize=14, legend = true, legendfontsize = 11, yticks=([1e2, 1e3, 1e4], [L"10^2", L"10^3", L"10^4"]))
    for (k, color, label) in k_reps
        M, norm2 = convergence_curve(detail, j_target, k)
        plot!(sp, M, norm2, color=color, linewidth=3.0, label=label, legend = :topleft)
        hline!(sp, [exact_norm2[k]], color=color, linestyle=:dot, linewidth=2.2, alpha=0.8, label=false)
    end
    # redraw the k=0.5 curve on top: at small M it lies under the other two
    k, color, _ = k_reps[1]
    plot!(sp, convergence_curve(detail, j_target, k)..., color=color, linewidth=3.0, label=false)
    ylims!(sp, 60, 2e5)
    xlabel!(sp, L"$M$")
    ylabel!(sp, L"$‖A_k^{(R)}‖^2$")

    savefig(p, "Figures/kicked_top_combined_inset.pdf")
    println("saved kicked_top_combined_inset.pdf")
end


function main()
    detail = read_csv_cols("kicked_top_near_zero_k_detail.csv")
    ref = read_csv_cols("kicked_top_exact_agp_reference.csv")
    exact_norm2 = Dict(zip(ref["k"], ref["exact_norm2"]))  # k => exact ‖A_k^(R)‖² at j=200
    make_convergence_vs_M_figure(detail, exact_norm2)
    make_Mconv_vs_k_figure(detail, exact_norm2)
    make_combined_inset_figure(detail, exact_norm2)
end

main()
