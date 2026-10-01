"""
generate_kicked_top_figures_split.jl

Chaos-indicator figures vs. kick strength k.

Outputs (in Figures/, which must exist):

  1. kicked_top_agp_exponent.pdf        -- AGP-norm finite-size scaling
                                           exponent alpha(k) (‖A_k‖² ~ j^alpha,
                                           fit over j >= 50), with an inset of
                                           ‖A_k‖² vs. j at k = 0.5 / 2.5 / 6.0
  2. kicked_top_lyapunov.pdf            -- classical Lyapunov exponent lambda(k)
                                           (Benettin two-trajectory method)
  3. kicked_top_level_spacing_ratio.pdf -- mean level-spacing ratio r_mean(k),
                                           referenced against COE (0.5307)
                                           and Poisson (0.3863) limits

Required input data files (current directory), all from core.jl:
  - kicked_top_near_zero_k_summary.csv   (converged AGP norm^2 per (j,k); run_agp_sweep)
  - kicked_top_lyapunov_sweep.csv        (Lyapunov exponent vs k; run_lyapunov_sweep)
  - level_stats_sweep.csv                (mean level-spacing ratio vs k; run_level_stats_sweep)

Requires: Plots, LaTeXStrings, DelimitedFiles
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
        guidefontsize=20,
        tickfontsize=18,
        legendfontsize=15,
        bottom_margin=4Plots.mm,
        left_margin=2Plots.mm,
        frame=:box,
        grid=false,
        fontfamily="Computer Modern",
        # size=(500, 360),
        # dpi=300,
        legend=false,
    )
end


"""
    raw_scaling_curve(df_nz, k_target) -> (j, norm2)

Raw (unfitted) squared AGP norm ‖A_k‖² = norm2_cmv vs. system size j, for the
single k value matching k_target on the near-zero-k grid, sorted by j.
"""
function raw_scaling_curve(df_nz, k_target)
    k_col, j_col, norm2_col = df_nz["k"], df_nz["j"], df_nz["norm2_cmv"]
    mask = isapprox.(k_col, k_target; atol=1e-9)
    j = j_col[mask]
    norm2 = norm2_col[mask]
    order = sortperm(j)
    return j[order], norm2[order]
end

"""
    add_agp_norm_inset!(p, df_nz)

Adds a log-log inset to plot `p` showing the raw AGP-norm finite-size
scaling curves ‖A_k‖² vs. j at three representative k values (integrable
k=0.5, transition k=2.5, chaotic k=6.0) over the full system-size range
(j = 10..200). Their log-log slope over j >= 50 gives alpha(k) in the
main panel.
"""
function add_agp_norm_inset!(p, df_nz)
    k_reps = [(0.5, "#3A6FB3", L"0.5"), (2.5, "#D08A2B", L"2.5"), (6.0, "#6A3D9A", L"6.0")]

    # inset box, as fractions of the main axes: x ∈ [0.565, 0.885], y ∈ [0.16, 0.48]
    plot!(p, inset=(1, bbox(0.115, 0.16, 0.32, 0.32, :bottom, :right)))
    ip = p[2]
    for (k, color, _) in k_reps
        j, norm2 = raw_scaling_curve(df_nz, k)
        plot!(ip, j, norm2, xscale=:log10, yscale=:log10,
              markershape=:circle, markersize=3, markerstrokewidth=0,
              linewidth=1.6, color=color, label=false,
              guidefontsize=14, tickfontsize=12,
              framestyle=:box, grid=false,
              xticks=([10, 100], [L"10^1", L"10^2"]),
              yticks=([1, 1e2, 1e4], [L"10^0", L"10^2", L"10^4"]))
    end
    xlims!(ip, 8, 260)
    xlabel!(ip, L"$j$")
    ylabel!(ip, L"‖A_k‖^2")

    # Vertical key to the right of the inset box:  k / ■ 6.0 / ■ 2.5 / ■ 0.5
    # (top-to-bottom in the same order as the curves), drawn in the main
    # axes' data coordinates, so the main limits are pinned.
    mp = p[1]
    (x0, x1), (y0, y1) = Plots.xlims(mp), Plots.ylims(mp)
    fx(f) = x0 + f * (x1 - x0)
    fy(f) = y0 + f * (y1 - y0)
    annotate!(mp, fx(0.935), fy(0.46), text(L"k", :black, :center, 14))
    for (i, (_, color, label)) in enumerate(reverse(k_reps))
        yk = fy(0.46 - 0.095 * i)
        scatter!(mp, [fx(0.907)], [yk], markershape=:rect, markersize=6,
                 markercolor=color, markerstrokewidth=0, label=false)
        annotate!(mp, fx(0.925), yk, text(label, :black, :left, 13))
    end
    xlims!(mp, x0, x1)
    ylims!(mp, y0, y1)
    return p
end


function compute_agp_exponents(df_nz)
    k_col, j_col, norm2_col = df_nz["k"], df_nz["j"], df_nz["norm2_cmv"]
    exponents = Dict{Float64,Float64}()
    for k in sort(unique(k_col))
        mask = k_col .== k
        j = j_col[mask]
        norm2 = norm2_col[mask]
        order = sortperm(j)
        j, norm2 = j[order], norm2[order]
        length(j) < 3 && continue
        fit_mask = j .>= 50
        sum(fit_mask) < 2 && continue
        logj, lognorm = log.(j[fit_mask]), log.(norm2[fit_mask])
        n = length(logj)
        xbar, ybar = sum(logj) / n, sum(lognorm) / n
        slope = sum((logj .- xbar) .* (lognorm .- ybar)) / sum((logj .- xbar) .^ 2)
        exponents[k] = slope
    end
    k_vals = sort(collect(keys(exponents)))
    exp_vals = [exponents[k] for k in k_vals]
    return k_vals, exp_vals
end


# ─────────────────────────────────────────────────────────────────────────
# Panel 1: AGP-norm scaling exponent alpha(k)
# ─────────────────────────────────────────────────────────────────────────
function make_agp_exponent_figure(df_nz)
    k_vals, exp_vals = compute_agp_exponents(df_nz)

    apply_figure_style()
    p = scatter(k_vals, exp_vals, markershape=:circle, markersize=5,
                markercolor="#B33A3A", markerstrokewidth=0, label=false)
    #hline!(p, [1.0], color=RGB(0.6, 0.6, 0.6), linestyle=:dot, linewidth=1.5, label=false)
    xlims!(p, -0.3, 8.5)
    xlabel!(p, L"kick strength $k$")
    ylabel!(p, L"$\alpha \;(‖A_k‖^2 \sim j^{\alpha})$")
    vline!(p, [1.8], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1.5, label=false)
    #title!(p, "AGP-norm finite-size scaling exponent")

    # Inset: raw ‖A_k‖² vs. j (log-log) at k = 0.5 / 2.5 / 6.0 — the finite-size
    # scaling curves whose fitted slope produces alpha(k) in the main panel.
    add_agp_norm_inset!(p, df_nz)

    savefig(p, "Figures/kicked_top_agp_exponent.pdf")
    println("saved kicked_top_agp_exponent.pdf")
end

# ─────────────────────────────────────────────────────────────────────────
# Panel 2: classical Lyapunov exponent lambda(k)
# ─────────────────────────────────────────────────────────────────────────
function make_lyapunov_figure(lyap)
    apply_figure_style()
    p = scatter(lyap["k"], lyap["lyapunov"], markershape=:utriangle, markersize=5,
                markercolor="#2E8B57", markerstrokewidth=0, label=false)
    hline!(p, [0.0], color=RGB(0.6, 0.6, 0.6), linestyle=:dot, linewidth=1.5, label=false)
    xlims!(p, -0.3, 8.5)
    xlabel!(p, L"kick strength $k$")
    ylabel!(p, L"$\lambda$")
    #title!(p, "Classical Lyapunov exponent")
    vline!(p, [1.8], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1.5, label=false)
    savefig(p, "Figures/kicked_top_lyapunov.pdf")
    println("saved kicked_top_lyapunov.pdf")
end


# ─────────────────────────────────────────────────────────────────────────
# Panel 3: mean level-spacing ratio (k)
# ─────────────────────────────────────────────────────────────────────────
function make_level_spacing_ratio_figure(lev)
    mask = (lev["k"] .> 0) .& (lev["k"] .<= 8.5)
    k_sub, r_mean_sub, r_std_sub = lev["k"][mask], lev["r_mean"][mask], lev["r_std"][mask]

    apply_figure_style()
    p = scatter(k_sub, r_mean_sub, yerror=r_std_sub, markershape=:square, markersize=5,
                markercolor="#3A6FB3", markerstrokewidth=0, linewidth=1.6,
                label=false, legend=:bottomright)
    hline!(p, [0.5307], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1.5, label="COE")
    hline!(p, [0.3863], color=RGB(0.6, 0.6, 0.6), linestyle=:dot, linewidth=1.5, label="Poisson")
    xlims!(p, -0.3, 8.5)
    xlabel!(p, L"kick strength $k$")
    ylabel!(p, L"$\langle \tilde{r} \rangle$ ")
    vline!(p, [1.8], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1.5, label=false)
    #title!(p, "Level-spacing statistics")
    savefig(p, "Figures/kicked_top_level_spacing_ratio.pdf")
    println("saved kicked_top_level_spacing_ratio.pdf")
end


function main()
    df_nz = read_csv_cols("kicked_top_near_zero_k_summary.csv")
    lyap = read_csv_cols("kicked_top_lyapunov_sweep.csv")
    lev = read_csv_cols("level_stats_sweep.csv")

    make_agp_exponent_figure(df_nz)
    make_lyapunov_figure(lyap)
    make_level_spacing_ratio_figure(lev)
end

main()
