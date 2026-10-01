"""
generate_figures.jl

Makes both brickwork figures from the CSVs in this directory:

  Figures/AGP_norm_Brickwork.pdf        <- agp_norm_L12.csv
      Regularized AGP norm ‖A_θ^(R)‖² vs. θ (η_reg = 0.1), one line per
      M log-spaced from 2 to 50 (plasma on a log-M scale, dark = larger M,
      labelled in a legend), exact curve in solid red.

  Figures/AGP_Brickwork_Infidelity.pdf  <- infidelity_L12.csv
      Infidelity 1 − F vs. circuit depth S (log–log), one line per
      M = 10, 30, 50 (same colors and legend style), with the
      exact AGP in solid red and the uncorrected ramp in gray dashed.

Usage (from this directory):  julia generate_figures.jl
Requires: Plots, LaTeXStrings
"""

using DelimitedFiles
using Plots
using LaTeXStrings

function read_csv_cols(path)
    raw, header = readdlm(path, ',', header=true)
    return Dict(strip(String(name)) => Float64.(raw[:, i]) for (i, name) in enumerate(vec(header)))
end

function apply_figure_style()
    default(
        titlefontsize=18,
        guidefontsize=24,
        tickfontsize=22,
        legendfontsize=20,
        labelfontsize=22,
        frame=:box,
        grid=false,
        fontfamily="Computer Modern",
        legend=false,
        left_margin=3Plots.mm, bottom_margin=4Plots.mm
    )
end

# plasma with the pale-yellow end cut off, so every line stays visible on white;
# dark = larger M = more converged
const CMAP = cgrad([cgrad(:plasma)[x] for x in range(0.78, 0.0, length=64)])
const EXACT_COLOR = RGB(0.9, 0.05, 0.05)

# colors spaced evenly in log M
m_color(M, M_values) = CMAP[log(M / minimum(M_values)) / log(maximum(M_values) / minimum(M_values))]

# common look for both figures: large text and a two-column legend headed "M"
figure_kwargs() = (guidefontsize=30, tickfontsize=26,
                   legend=:topright, legendfontsize=17, legend_columns=2,
                   legend_title=L"M", legend_title_font_pointsize=19, size=(700, 500))

# legend entry as a square color swatch (an invisible NaN point carries the label)
legend_swatch!(p, color, label) = scatter!(p, [NaN], [NaN]; marker=:square, markersize=9,
                                           markercolor=color, markerstrokewidth=0, label=label)

function save_figure(p, outpath)
    mkpath(dirname(outpath))
    savefig(p, outpath)
    println("saved $outpath")
end

function make_agp_norm_figure(df; outpath)
    # log-spaced from 2 to 50, rounded to the even M values in the CSV
    M_values = unique(2 .* round.(Int, exp.(range(log(2), log(50), length=7)) ./ 2))
    apply_figure_style()
    default(bottom_margin=3Plots.mm)

    p = plot(; xlabel=L"$\theta$", ylabel=L"$‖A^{(R)}_\theta‖^2$",
             ylims=(0, 0.63), yticks=0.1:0.2:0.5, figure_kwargs()...)
    for M in M_values
        plot!(p, df["theta"], df["M=$M"], color=m_color(M, M_values), linewidth=3.2, label=false)
    end
    plot!(p, df["theta"], df["norm_exact_reg"], color=EXACT_COLOR, linewidth=4.0, label=false)

    for M in M_values
        legend_swatch!(p, m_color(M, M_values), string(M))
    end
    legend_swatch!(p, EXACT_COLOR, "Exact")

    save_figure(p, outpath)
end

function make_infidelity_figure(df; outpath)
    M_values = [10, 30, 50]
    S = df["S"]
    apply_figure_style()
    default(top_margin=4Plots.mm)   # room for the 10^0 tick label

    p = plot(; xlabel="Circuit Depth", ylabel=L"$1-\mathcal{F}$",
             xscale=:log2, yscale=:log10,
             xticks=(S, [latexstring("2^{$k}") for k in round.(Int, log2.(S))]),
             ylims=(5e-11, 1.0), yticks=10.0 .^ (-10:2:0),
             figure_kwargs()..., legend=:bottomleft, legend_columns=2)
    plot!(p, S, df["leak_uncorrected"], color=RGB(0.5, 0.5, 0.5), linewidth=3.2,
          linestyle=:dash, marker=:square, markersize=5, markerstrokewidth=0, label=false)
    for M in M_values
        plot!(p, S, df["M=$M"], color=m_color(M, M_values), linewidth=3.2,
              marker=:circle, markersize=6, markerstrokewidth=0, label=false)
    end
    plot!(p, S, df["leak_exact"], color=EXACT_COLOR, linewidth=4.0,
          marker=:diamond, markersize=7, markerstrokewidth=0, label=false)

    legend_swatch!(p, RGB(0.5, 0.5, 0.5), "Bare")
    for M in M_values
        legend_swatch!(p, m_color(M, M_values), string(M))
    end
    legend_swatch!(p, EXACT_COLOR, "Exact")

    save_figure(p, outpath)
end

make_agp_norm_figure(read_csv_cols(joinpath(@__DIR__, "agp_norm_L12.csv"));
                     outpath=joinpath(@__DIR__, "Figures", "AGP_norm_Brickwork.pdf"))
make_infidelity_figure(read_csv_cols(joinpath(@__DIR__, "infidelity_L12.csv"));
                       outpath=joinpath(@__DIR__, "Figures", "AGP_Brickwork_Infidelity.pdf"))
