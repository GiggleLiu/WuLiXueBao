function f_nilang(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Nt=10000, Δt = 3e-3)
    i_ODESolve(RK4(), lorenz!, zeros(typeof(y0), Nt+1), y0, θ; ts=0.0:Δt:Δt*Nt)[3]
end

function g_nilang(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Nt=10000, Δt = 3e-3)
    NiLang.gradient(iloss!, (0.0, lorenz!, zeros(typeof(y0), Nt+1), y0, θ); ts=0.0:Δt:Δt*Nt, iloss=1)[[4,5]];
end

function f_julia(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Δt=3e-3, Nt=10000, logger=nothing)
    ODESolve(RK4(), lorenz, y0, θ; ts=0.0:Δt:Nt*Δt, logger=logger)
end

function g_forwarddiff(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Δt=3e-3, Nt=10000)
    ForwardDiff.gradient(x->ODESolve(RK4(), lorenz, P3(x[1:3]...), (x[4:6]...,); ts=ts=0.0:Δt:Nt*Δt).x, [y0.x, y0.y, y0.z, θ...])
end

export run_lorenz_benchmarks

function run_lorenz_benchmarks(; output_file = joinpath(pwd(), "benchmark_lorenz.dat"))
    run_benchmarks(
        ["NiLang"=>f_nilang, "NiLang.AD"=>g_nilang, "Julia"=>f_julia, "ForwardDiff"=>g_forwarddiff];
        output_file = output_file
    )
end

using Plots
function run_lorenz_phase(σs, ρs, βs; fname=joinpath(dirname(dirname(@__DIR__)), "paper/lorenz_grad"))
    mean_abs_grads = [sum(abs, g_forwarddiff(;θ=(σ,ρ,β)))/6 for σ in σs, ρ in ρs, β in βs]
    curve = [critical_ρ(σ, β) for σ in σs, β in βs]
    if fname !== nothing
        writedlm(fname*"_heatmap.dat", mean_abs_grads)
        writedlm(fname*"_curve.dat", curve)
    end
    return mean_abs_grads, curve
end

function plot_lorenz_grad(; fname=joinpath(dirname(dirname(@__DIR__)), "paper/lorenz_grad"))
    #mg, curve = run_lorenz_phase(σs, ρs, β)
    mg = readdlm(fname*"_heatmap.dat")
    curve = readdlm(fname*"_curve.dat")
    σs = LinRange(0,20,size(mg, 1))
    ρs = LinRange(0,50,size(mg, 2))
    _plot_lorenz_grad(σs,ρs,mg,curve,fname)
end

function _plot_lorenz_grad(σs,ρs,mg,curve, fname)
    plt = heatmap(σs, ρs, log10.(mg)'; ylims=(0,50), xlims=(0,20), label="log mean gradient")
    @show length(curve)
    @show size(mg)
    curve[σs .< 4] .= 51
    plt = plot!(σs, curve, color="black", lw=2, label="theoretical", xtickfontsize=14, ytickfontsize=14, legendfontsize=14, xlabel="")
    #savefig(fname*".png")
    plt
end
