using BenchmarkTools
using ForwardDiff
using DelimitedFiles
using Test

include("point.jl")
include("julia.jl")
include("reversible_programming.jl")

function f_nilang(; y0 = P3(1.0, 0.0, 0.0), Nt=10000, Δt = 3e-3)
    rk4!(lorentz!, zeros(typeof(y0), Nt+1), y0, nothing; t0=0.0, Δt=3e-3, Nt=Nt)[2]
end

#using Plots
#plot(getfield.(history, :x), getfield.(history, :y), getfield.(history, :z))

function g_nilang(; y0 = P3(1.0, 0.0, 0.0), Nt=10000, Δt = 3e-3)
    NiLang.gradient(iloss!, (0.0, lorentz!, zeros(typeof(y0), Nt+1), y0, nothing); t0=0.0, Δt=Δt, Nt=Nt, iloss=1)[4];
end

function f_julia(; y0 = P3(1.0, 0.0, 0.0), Δt=3e-3, Nt=10000)
    rk4(lorentz, y0, (); t0=0.0, Δt=Δt, Nt=Nt)
end

function g_forwarddiff(; y0 = P3(1.0, 0.0, 0.0), Δt=3e-3, Nt=10000)
    ForwardDiff.gradient(x->rk4(lorentz, P3(x...), (); t0=0.0, Δt=Δt, Nt=Nt)[end].x, [y0.x, y0.y, y0.z])
end

# make sure they are computing the same thing
@testset "lorentz" begin
    history_julia = f_julia()
    history_nilang = f_nilang()
    gf = g_forwarddiff()
    gn = g_nilang()
    @test gn.x ≈ gf[1]
    @test gn.y ≈ gf[2]
    @test gn.z ≈ gf[3]
    @test history_julia[end].x ≈ history_nilang[end].x
    @test history_julia[end].y ≈ history_nilang[end].y
    @test history_julia[end].z ≈ history_nilang[end].z
end

function run_benchmarks(cases; output_file)
    suite = BenchmarkGroup()
    for (case, f) in cases
        suite[case] = @benchmarkable $f()
    end

    tune!(suite)
    res = run(suite)

    times = zeros(length(cases))
    for (k, (case, f)) in enumerate(cases)
        times[k] = minimum(res[case].times)
    end

    println("Writing benchmark results to file: $output_file.")
    mkpath(dirname(output_file))
    writedlm(output_file, times)
end

run_benchmarks(
    ["NiLang"=>f_nilang, "NiLang.AD"=>g_nilang, "Julia"=>f_julia, "ForwardDiff"=>g_forwarddiff];
    output_file = joinpath(@__DIR__, "data", "benchmark_lorentz.dat")
)