function f_nilang(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Nt=10000, Δt = 3e-3)
    i_ODESolve(RK4(), lorentz!, zeros(typeof(y0), Nt+1), y0, θ; ts=0.0:Δt:Δt*Nt)[3]
end

function g_nilang(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Nt=10000, Δt = 3e-3)
    NiLang.gradient(iloss!, (0.0, lorentz!, zeros(typeof(y0), Nt+1), y0, θ); ts=0.0:Δt:Δt*Nt, iloss=1)[4];
end

function f_julia(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Δt=3e-3, Nt=10000, logger=nothing)
    ODESolve(RK4(), lorentz, y0, θ; ts=0.0:Δt:Nt*Δt, logger=logger)
end

function g_forwarddiff(; y0 = P3(1.0, 0.0, 0.0), θ=(10.0, 27.0, 8/3), Δt=3e-3, Nt=10000)
    ForwardDiff.gradient(x->ODESolve(RK4(), lorentz, P3(x...), θ; ts=ts=0.0:Δt:Nt*Δt).x, [y0.x, y0.y, y0.z])
end

export run_lorentz_benchmarks

function run_lorentz_benchmarks(; output_file = joinpath(pwd(), "benchmark_lorentz.dat"))
    run_benchmarks(
        ["NiLang"=>f_nilang, "NiLang.AD"=>g_nilang, "Julia"=>f_julia, "ForwardDiff"=>g_forwarddiff];
        output_file = output_file
    )
end