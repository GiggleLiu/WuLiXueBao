function f_nilang(; y0 = P3(1.0, 0.0, 0.0), Nt=10000, Δt = 3e-3)
    rk4!(lorentz!, zeros(typeof(y0), Nt+1), y0, nothing; t0=0.0, Δt=3e-3, Nt=Nt)[2]
end

function g_nilang(; y0 = P3(1.0, 0.0, 0.0), Nt=10000, Δt = 3e-3)
    NiLang.gradient(iloss!, (0.0, lorentz!, zeros(typeof(y0), Nt+1), y0, nothing); t0=0.0, Δt=Δt, Nt=Nt, iloss=1)[4];
end

function f_julia(; y0 = P3(1.0, 0.0, 0.0), Δt=3e-3, Nt=10000)
    rk4(lorentz, y0, (); t0=0.0, Δt=Δt, Nt=Nt)
end

function g_forwarddiff(; y0 = P3(1.0, 0.0, 0.0), Δt=3e-3, Nt=10000)
    ForwardDiff.gradient(x->rk4(lorentz, P3(x...), (); t0=0.0, Δt=Δt, Nt=Nt)[end].x, [y0.x, y0.y, y0.z])
end

export run_lorentz_benchmarks

function run_lorentz_benchmarks(; output_file = joinpath(pwd(), "benchmark_lorentz.dat"))
    run_benchmarks(
        ["NiLang"=>f_nilang, "NiLang.AD"=>g_nilang, "Julia"=>f_julia, "ForwardDiff"=>g_forwarddiff];
        output_file = output_file
    )
end