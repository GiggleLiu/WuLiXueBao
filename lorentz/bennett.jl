y0 = P3(1.0, 0.0, 0.0)

@i function lorentz_step!(y!::T, y::T; Δt) where T
    rk4_step!((@skip! lorentz!), y!, y, (@const nothing); Δt, t=0.0)
end

@i function bennett_loss(out, step, y, x; kwargs...)
    bennett((@skip! step), y, x; kwargs...)
    out += y.x
end

using Test, ForwardDiff
@testset "bennett gradient" begin
    x0 = P3(1.0, 0.0, 0.0)
    state = Dict{Int,Tuple{Float64,P3{Float64}}}()

    for Nt in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])
        g_bn = NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=3, nsteps=Nt)[4]
        @test g_fd ≈ [g_bn.x, g_bn.y, g_bn.z]
    end
end