using Test, ForwardDiff
using TreeverseAndBennett
using TreeverseAndBennett.Lorentz
using NiLang

@testset "treeverse gradient" begin
    x0 = P3(1.0, 0.0, 0.0)

    for N in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
        g = (0.0, P3(1.0, 0.0, 0.0))
        g_tv, log = treeverse(Lorentz.step_fun, Lorentz.grad_fun, (0.0, x0), g; δ=4, N=N)
        @test g_fd ≈ [g_tv[2].x, g_tv[2].y, g_tv[2].z]
    end
end

@testset "bennett gradient" begin
    x0 = P3(1.0, 0.0, 0.0)
    state = Dict{Int,Tuple{Float64,P3{Float64}}}()

    for Nt in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])
        logger = NiLang.BennettLog()
        g_bn = NiLang.AD.gradient(Lorentz.bennett_loss, (0.0, Lorentz.lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=3, N=Nt, logger=logger)[4]
        @test g_fd ≈ [g_bn.x, g_bn.y, g_bn.z]
        @test length(logger.fcalls) > 0
    end
end

# make sure they are computing the same thing
@testset "lorentz" begin
    history_julia = Lorentz.f_julia()
    history_nilang = Lorentz.f_nilang()
    gf = Lorentz.g_forwarddiff()
    gn = Lorentz.g_nilang()
    @test gn.x ≈ gf[1]
    @test gn.y ≈ gf[2]
    @test gn.z ≈ gf[3]
    @test history_julia[end].x ≈ history_nilang[end].x
    @test history_julia[end].y ≈ history_nilang[end].y
    @test history_julia[end].z ≈ history_nilang[end].z
end
