include("julia.jl")
include("reversible_programming.jl")

@i function lorentz_step!(y!::T, y::T; Δt) where T
    rk4_step!((@skip! lorentz!), y!, y, (@const nothing); Δt, t=0.0)
end

@i function bennett_loss(out, step, y, x; kwargs...)
    bennett((@skip! step), y, x; kwargs...)
    out += y.x
end

using Test, ForwardDiff
#=
@testset "bennett gradient" begin
    x0 = P3(1.0, 0.0, 0.0)
    state = Dict{Int,Tuple{Float64,P3{Float64}}}()

    for Nt in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])
        logger = NiLang.BennettLog()
        g_bn = NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=3, N=Nt, logger=logger)[4]
        @test g_fd ≈ [g_bn.x, g_bn.y, g_bn.z]
        @test length(logger.fcalls) > 0
    end
end
=#

using Compose, Viznet
function bennett_finger_printing(N::Int, k)
    x0 = P3(1.0, 0.0, 0.0)
    logger = BennettLog()
    #NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=k, N=N, logger=logger)[4]
    bennett_loss(0.0, lorentz_step!, zero(P3{Float64}), x0; Δt=3e-3, k=k, N=N, logger=logger)
    fcalls = logger.fcalls[1:length(logger.fcalls)*4÷7]

    eb1 = bondstyle(:line, linewidth(0.1mm), stroke("red"))
    eb2 = bondstyle(:line, linewidth(0.1mm), stroke("green"))
    Compose.set_default_graphic_size(15cm, 15cm)
    img = canvas() do
        for (depth, i, f) in fcalls
            if f isa Inv
                eb2 >> ((i-1.0, depth-0.0), (i*1.0, depth+0.0))
            else
                eb1 >> ((i-1.0, depth-0.0), (i*1.0, depth+0.0))
            end
        end
    end
    d = maximum(getindex.(fcalls, 1))
    Compose.compose(context(0.5/(N+1), 0.5/(1+d), 1/(N+1), 1/(1+d)), img)
end

#bennett_finger_printing(4^4, 4)
#x |> SVG(fname * ".svg")
#run(`rsvg-convert -f pdf -o $fname.pdf $fname.svg`)