using NiLang.AD: GVar
include("julia.jl")
include("reversible_programming.jl")

struct TreeverseAction
    action::Symbol
    τ::Int
    δ::Int
    step::Int
    depth::Int
end

struct TreeverseLog
    actions::Vector{TreeverseAction}
    depth::Base.RefValue{Int}
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
end
TreeverseLog() = TreeverseLog(TreeverseAction[], Ref(0), Ref(0))
Base.push!(tlog::TreeverseLog, args...) = push!(tlog.actions, TreeverseAction(args..., tlog.depth[]))

function binomial_fit(N::Int, δ::Int)
    τ = 1
    while N > binomial(τ+δ, τ)
        τ += 1
    end
    return τ
end

function mid(δ, τ, σ, ϕ, d)
    κ = ceil(Int, (δ*σ + τ*ϕ)/(τ+δ))
    if κ >= ϕ && d > 0
        @show "@@@"
        κ = max(σ+1, ϕ-1)
    end
    return κ
end

function treeverse!(f, s::T, g; δ, N, τ=binomial_fit(N,δ)) where T
    state = Dict{Int,typeof(s)}()
    if N > binomial(τ+δ, τ)
        error("please input a larger `τ` and `δ` so that `binomial(τ+δ, τ) >= N`!")
    end
    logger = TreeverseLog()
    g = treeverse!(f, s, state, g, δ, τ, 0, 0, N, logger)
    return g, logger
end
function treeverse!(f, s::T, state::Dict{Int,T}, g, δ, τ, β, σ, ϕ, logger) where T
    logger.depth[] += 1
    if σ > β
        δ -= 1
        # snapshot s
        state[β] = s
        push!(logger, :store, τ, δ, β)
        logger.peak_mem[] = max(logger.peak_mem[], length(state))
        for j=β:σ-1
            s = f(s)
            push!(logger, :call, τ, δ, j+1)
        end
    end

    κ = mid(δ, τ, σ, ϕ, δ)
    while τ>0 && κ < ϕ
        g = treeverse!(f, s, state, g, δ, τ, σ, κ, ϕ, logger)
        τ -= 1
        ϕ = κ
        κ = mid(δ, τ, σ, ϕ, δ)
    end

    if ϕ-σ != 1
        error("treeverse fails!")
    end
    q = s
    s = f(s)
    g = grad_func(f, s, q, g)
    push!(logger, :call, τ, δ, ϕ)
    push!(logger, :grad, τ, δ, ϕ)
    if σ>β
        # retrieve s
        s = pop!(state, β)
        push!(logger, :fetch, τ, δ, β)
    end
    return g
end

@i function i_step_fun(state2, state)
    rk4_step!((@const lorentz!), (state2 |> tget(2)), (state |> tget(2)), (); Δt=3e-3, t=state[1])
    (state2 |> tget(1)) += (state |> tget(1)) + 3e-3
end

function step_fun(x)
    i_step_fun((0.0, zero(x[2])), x)[1]
end

function grad_func(::typeof(step_fun), y, x, g)
    _, gs = (~i_step_fun)(GVar(y, g), GVar(x))
    NiLang.AD.grad(gs)
end

using Test, ForwardDiff
@testset "treeverse gradient" begin
    x0 = P3(1.0, 0.0, 0.0)

    for N in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
        g = (0.0, P3(1.0, 0.0, 0.0))
        g_tv, log = treeverse!(step_fun, (0.0, x0), g; δ=4, N=N)
        @test g_fd ≈ [g_tv[2].x, g_tv[2].y, g_tv[2].z]
    end
end