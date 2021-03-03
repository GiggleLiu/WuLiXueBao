using NiLang.AD: GVar
include("julia.jl")
include("reversible_programming.jl")

PROG_COUNTER = Ref(0)   # (2k-1)^n
PEAK_MEM = Ref(0)    # n*(k-1)+2

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
        κ = max(σ+1, ϕ-1)
    end
    return κ
end

function treeverse!(f, s::T, state::Dict{Int,T}, g, δ, τ; N=binomial(τ+δ, τ)) where T
    if N > binomial(τ+δ, τ)
        error("please input a larger `τ` and `δ` so that `binomial(τ+δ, τ) >= N`!")
    end
    treeverse!(f, s, state, g, δ, τ, 0, 0, N)
end
function treeverse!(f, s::T, state::Dict{Int,T}, g, δ, τ, β, σ, ϕ) where T
    if σ > β
        δ -= 1
        # snapshot s
        state[β] = s
        PEAK_MEM[] = max(PEAK_MEM[], length(state))
        for j=β:σ-1
            s = f(s)
            PROG_COUNTER[] += 1
        end
    end

    κ = mid(δ, τ, σ, ϕ, δ)
    while τ>0 && κ < ϕ
        g = treeverse!(f, s, state, g, δ, τ, σ, κ, ϕ)
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
    PROG_COUNTER[] += 1
    if σ>β
        # retrieve s
        s = pop!(state, β)
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
    _, gs = (~i_step_fun)(
        (GVar(y[1], g[1]), P3(GVar(y[2].x, g[2].x), GVar(y[2].y, g[2].y), GVar(y[2].z,g[2].z))),
        (GVar(x[1]), GVar(x[2])))
    NiLang.AD.grad(gs)
end

using Test, ForwardDiff
@testset "treeverse gradient" begin
    x0 = P3(1.0, 0.0, 0.0)
    state = Dict{Int,Tuple{Float64,P3{Float64}}}()

    for N in [20, 120, 126]
        δ = 4
        N = 125
        τ = binomial_fit(N, δ)

        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
        g = (0.0, P3(1.0, 0.0, 0.0))
        g_tv = treeverse!(step_fun, (0.0, x0), state, g, δ, τ; N=N)
        @test g_fd ≈ [g_tv[2].x, g_tv[2].y, g_tv[2].z]
    end
end

x0 = P3(1.0, 0.0, 0.0)
state = Dict{Int,Tuple{Float64,P3{Float64}}}()

δ = 4
τ = 5
N = binomial(τ+δ, τ)

g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
g = (0.0, P3(1.0, 0.0, 0.0))
#treeverse!(step_fun, (0.0, x0), state, g, δ, τ; N=N)


