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
            push!(logger, :call, τ, δ, j)
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
    g = Lorentz.grad_func(f, s, q, g)
    push!(logger, :call, τ, δ, σ)
    push!(logger, :grad, τ, δ, σ)
    if σ>β
        # retrieve s
        s = pop!(state, β)
        push!(logger, :fetch, τ, δ, β)
    end
    return g
end