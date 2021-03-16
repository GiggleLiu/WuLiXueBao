export Glued, RK4, ODESolve, ODEStep

struct Glued{T<:Tuple}
    data::T
end
Glued(args...) = Glued(args)

Base.zero(c::Glued) = Glued(zero.(c.data))
@generated function Base.zero(::Type{Glued{T}}) where T
    :(Glued($([zero(t) for t in T.types]...)))
end

@inline function Base.:(+)(a::Glued, b::Glued)
    Glued(a.data .+ b.data)
end

@inline function Base.:(/)(a::Glued, b::Real)
    Glued(a.data ./ b)
end

@inline function Base.:(*)(a::Real, b::Glued)
    Glued(a .* b.data)
end

function build_aug_dynamics(ag)
    function aug_dynamics(t, z::Glued, θ)
        y = z.data[1]
        gy = z.data[2]
        a, gx = ag(t, y, θ, gy)
        Glued(a, gx)
    end
end

function checkpointed_neuralode(solver, f, ag, x0::T, gn, θ; ts, checkpoint_step) where T
    N = length(ts)- 1
    ncheckpoint = ceil(Int, N / checkpoint_step)
    # compute checkpoints
    checkpoints = zeros(T, ncheckpoint)
    x = x0
    for i=1:ncheckpoint
        tsi = ts[(i-1)*checkpoint_step+1:min(i*checkpoint_step, N)+1]
        x = ODESolve(solver, f, x, θ; ts=tsi, logger=nothing)
        checkpoints[i] = x
    end
    local z
    for i=ncheckpoint:-1:1
        x = checkpoints[i]
        if i==ncheckpoint
            z = Glued(x, gn)
        else
            z = Glued(x, z.data[2])
        end
        tsi = ts[(i-1)*checkpoint_step+1:min(i*checkpoint_step, N)+1]
        z = ODESolve(solver, build_aug_dynamics(ag), z, nothing; ts=Iterators.reverse(tsi))
    end
    z.data[2]
end

export ODELog
struct ODELog{T}
    history::Vector{T}
end

logstate!(logger::ODELog, x) = push!(logger.history, x)
logstate!(::Nothing, x) = nothing

struct RK4
end

function ODESolve(solver, f, y0::T, θ; ts, logger=nothing) where T
    logstate!(logger, y0)
    y = y0
    for i=1:length(ts)-1
        y = ODEStep(solver, f, ts[i], y, θ; Δt=ts[i+1]-ts[i])
        logstate!(logger, y)
    end
    return y
end

# RK4
function ODEStep(::RK4, f, t, y, θ; Δt)
    k1 = Δt * f(t, y, θ)
    k2 = Δt * f(t+Δt/2, y + k1 / 2, θ)
    k3 = Δt * f(t+Δt/2, y + k2 / 2, θ)
    k4 = Δt * f(t+Δt, y + k3, θ)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end