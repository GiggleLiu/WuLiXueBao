function build_aug_dynamics(ag)
    function aug_dynamics(t, z::Glued, θ)
        y = z.data[1]
        gy = z.data[2]
        gθ = z.data[3]
        a, ax, aθ = ag(t, y, θ, gy)
        Glued(a, ax, aθ)
    end
end

function checkpointed_neuralode(solver, f, ag, x0::T, gn::TN, θ, gθ::TT; ts, checkpoint_step) where {T,TN,TT}
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
    z = Glued{Tuple{T,TN,TT}}((checkpoints[end], gn, gθ))
    for i=ncheckpoint:-1:1
        if i!=ncheckpoint
            x = checkpoints[i]
            z = Glued{Tuple{T,TN,TT}}((x, z.data[2], z.data[3]))
        end
        tsi = ts[(i-1)*checkpoint_step+1:min(i*checkpoint_step, N)+1]
        z = ODESolve(solver, build_aug_dynamics(ag), z, θ; ts=Iterators.reverse(tsi))
    end
    z.data[2], z.data[3]
end

struct ODELog{T}
    history::Vector{T}
end

logstate!(logger::ODELog, x) = push!(logger.history, x)
logstate!(::Nothing, x) = nothing

function ODESolve(solver, f, y0::T, θ; ts, logger=nothing) where T
    logstate!(logger, y0)
    y = y0
    for i=1:length(ts)-1
        y = ODEStep(solver, f, ts[i], y, θ; Δt=ts[i+1]-ts[i])
        logstate!(logger, y)
    end
    return y
end

@i function i_ODESolve(solver, f, history, y0::T, θ; ts) where T
    history[1] += y0
    @invcheckoff @inbounds for i=1:length(ts)-1
        i_ODEStep(solver, f, history[i+1], history[i], θ; Δt=ts[i+1]-ts[i], t=ts[i])
    end
end

# RK4
struct RK4
end

function ODEStep(::RK4, f, t, y, θ; Δt)
    k1 = Δt * f(t, y, θ)
    k2 = Δt * f(t+Δt/2, y + k1 / 2, θ)
    k3 = Δt * f(t+Δt/2, y + k2 / 2, θ)
    k4 = Δt * f(t+Δt, y + k3, θ)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

@i function i_ODEStep(solver::RK4, f, y!::T, y::T, θ; Δt, t) where T
    @routine @invcheckoff begin
        @zeros T k1 k2 k3 k4 o1 o2 o3 o4 yk1 yk2 yk3
        f(o1, t, y, θ)
        k1 += Δt * o1
        yk1 += y
        yk1 += k1 / 2
        t += Δt/2
        f(o2, t, yk1, θ)
        k2 += Δt * o2
        yk2 += y
        yk2 += k2 / 2
        f(o3, t, yk2, θ)
        k3 += Δt * o3
        yk3 += y
        yk3 += k3
        t += Δt/2
        f(o4, t, yk3, θ)
        k4 += Δt * o4
    end
    y! += y
    y! += k1 / 6
    y! += k2 / 3
    y! += k3 / 3
    y! += k4 / 6
    ~@routine
end
