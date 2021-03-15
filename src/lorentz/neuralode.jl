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

function aug_dynamics(t, z::Glued, θ)
    x = z.data[1]
    y = lorentz(t, x, θ)
    a = z.data[2]
    _, _, r, _ = (~lorentz!)(P3(GVar(y.x, a.x), GVar(y.y, a.y), GVar(y.z, a.z)), t, GVar(x), nothing)
    Glued(y, P3(-r.x.g, -r.y.g, -r.z.g))
end

function error(Nt::Int; nrepeat=100)
    res = map(1:nrepeat) do i
        x0 = P3(randn(3)...)
        Δt=3e-3

        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=Δt, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])

        x1 = rk4(lorentz, x0, nothing; t0=0.0, Δt=Δt, Nt=Nt)[end]
        z0 = Glued(x1, P3(1.0, 0.0, 0.0))
        g_neural_ode = rk4(aug_dynamics, z0, nothing; t0=Δt*Nt, Δt=-Δt, Nt=Nt)[end].data[2]
        norm(g_fd .- [g_neural_ode.x, g_neural_ode.y, g_neural_ode.z])/norm(g_fd)
    end
    median(res)
end

function checkpointed_neuralode(; checkpoint_step=200)
    Nt = 10000
    ncheckpoint = ceil(Int, Nt / checkpoint_step)

    Δt=3e-3
    # compute checkpoints
    x = P3(1.0, 0.0, 0.0)
    checkpoints = zeros(typeof(x), ncheckpoint)
    for i=1:ncheckpoint
        t0 = Δt*(i-1)*checkpoint_step
        nstep = min(Nt-(i-1)*checkpoint_step, checkpoint_step)
        x = rk4(lorentz, x, nothing; t0=t0, Δt=Δt, Nt=nstep)[end]
        checkpoints[i] = x
    end
    a = P3(1.0, 0.0, 0.0)
    local z
    for i=ncheckpoint:-1:1
        x = checkpoints[i]
        if i==ncheckpoint
            z = Glued(x, a)
        else
            z = Glued(x, z.data[2])
        end
        nstep = min(Nt-(i-1)*checkpoint_step, checkpoint_step)
        z = rk4(aug_dynamics, z, nothing; t0=Δt*i*checkpoint_step, Δt=-Δt, Nt=nstep)[end]
    end
    z.data[2]
end

function run_checkpoint_errors(; output_file=joinpath(pwd(), "neuralode_checkpoint.dat"))
    nsteps = [1,20,50,100,150,200,250,300, 350, 400, 450, 500]
    res = map(nsteps) do n
        g1 = checkpointed_neuralode(checkpoint_step=n)
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=10000)[end].x, [1.0, 0.0, 0.0])
        norm(g_fd .- [g1.x, g1.y, g1.z])/norm(g_fd)
    end
    writedlm(output_file, hcat(nsteps, res))
end

function dumperrors(; output_file=joinpath(pwd(), "errors.dat"))
    xs = 1:1000
    errors = error.(xs; nrepeat=10)
    writedlm(output_file, errors)
end

