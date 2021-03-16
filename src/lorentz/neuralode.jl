function aug_dynamics(t, z::Glued, θ)
    x = z.data[1]
    y = lorentz(t, x, θ)
    a = z.data[2]
    _, _, r, _ = (~lorentz!)(P3(GVar(y.x, a.x), GVar(y.y, a.y), GVar(y.z, a.z)), t, GVar(x), nothing)
    Glued(y, P3(-r.x.g, -r.y.g, -r.z.g))
end

function force_gradient(t, x, θ, gy)
    a = lorentz(t, x, θ)
    _, _, r, _ = (~lorentz!)(P3(GVar(a.x, gy.x), GVar(a.y, gy.y), GVar(a.z, gy.z)), t, GVar(x), θ)
    a, P3(-r.x.g, -r.y.g, -r.z.g)
end

function error(Nt::Int; nrepeat=100)
    res = map(1:nrepeat) do i
        x0 = P3(randn(3)...)
        Δt=3e-3

        g_fd = ForwardDiff.gradient(x->ODESolve(RK4(), lorentz, P3(x...), nothing; ts=0.0:Δt:Δt*Nt).x, [x0.x, x0.y, x0.z])

        x1 = ODESolve(RK4(), lorentz, x0, nothing; ts=0.0:Δt:Δt*Nt)
        z0 = Glued(x1, P3(1.0, 0.0, 0.0))
        g_neural_ode = ODESolve(RK4(), aug_dynamics, z0, nothing; ts=Δt*Nt:-Δt:0.0).data[2]
        norm(g_fd .- [g_neural_ode.x, g_neural_ode.y, g_neural_ode.z])/norm(g_fd)
    end
    median(res)
end

function run_neuralode_checkpoint_errors(; output_file=nothing)
    nsteps = [1,20,50,100,150,200,250,300, 350, 400, 450, 500]
    x0 = [1.0, 0.0, 0.0]
    ts = 0.0:3e-3:30
    res = map(nsteps) do n
        gn = P3(1.0, 0.0, 0.0)
        g1 = checkpointed_neuralode(RK4(), lorentz, force_gradient, P3(x0...), gn, nothing; ts=ts, checkpoint_step=n)
        g_fd = ForwardDiff.gradient(x->ODESolve(RK4(), lorentz, P3(x...), nothing; ts=ts).x, x0)
        norm(g_fd .- [g1.x, g1.y, g1.z])/norm(g_fd)
    end
    out = hcat(nsteps, res)
    output_file !== nothing && writedlm(output_file, out)
    return out
end

function run_neuralode_errors(; output_file=nothing)
    xs = 1:1000
    errors = error.(xs; nrepeat=10)
    output_file !== nothing && writedlm(output_file, errors)
    return errors
end

