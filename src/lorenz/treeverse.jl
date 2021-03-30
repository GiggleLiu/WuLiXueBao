@i function i_step_fun(state2, state, θ)
    i_ODEStep((@const RK4()), (@const lorenz!), (state2 |> tget(2)), (state |> tget(2)), θ; Δt=3e-3, t=state[1])
    (state2 |> tget(1)) += (state |> tget(1)) + 3e-3
end

function step_fun(x, θ)
    i_step_fun((0.0, zero(x[2])), x, θ)[1]
end

# g is (g_state, g_θ)
function grad_fun(x, g, θ)
    y = step_fun(x, θ)
    _, gs, gθ = (~i_step_fun)(GVar(y, g[1]), GVar(x), GVar(θ, g[2]))
    NiLang.AD.grad(gs), NiLang.AD.grad(gθ)
end
