@i function i_step_fun(state2, state)
    rk4_step!((@const lorentz!), (state2 |> tget(2)), (state |> tget(2)), (); Δt=3e-3, t=state[1])
    (state2 |> tget(1)) += (state |> tget(1)) + 3e-3
end

function step_fun(x)
    i_step_fun((0.0, zero(x[2])), x)[1]
end

function grad_fun(y, x, g)
    _, gs = (~i_step_fun)(GVar(y, g), GVar(x))
    NiLang.AD.grad(gs)
end