using ReversibleSeismic, Test, ForwardDiff
using NiLang
using NiLang.AD: GVar

function force_gradient(t, x, θ, gy)
    a = lorentz(t, x, θ)
    _, _, r, _ = (~lorentz!)(P3(GVar(a.x, gy.x), GVar(a.y, gy.y), GVar(a.z, gy.z)), t, GVar(x), θ)
    a, P3(-r.x.g, -r.y.g, -r.z.g)
end

