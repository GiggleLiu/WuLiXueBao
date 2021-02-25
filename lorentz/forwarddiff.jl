using ForwardDiff

include("point.jl")

function Base.:(+)(a::P3, b::P3)
    P3(a.x + b.x, a.y + b.y, a.z + b.z)
end

function Base.:(/)(a::P3, b::Real)
    P3(a.x/b, a.y/b, a.z/b)
end

function Base.:(*)(a::Real, b::P3)
    P3(a*b.x, a*b.y, a*b.z)
end


function lorentz(t, y)
    P3(10*(y.y-y.x), y.x*(27-y.z)-y.y, y.x*y.y-8/3*y.z)
end

function rk4_step(f, t, y; Δt)
    k1 = Δt * f(t, y)
    k2 = Δt * f(t+Δt/2, y + k1 / 2)
    k3 = Δt * f(t+Δt/2, y + k2 / 2)
    k4 = Δt * f(t+Δt, y + k3)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

function rk4(f, y0::T; Δt, Nt) where T
    history = zeros(T, Nt+1)
    history[1] = y0
    y = y0
    for i=1:Nt
        y = rk4_step(f, (i-1)*Δt, y; Δt=Δt)
        history[i+1] = y
    end
    return history
end

y0 = P3(1.0, 0.0, 0.0)
@time history = rk4(lorentz, y0; Δt=3e-3, Nt=10000)
@time g = ForwardDiff.gradient(x->rk4(lorentz, P3(x...); Δt=3e-3, Nt=10000)[end].x, [y0.x, y0.y, y0.z])

using Plots
plot(getfield.(history[:], :x), getfield.(history[:], :y), getfield.(history[:], :z))


using ReverseDiff
@time g = ReverseDiff.gradient(x->rk4(lorentz, P3(x...); Δt=3e-3, Nt=10000)[end].x, [y0.x, y0.y, y0.z])