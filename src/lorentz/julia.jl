@inline function Base.:(+)(a::P3, b::P3)
    P3(a.x + b.x, a.y + b.y, a.z + b.z)
end

@inline function Base.:(/)(a::P3, b::Real)
    P3(a.x/b, a.y/b, a.z/b)
end

@inline function Base.:(*)(a::Real, b::P3)
    P3(a*b.x, a*b.y, a*b.z)
end


function lorentz(t, y, θ)
    P3(10*(y.y-y.x), y.x*(27-y.z)-y.y, y.x*y.y-8/3*y.z)
end

function rk4_step(f, t, y, θ; Δt)
    k1 = Δt * f(t, y, θ)
    k2 = Δt * f(t+Δt/2, y + k1 / 2, θ)
    k3 = Δt * f(t+Δt/2, y + k2 / 2, θ)
    k4 = Δt * f(t+Δt, y + k3, θ)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

function rk4(f, y0::T, θ; t0, Δt, Nt) where T
    history = zeros(T, Nt+1)
    history[1] = y0
    y = y0
    for i=1:Nt
        y = rk4_step(f, t0+(i-1)*Δt, y, θ; Δt=Δt)
        @inbounds history[i+1] = y
    end
    return history
end