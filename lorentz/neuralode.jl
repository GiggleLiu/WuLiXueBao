using ForwardDiff
using ForwardDiff: Dual
using NiLang.AD: GVar

include("julia.jl")
include("reversible_programming.jl")

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

using Test
function aug_dynamics(t, z::Glued, θ)
    x = z.data[1]
    y = lorentz(t, x, θ)
    a = z.data[2]
    _, _, r, _ = (~lorentz!)(P3(GVar(y.x, a.x), GVar(y.y, a.y), GVar(y.z, a.z)), t, GVar(x), nothing)
    Glued(y, P3(-r.x.g, -r.y.g, -r.z.g))
end

using Statistics

function error(Nt::Int; nrepeat=100)
    res = map(1:nrepeat) do i
        x0 = P3(randn(3)...)
        Δt=3e-3

        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=Δt, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])

        x1 = rk4(lorentz, x0, nothing; t0=0.0, Δt=Δt, Nt=Nt)[end]
        z0 = Glued(x1, P3(1.0, 0.0, 0.0))
        g_neural_ode = rk4(aug_dynamics, z0, nothing; t0=Δt*Nt, Δt=-Δt, Nt=Nt)[end].data[2]
        sqrt(sum(abs2, g_fd .- [g_neural_ode.x, g_neural_ode.y, g_neural_ode.z]))
    end
    median(res)
end

xs = 1:1000
errors = error.(xs; nrepeat=10)
using Plots
plot(xs, errors; yscale=:log10)