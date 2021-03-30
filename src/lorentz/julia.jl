@inline function Base.:(+)(a::P3, b::P3)
    P3(a.x + b.x, a.y + b.y, a.z + b.z)
end

@inline function Base.:(/)(a::P3, b::Real)
    P3(a.x/b, a.y/b, a.z/b)
end

@inline function Base.:(*)(a::Real, b::P3)
    P3(a*b.x, a*b.y, a*b.z)
end

# test case, θ = (10, 27, 8/3)
function lorentz(t, y, θ)
    σ, ρ, β = θ
    P3(σ*(y.y-y.x), y.x*(ρ-y.z)-y.y, y.x*y.y-β*y.z)
end