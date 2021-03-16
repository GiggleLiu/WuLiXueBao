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