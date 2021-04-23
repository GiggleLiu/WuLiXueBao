struct P3{T}
    x::T
    y::T
    z::T
end

Base.zero(::Type{P3{T}}) where T = P3(zero(T), zero(T), zero(T))
Base.zero(::P3{T}) where T = P3(zero(T), zero(T), zero(T))
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
@inline function lorenz(t, y, θ)
    σ, ρ, β = θ
    P3(σ*(y.y-y.x), y.x*(ρ-y.z)-y.y, y.x*y.y-β*y.z)
end

function equilibrium(θ)
    σ, ρ, β = θ
    ρ < critical_ρ(σ, β)
end

function critical_ρ(σ, β)
    σ * (σ + β + 3)/(σ - β - 1)
end


# reversible implementation
@i @inline function :(+=)(identity)(Y::P3, X::P3)
    Y.x += X.x
    Y.y += X.y
    Y.z += X.z
end

@i @inline function :(+=)(*)(Y::P3, a::Real, X::P3)
    Y.x += a * X.x
    Y.y += a * X.y
    Y.z += a * X.z
end

@i @inline function :(+=)(/)(Y::P3, X::P3, b::Real)
    Y.x += X.x/b
    Y.y += X.y/b
    Y.z += X.z/b
end

@i function i_lorenz(y!::P3{T}, t, y::P3{T}, θ) where T
    @routine @invcheckoff begin
        @zeros T a b c b_a ab αc ac
        a += y.x
        b += y.y
        c += y.z
        b_a += b-a
        ab += a * b
        αc += θ.:3 * c
        c -= θ.:2
        ac += a * c
    end
    y!.x += θ.:1 * b_a
    y!.y -= ac + b
    y!.z += ab - αc
    ~@routine
end


@i function iloss!(out, f, history, y0, θ; ts)
    i_ODESolve(RK4(), (@const f), history, y0, θ; ts=ts)
    out += history[end].x
end