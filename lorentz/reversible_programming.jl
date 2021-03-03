using NiLang

include("point.jl")

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

@i function lorentz!(y!::P3{T}, t, y::P3{T}, θ) where T
    @routine @invcheckoff begin
        @zeros T a b c b_a ab αc ac
        a += y.x
        b += y.y
        c += y.z
        b_a += b-a
        ab += a * b
        αc += (8/3) * c
        c -= 27
        ac += a * c
    end
    y!.x += 10 * b_a
    y!.y -= ac + b
    y!.z += ab - αc
    ~@routine
end

@i function rk4_step!(f, y!::T, y::T, θ; Δt, t) where T
    @routine @invcheckoff begin
        @zeros T k1 k2 k3 k4 o1 o2 o3 o4 yk1 yk2 yk3
        f(o1, t, y, θ)
        k1 += Δt * o1
        yk1 += y
        yk1 += k1 / 2
        t += Δt/2
        f(o2, t, yk1, θ)
        k2 += Δt * o2
        yk2 += y
        yk2 += k2 / 2
        f(o3, t, yk2, θ)
        k3 += Δt * o3
        yk3 += y
        yk3 += k3
        t += Δt/2
        f(o4, t, yk3, θ)
        k4 += Δt * o4
    end
    y! += y
    y! += k1 / 6
    y! += k2 / 3
    y! += k3 / 3
    y! += k4 / 6
    ~@routine
end

@i function rk4!(f, history, y0::T, θ; t0, Δt, Nt) where T
    history[1] += y0
    @invcheckoff @inbounds for i=1:Nt
        rk4_step!(f, history[i+1], history[i], θ; Δt=Δt, t=t0+(i-1)*Δt)
    end
end

@i function iloss!(out, f, history, y0, θ; t0, Δt, Nt)
    rk4!((@const f), history, y0, θ; t0=t0, Δt=Δt, Nt=Nt)
    out += history[end].x
end