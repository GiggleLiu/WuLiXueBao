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
        αc += (θ |> tget(3)) * c
        c -= (θ |> tget(2))
        ac += a * c
    end
    y!.x += (θ |> tget(1)) * b_a
    y!.y -= ac + b
    y!.z += ab - αc
    ~@routine
end


@i function iloss!(out, f, history, y0, θ; ts)
    i_ODESolve(RK4(), (@const f), history, y0, θ; ts=ts)
    out += history[end].x
end