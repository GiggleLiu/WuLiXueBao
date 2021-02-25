y0 = P3(1.0, 0.0, 0.0)

@i function lorentz_step!(y!::T, y::T; Δt) where T
    rk4_step!((@skip! lorentz!), y!, y; Δt, t=0.0)
end

_, x_last_b, _ = bennett(lorentz_step!, zero(P3{Float64}), y0; k=4, nsteps=Nt, Δt=Δt)
@i function loss(out, step, y, x; kwargs...)
    bennett((@skip! step), y, x; kwargs...)
    out += y[n÷2]
end