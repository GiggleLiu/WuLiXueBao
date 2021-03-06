include("julia.jl")
include("reversible_programming.jl")

struct BennettLog
    fcalls::Vector{NTuple{3,Any}}  # depth, function index f_i := s_{i-1} -> s_{i}, length should be `(2k-1)^n` and function
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
    depth::Base.RefValue{Int}
end
BennettLog() = BennettLog(NTuple{3,Any}[], Ref(0), Ref(0))


# hacking the reversible program
function logfcall(l::BennettLog, i, f)
    push!(l.fcalls, (l.depth[], i, f))
    l, i, f
end
function ilogfcall(l::BennettLog, i, f)
    push!(l.fcalls, (l.depth[], i, ~f))
    l, i, f
end

@dual logfcall ilogfcall


@i function bennett_algorithm(step, y::T, x::T, args...; k::Int, nsteps::Int, kwargs...) where T
    state ← Dict{Int, T}()
    state[1] ← zero(x)
    state[1] +=  x
    bennett_algorithm((@skip! step), state, k, 1, nsteps, args...; kwargs...)
    SWAP(y, state[nsteps+1])
    state[1] -= x
    state[1] → zero(x)
    state[nsteps+1] → zero(x)
    state → Dict{Int, T}()
end

@i function bennett_algorithm(step, state::Dict{Int,T}, k::Int, base, len, args...; logger, kwargs...) where T
    @safe logger.depth[] += 1
    @invcheckoff if len == 1
        state[base+1] ← zero(state[base])
        @safe logger.peak_mem[] = max(logger.peak_mem[], length(state))
        step(state[base+1], state[base], args...; kwargs...)
        logfcall(logger, (@const base+1), step)
    else
        @routine begin
            @zeros Int nstep n
            n += ceil((@skip! Int), (@const len / k))
            nstep += ceil((@skip! Int), (@const len / n))
        end
        for j=1:nstep
            bennett_algorithm(step, state, k, (@const base+n*(j-1)), (@const min(n,len-n*(j-1))), args...; logger=logger, kwargs...)
        end
        for j=nstep-1:-1:1
            ~bennett_algorithm(step, state, k, (@const base+n*(j-1)), n, args...; logger=logger, kwargs...)
        end
        ~@routine
    end
end

@i function lorentz_step!(y!::T, y::T; Δt) where T
    rk4_step!((@skip! lorentz!), y!, y, (@const nothing); Δt, t=0.0)
end

@i function bennett_loss(out, step, y, x; kwargs...)
    bennett_algorithm((@skip! step), y, x; kwargs...)
    out += y.x
end

using Test, ForwardDiff
@testset "bennett gradient" begin
    x0 = P3(1.0, 0.0, 0.0)
    state = Dict{Int,Tuple{Float64,P3{Float64}}}()

    for Nt in [20, 120, 126]
        g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=Nt)[end].x, [x0.x, x0.y, x0.z])
        logger = BennettLog()
        g_bn = NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=3, nsteps=Nt, logger=logger)[4]
        @test g_fd ≈ [g_bn.x, g_bn.y, g_bn.z]
        @test length(logger.fcalls) > 0
    end
end