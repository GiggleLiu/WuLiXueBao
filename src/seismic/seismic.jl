module Seismic
using NiLang, Plots
import ReversibleSeismic
using ReversibleSeismic: i_one_step!, AcousticPropagatorParams, Ricker, i_solve!, solve, treeverse_solve, bennett_step!

"""
the reversible loss
"""
@i function i_loss!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T}, tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
    i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
	out -= tu[size(c,1)÷2,size(c,2)÷2+20,end]
end

@i function i_loss_bennett!(out, step, state, param, srci, srcj, srcv, c; k, logger=NiLang.BennettLog())
    #bennett((@skip! step), y, x, param, srci, srcj, srcv, c; kwargs...)
    bennett!((@const step), state, k, 1, (@const param.NSTEP-1), param, srci, srcj, srcv, c; do_uncomputing=false, logger=logger)
    out -= state[param.NSTEP].u[size(c,1)÷2,size(c,2)÷2+20]
end

function loss(param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T}) where T
    useq = solve(param, srci, srcj, srcv, c)
	-useq[size(tu,1)÷2,size(tu,2)÷2+20,end]
end

function generate_useq(c; nstep, method=:julia, bennett_k=50)
	nx, ny = size(c) .- 2
	param = AcousticPropagatorParams(nx=nx, ny=ny,
	    Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
 	srci = nx ÷ 2
 	srcj = ny ÷ 2
    srcv = Ricker(param, 100.0, 500.0)
    c = copy(c)
    if method == :julia
        useq = solve(param, srci, srcj, srcv, c)
        return useq
    elseif method == :nilang
        tu = zeros(nx+2, ny+2, nstep+1)
        tφ = zeros(nx+2, ny+2, nstep+1)
        tψ = zeros(nx+2, ny+2, nstep+1)
 	    res = i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
	    return res[end-2]
    elseif method == :bennett
        x = ReversibleSeismic.SeismicState(Float64, nx, ny)
        logger = NiLang.BennettLog()
        state = Dict(1=>copy(x))
        bennett!(bennett_step!, state, bennett_k, 1, nstep-1, param, srci, srcj, srcv, copy(c); do_uncomputing=false, logger=logger)
        return state[nstep].u
    else
        error("")
    end
end

# generate u sequence
function generate_useq_demo(; nx=80, ny=80, nstep=1500, method=:julia)
    c = 700 * (1 .+ sin.(LinRange(0, 5π, nx+2))' .* cos.(LinRange(0, 3π, ny+2)));
    return generate_useq(c; nstep=nstep, method=method)
end

function generate_animation(tu_seq; stepsize=5, fps=5)
    ani = @animate for i=1:stepsize:size(tu_seq, 3)
        heatmap(tu_seq[2:end-1,2:end-1,i], clim=(-0.005, 0.005))
    end
    gif(ani, fps=fps)
end

"""
obtain gradients with NiLang.AD
"""
function getgrad(c::AbstractMatrix{T}; nstep::Int, method=:nilang, treeverse_δ=50, bennett_k=50) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
    srci = size(c, 1) ÷ 2 - 1
    srcj = size(c, 2) ÷ 2 - 1
    srcv = Ricker(param, 100.0, 500.0)
    c = copy(c)
    if method == :nilang
       tu = zeros(T, size(c)..., nstep+1)
       tφ = zeros(T, size(c)..., nstep+1)
       tψ = zeros(T, size(c)..., nstep+1)
       res = NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ))
       return res[end-2][:,:,2], res[end-4], res[end-3]
    elseif method == :treeverse
        nx, ny = size(c, 1) - 2, size(c, 2) - 2
        log = ReversibleSeismic.TreeverseLog()
        s0 = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn.u[size(c,1)÷2,size(c,2)÷2+20] -= 1.0
        g_tv_x, g_tv_srcv, g_tv_c = treeverse_solve(s0, (gn, zero(srcv), zero(c));
            param=param, c=c, srci=srci, srcj=srcj,
            srcv=srcv, δ=50, logger=log)
        return g_tv_x.u, g_tv_srcv, g_tv_c
    elseif method == :bennett
        nx, ny = size(c, 1) - 2, size(c, 2) - 2
        x = ReversibleSeismic.SeismicState(Float64, nx, ny)
        _,_,gx,_,_,_,gsrcv,gc = NiLang.AD.gradient(i_loss_bennett!, (0.0, bennett_step!, Dict(1=>copy(x)), param, srci, srcj, srcv, copy(c)); iloss=1, k=bennett_k)
        return gx[1].u, gsrcv, gc
    else
        error("")
    end
end

#=
using Optim
function optimize_c(c, nstep)
    res = optimize(c->(@show i_loss_and_useq(c, nstep=nstep)[1]), (g, c)->(g.=getgrad(c; nstep=nstep)), c, BFGS(), Optim.Options(iterations=5, g_tol=1e-30, f_tol=1e-30))
    return res.minimizer
end
=#
end