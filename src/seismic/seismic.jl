module Seismic
using NiLang, Plots
import ReversibleSeismic
using ReversibleSeismic: i_one_step!, AcousticPropagatorParams, Ricker, i_solve!, solve, treeverse_solve, bennett_step!,
    SeismicState

"""
the reversible loss
"""
@i function i_loss!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T}, tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
    i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
	out -= tu[size(c,1)÷2,size(c,2)÷2+20,end]
end

@i function i_loss_bennett!(out, state, param, srci, srcj, srcv, c; k, logger=NiLang.BennettLog())
    bennett!((@const bennett_step!), state, k, 1, (@const param.NSTEP-1), param, srci, srcj, srcv, c; do_uncomputing=false, logger=logger)
    out -= state[param.NSTEP].u[size(c,1)÷2,size(c,2)÷2+20]
end

function loss(param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T}) where T
    useq = solve(param, srci, srcj, srcv, c)
	-useq[size(tu,1)÷2,size(tu,2)÷2+20,end]
end

function generate_useq(c; nstep, method=:julia, bennett_k=50, usecuda=false)
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
        if usecuda
            state = Dict(1=>ReversibleSeismic.CuSeismicState(Float64, nx, ny))
            param = cu(param)
            c = CuArray(c)
        end
        x = SeismicState(Float64, nx, ny)
        logger = NiLang.BennettLog()
        state = Dict(1=>copy(x))
        bennett!(bennett_step!, state, bennett_k, 1, nstep-1, param, srci, srcj, srcv, c; do_uncomputing=false, logger=logger)
        println(logger)
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
function getgrad(c::AbstractMatrix{T}; nstep::Int, method=:nilang, treeverse_δ=50, bennett_k=50, usecuda=false) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
    srci = size(c, 1) ÷ 2 - 1
    srcj = size(c, 2) ÷ 2 - 1
    srcv = Ricker(param, 100.0, 500.0)
    nx, ny = size(c, 1) - 2, size(c, 2) - 2
    c = copy(c)
    if method == :nilang
        tu = zeros(T, size(c)..., nstep+1)
        tφ = zeros(T, size(c)..., nstep+1)
        tψ = zeros(T, size(c)..., nstep+1)
        res = NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ))
        return res[end-2][:,:,2], res[end-4], res[end-3]
    elseif method == :treeverse
        logger = ReversibleSeismic.TreeverseLog()
        s0 = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn.u[size(c,1)÷2,size(c,2)÷2+20] -= 1.0
        g_tv_x, g_tv_srcv, g_tv_c = treeverse_solve(s0, x->(gn, zero(srcv), zero(c));
            param=param, c=c, srci=srci, srcj=srcj,
            srcv=srcv, δ=treeverse_δ, logger=logger)
        println(logger)
        return g_tv_x.u, g_tv_srcv, g_tv_c
    elseif method == :bennett
        if usecuda
            c = CuArray(c)
            s0 = ReversibleSeismic.CuSeismicState(Float64, nx, ny)
        else
            s0 = SeismicState(Float64, nx, ny)
        end
        logger = NiLang.BennettLog()
        state = Dict(1=>s0)
        _,gx,_,_,_,gsrcv,gc = NiLang.AD.gradient(i_loss_bennett!, (0.0, state, param, srci, srcj, srcv, c); iloss=1, k=bennett_k, logger=logger)
        println(logger)
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