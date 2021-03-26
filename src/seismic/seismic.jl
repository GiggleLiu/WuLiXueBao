module Seismic
using NiLang, Plots
using KernelAbstractions
using ReversibleSeismic
using KernelAbstractions.CUDA

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
        x = SeismicState(Float64, nx, ny)
        if usecuda
            x = togpu(x)
            param = togpu(param)
            c = CuArray(c)
        end
        logger = NiLang.BennettLog()
        state = Dict(1=>x)
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
        if usecuda
            c = CuArray(c)
            s0 = togpu(s0)
            gn = togpu(gn)
            param = togpu(param)
        end
        g_tv_x, g_tv_srcv, g_tv_c = treeverse_solve(s0, x->(gn, zero(srcv), zero(c));
            param=param, c=c, srci=srci, srcj=srcj,
            srcv=srcv, δ=treeverse_δ, logger=logger)
        println(logger)
        return g_tv_x.u, g_tv_srcv, g_tv_c
    elseif method == :bennett
        s0 = SeismicState(Float64, nx, ny)
        if usecuda
            c = CuArray(c)
            s0 = togpu(s0)
            param = togpu(param)
        end
        logger = NiLang.BennettLog()
        state = Dict(1=>s0)
        CUDA.allowscalar(true)
        _,gx,_,_,_,gsrcv,gc = NiLang.AD.gradient(i_loss_bennett!, (0.0, state, param, srci, srcj, srcv, c); iloss=1, k=bennett_k, logger=logger)
        CUDA.allowscalar(false)
        println(logger)
        return gx[1].u, gsrcv, gc
    else
        error("")
    end
end

function timing(nx::Int, ny::Int; nstep::Int, δ, usecuda=true)
    c = rand(nx+2, ny+2)
    #g_nilang = getgrad(c, nstep=1000, method=:nilang)
    println("treeverse GPU, size = ($nx, $ny), step = $(nstep), δ = $δ")
    @time getgrad(c; nstep=nstep, method=:treeverse, usecuda=usecuda, treeverse_δ=δ)
    @time getgrad(c; nstep=nstep, method=:treeverse, usecuda=usecuda, treeverse_δ=δ)
    #g_bennett = getgrad(c, nstep=1000, method=:bennett)
end

end
