module Seismic
using NiLang, Plots
using NiLang.AD
using KernelAbstractions
using CUDAKernels
using CUDA
using ReversibleSeismic
using ..TreeverseAndBennett
using DelimitedFiles

export generate_animation, three_layer, getgrad_three_layer, targetpulses_three_layer, loss_three_layer

"""
the reversible loss
"""
@i function i_loss!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T}, tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
    i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
	out -= tu[size(c,1)÷2,size(c,2)÷2+20,end]
end

@i function i_loss_bennett!(out, state, param, srci, srcj, srcv, c; k, logger=NiLang.BennettLog())
    bennett!((@const bennett_step!), state, k, 1, (@const param.NSTEP-1), param, srci, srcj, srcv, c; do_uncomputing=false, logger=logger)
    out -= state[param.NSTEP].u[ReversibleSeismic.SafeIndex(size(c,1)÷2,size(c,2)÷2+20)]
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

function generate_animation(tu_seq; stepsize=5, fps=5, clim=(-0.005, 0.005))
    ani = @animate for i=1:stepsize:size(tu_seq, 3)
        heatmap(tu_seq[2:end-1,2:end-1,i], clim=clim)
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
        res, (g_tv_x, g_tv_srcv, g_tv_c) = treeverse_solve(s0, x->(gn, zero(srcv), zero(c));
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
        _,gx,_,_,_,gsrcv,gc = NiLang.AD.gradient(i_loss_bennett!, (0.0, state, param, srci, srcj, srcv, c); iloss=1, k=bennett_k, logger=logger)
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

function get_layers(nx, ny)
    layers = ones(nx+2, ny+2)
    n_piece = div(nx + 1, 3) + 1
    for k = 1:3
        i_interval = (k-1)*n_piece+1:min(k*n_piece, nx+2)
        layers[:, i_interval] .= 0.5 + (k-1)*0.25
    end
    return (3300 .* layers) .^ 2
end


function three_layer(; nstep=1000, nx=201, ny=201)
    param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)

    rc = Ricker(param, 30.0, 200.0, 1e6)
 	srci = nx ÷ 2
 	srcj = ny ÷ 5
    c = get_layers(nx, ny)
    solve(param, srci, srcj, rc, c)
end

function loss_three_layer(; nx=201, ny=201, c=3300^2*ones(nx+2, ny+2), target_pulses, detector_locs, nstep=1000, usecuda=false)
    param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)
    rc = Ricker(param, 30.0, 200.0, 1e6)
 	srci = nx ÷ 2
 	srcj = ny ÷ 5
    if usecuda
        c = CuArray(c)
        detector_locs = CuArray(detector_locs)
        param = togpu(param)
    end
    solve_detector2(param, srci, srcj, rc, c, target_pulses, detector_locs).data[1]
end

function targetpulses_three_layer(; nx=201, ny=201, c=get_layers(nx, ny), nstep=1000)
    param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)
    li = LinearIndices((nx+2, ny+2))
    detector_locs = [li[i,ny÷5] for i=round.(Int, LinRange(1,nx+2,200))]
    #detector_locs = [CartesianIndex((rand(1:nx+2), rand(1:ny+2))) for i=1:200]
    rc = Ricker(param, 30.0, 200.0, 1e6)
 	srci = nx ÷ 2
 	srcj = ny ÷ 5
    target_pulses = solve_detector(param, srci, srcj, rc, c, detector_locs)
    return detector_locs, target_pulses
end

function getgrad_three_layer(; nx=201, ny=201, c0=3300^2*ones(nx+2, ny+2), nstep=1000, target_pulses, detector_locs,
         method=:treeverse, treeverse_δ=50, bennett_k=50, usecuda=false)
    param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)
    rc = Ricker(param, 30.0, 200.0, 1e6)
 	srci = nx ÷ 2
 	srcj = ny ÷ 5
    res, g, log = _getgrad(c0, param, srci, srcj, rc, target_pulses, detector_locs, method, treeverse_δ, bennett_k, usecuda)
    res, g, log
end

# loss is |u[:,40,:]-ut[:,40,:]|^2
function _getgrad(c, param, srci, srcj, srcv, target_pulses, detector_locs, method, treeverse_δ, bennett_k, usecuda)
    nx, ny = size(c) .- 2
    if method == :treeverse
        logger = ReversibleSeismic.TreeverseLog()
        s0 = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn.u[size(c,1)÷2,size(c,2)÷2+20] -= 1.0
        if usecuda
            c = CuArray(c)
            target_pulses = CuArray(target_pulses)
            detector_locs = CuArray(detector_locs)
            s0 = togpu(s0)
            gn = togpu(gn)
            param = togpu(param)
        end
        res, (g_tv_x, g_tv_srcv, g_tv_c) = treeverse_solve_detector(Glued(0.0, s0);
            target_pulses=target_pulses, detector_locs=detector_locs,
            param=param, c=c, srci=srci, srcj=srcj,
            srcv=srcv, δ=treeverse_δ, logger=logger)
        println(logger)
        return res.data[1], (g_tv_x.data[2].u, g_tv_srcv, g_tv_c), logger
    elseif method == :bennett
        s0 = SeismicState(Float64, nx, ny)
        if usecuda
            c = CuArray(c)
            target_pulses = CuArray(target_pulses)
            detector_locs = CuArray(detector_locs)
            s0 = togpu(s0)
            param = togpu(param)
        end
        logger = NiLang.BennettLog()
        state = Dict(1=>Glued(0.0, s0))
        args = i_loss_bennett_detector!(0.0, state, param, srci, srcj, srcv, c, target_pulses, detector_locs; k=bennett_k, logger=logger)
        loss = args[1]
        gargs = (~i_loss_bennett_detector!)(GVar(loss, 1.0), GVar.(args[2:end])...; k=bennett_k, logger=logger)
        _,gx,_,_,_,gsrcv,gc = grad.(gargs) #NiLang.AD.gradient(i_loss_bennett_detector!, (0.0, state, param, srci, srcj, srcv, c, target_pulses, detector_locs); iloss=1, k=bennett_k, logger=logger)
        println(logger)
        return loss, (gx[1].data[2].u, gsrcv, gc), logger
    else
        error("")
    end
end

#=
using Optim
export train_three_layer

function train_three_layer(; nx=201, ny=201, nstep=1000, usecuda=false, method=:treeverse, treeverse_δ=50, bennett_k=50, iterations=100)
    detector_locs, target_pulses = targetpulses_three_layer(; nx=nx, ny=ny, c=get_layers(nx, ny), nstep=nstep)
    c0=3300^2*ones(nx+2, ny+2)
    function gf!(g, x)
        _, _, gs = getgrad_three_layer(; nx=nx, ny=ny, nstep=nstep, c0=x, treeverse_δ=treeverse_δ, bennett_k=bennett_k, detector_locs=detector_locs, target_pulses=target_pulses, usecuda=usecuda, method=method)
        g .= 0
        g[2:nx+1, 2:ny+1] .= gs[3][2:nx+1, 2:ny+1]
    end
    opt = optimize(x->loss_three_layer(; c=x, detector_locs=detector_locs, target_pulses=target_pulses, nstep=nstep, nx=nx, ny=ny), gf!, c0, LBFGS(),
        Optim.Options(iterations=iterations, g_abstol=0, g_reltol=0, f_abstol=0, f_reltol=0))
    return opt.minimizer
end
=#

"""
returns the `target_pulses`
"""
function run_paper_example(; nx=1000, ny=1000, nstep=10000, method=:treeverse, treeverse_δ=50, bennett_k=50, usecuda=false)
    detector_locs, target_pulses = targetpulses_three_layer(; nx=nx, ny=ny, c=get_layers(nx, ny), nstep=nstep)
    c0 = 3300^2*(ones(nx+2, ny+2))
    t = @elapsed loss, (gx, gsrcv, gc), log = getgrad_three_layer(; nx=nx, ny=ny, c0=c0, nstep=nstep, target_pulses=target_pulses, detector_locs=detector_locs,
         method=method, treeverse_δ=treeverse_δ, bennett_k=bennett_k, usecuda=usecuda)
    return loss, gc, log, t
end

function benchmark_treeverse(; n=1000, nstep=10000,
        treeverse_δs = [5, 10, 20, 40, 80, 160],
        device=0,
    )
    CUDA.device!(device)
    run_paper_example(nx=n, ny=n, nstep=nstep, method=:treeverse, treeverse_δ=50, usecuda=true)
    res1 = zeros(4, length(treeverse_δs))
    for (i, treeverse_δ) in enumerate(treeverse_δs)
        println("case $i: δ = $treeverse_δ")
        _, _, log, t = run_paper_example(nx=n, ny=n, nstep=nstep, method=:treeverse, treeverse_δ=treeverse_δ, usecuda=true)
        ngcalls = count(x->x.action==:grad, log.actions)
        nfcalls = count(x->x.action==:call, log.actions) + ngcalls
        res1[:,i] .= log.peak_mem[], t, nfcalls, ngcalls
    end
    output_file1 = TreeverseAndBennett.project_relative_path("data", "cuda-gradient-treeverse.dat")
    writedlm(output_file1, res1)
end

function benchmark_bennett(; n=1000, nstep=10000,
        bennett_ks = [5, 10, 20, 40, 80, 160],
        device=0,
    )
    CUDA.device!(device)
    run_paper_example(nx=n, ny=n, nstep=nstep, method=:bennett, bennett_k=50, usecuda=true)
    res2 = zeros(4, length(bennett_ks))
    for (i,bennett_k) in enumerate(bennett_ks)
        println("case $i: k = $bennett_k")
        _, _, log, t = run_paper_example(nx=n, ny=n, nstep=nstep, method=:bennett, bennett_k=bennett_k, usecuda=true)
        nfcalls = ngcalls = length(log.fcalls) ÷ 2
        res2[:,i] .= log.peak_mem[], t, nfcalls, ngcalls
    end
    output_file2 = TreeverseAndBennett.project_relative_path("data", "cuda-gradient-bennett.dat")
    writedlm(output_file2, res2)
end

end
