using ReversibleSeismic, TreeverseAndBennett.Seismic, Plots

nx = ny = 201

nstep = 1000

src = (nx÷2, ny÷5)

detector_locs = CartesianIndex.([(50, 50), (50, 100), (50, 150), (150, 50), (150, 100), (150, 150)])

param = AcousticPropagatorParams(nx=nx, ny=ny, 
        nstep=nstep, dt=0.1/nstep,  dx=200/(nx-1), dy=200/(nx-1),
        Rcoef = 1e-8)

rc = Ricker(param, 30.0, 200.0, 1e6)

c2 = three_layers(nx, ny)

solve(param, src, rc, c2)

target_pulses = solve_detector(param, src, rc, c2, detector_locs)

heatmap(sqrt.(c2))
scatter!(map(x->x.I[2], detector_locs), map(x->x.I[1], detector_locs))
scatter!([src[2]], [src[1]])

plot(target_pulses')

c20 = 3300^2*ones(nx+2,ny+2)
loss, (gin, gsrcv, gc), log = getgrad_mse(c2=c20, param=param, src=src, srcv=rc,
                    target_pulses=target_pulses, detector_locs=detector_locs,
                    method=:treeverse, treeverse_δ=20, usecuda=false)
heatmap(gc, clim=(0, 1e-13))
