# https://www.researchgate.net/profile/Jos-Stam/publication/2560062_Real-Time_Fluid_Dynamics_for_Games/links/0f317536cf9c741475000000/Real-Time-Fluid-Dynamics-for-Games.pdf

# grid space h = 1/Nx
# Moving Densities
function add_source(x, s, dt)
    @inbounds for i=1:length(x)
        x[i] += dt*s[i]
    end
end

function swap(a::AbstractArray, b::AbstractArray)
    @inbounds for i=1:length(a)
        a[i], b[i] = b[i], a[i]
    end
end

# Gauss-Seidel relaxation to simulate diffuse
function diffuse(b, x, x0, diff, dt, h)
    Nx, Ny = size(x) .- 2
    a=dt*diff/h/h
    f1, f2 = 1/(1+4*a), a/(1+4*a)
    @inbounds for k=1:20
        for j=2:Ny+1, i=2:Nx+1
            x[i,j] = f1*x0[i,j] + f2*(x[i-1,j]+x[i+1,j] + x[i,j-1]+x[i,j+1])
        end
        set_bnd(b, x)
    end
end
function diffuse_bad(b, x, x0, diff, dt, h)
    a=dt*diff/h/h
    Nx, Ny = size(x) .- 2
    for j=2:Ny+1, i=2:Nx+1
        x[i,j] = x0[i,j] + a*(x0[i-1,j]+x0[i+1,j] + x0[i,j-1]+x0[i,j+1]-4*x0[i,j])
        set_bnd(b, x)
    end
    @show sum(x0), sum(x)
end

clip(x, a, b) = (x<a ? a : (x>b ? b : x))

function advect(b, d, d0, u, v, dt, h)
    Nx, Ny = size(u) .- 2
    dt0 = dt/h
    @inbounds for j=2:Ny+1, i=2:Nx+1
        x = i - dt0*u[i,j]
        y = j - dt0*v[i,j]
        (x, y) = clip.((x, y), 1.5, (Nx, Ny) .+ 1.5)
        (i0, j0) = floor.(Int, (x, y))
        (s1, t1) = (x, y) .- (i0, j0)

        d[i,j] = (1-s1)*((1-t1)*d0[i0,j0]+t1*d0[i0,j0+1]) + s1*((1-t1)*d0[i0+1,j0]+t1*d0[i0+1,j0+1])
    end
    set_bnd(b, d)
end

function dens_step(x, x0, u, v; diffusion, dt, h)
    add_source(x, x0, dt)
    swap(x0, x)
    diffuse(0, x, x0, diffusion, dt, h)
    swap(x0, x)
    advect(0, x, x0, u, v, dt, h)
end

# Evolving Velocities
function vel_step(u, v, u0, v0; viscosity, dt, h)
    add_source(u, u0, dt)
    add_source(v, v0, dt)
    swap(u0, u)
    diffuse(1, u, u0, viscosity, dt, h)
    swap(v0, v)
    diffuse(2, v, v0, viscosity, dt, h)
    project(u, v, u0, v0, h)
    swap(u0, u)
    swap(v0, v)
    advect(1, u, u0, u0, v0, dt, h)
    advect(2, v, v0, u0, v0, dt, h)
    project(u, v, u0, v0, h)
end

function project(u, v, p, div, h)
    Nx, Ny = size(u) .- 2
    p .= 0
    @inbounds for j=2:Ny+1, i=2:Nx+1
        div[i,j] = -0.5*h*(u[i+1,j]-u[i-1,j] + v[i,j+1]-v[i,j-1])
    end
    set_bnd(0, div)
    @inbounds for k=1:20, j=2:Ny+1, i=2:Nx+1
        p[i,j] = (div[i,j]+p[i-1,j]+p[i+1,j] + p[i,j-1]+p[i,j+1]) * 0.25
    end
    set_bnd(0, p)
    
    f1 = 0.5/h
    @inbounds for j=2:Ny+1, i=2:Nx+1
        u[i,j] -= f1*(p[i+1,j]-p[i-1,j])
        v[i,j] -= f1*(p[i,j+1]-p[i,j-1])
    end
    set_bnd(1, u)
    set_bnd(2, v)
end

function set_bnd(b, x)
    Nx, Ny = size(x) .- 2
    @inbounds for i=2:Nx+1
        x[i,1] = b==2 ? -x[i,2] : x[i,2]
        x[i,Ny+2] = b==2 ? -x[i,Ny+1] : x[i,Ny+1]
    end
    @inbounds for j=2:Ny+1
        x[1,j] = b==1 ? -x[2,j] : x[2,j]
        x[Nx+2,j] = b==1 ? -x[Nx+1,j] : x[Nx+1,j]
    end

    @inbounds begin
        x[1,1] = 0.5*(x[2,1]+x[1,2])
        x[1,Ny+2] = 0.5*(x[2,Ny+2]+x[1,Ny+1])
        x[Nx+2,1] = 0.5*(x[Nx+1,1]+x[Nx+2,2])
        x[Nx+2,Ny+2] = 0.5*(x[Nx+1,Ny+2]+x[Nx+2,Ny+1])
    end
end

function run(Nx::Int, Ny::Int; nstep, dt, h=1.0/Nx, visc=0.0, diff)
    u, v, dens, u_prev, v_prev, dens_prev = [zeros(Nx+2, Ny+2) for i=1:6]

    dens_prev[Nx÷2:Nx÷2+5,Nx÷2:Nx÷2+5] .= 1
    u_prev .= 1.0
    v_prev .= 0.0

    res = zeros(Nx+2, Ny+2, nstep+1)
    res[:,:,1] .= dens_prev
    for i=1:nstep
        @show i
        vel_step(u, v, u_prev, v_prev; viscosity=visc, dt=dt, h=h)
        dens_step(dens, dens_prev, u, v; diffusion=diff, dt=dt, h=h)
        res[:,:,i+1] .= dens
    end
    return res
end

using Plots
function generate_animation(tu_seq; stepsize=20, fps=5)
    ani = @animate for i=1:stepsize:size(tu_seq, 3)
        heatmap(tu_seq[2:end-1,2:end-1,i])
    end
    gif(ani, fps=fps)
end

res = run(200, 200; nstep=100, dt=0.1, h=1/200, visc=5e-4, diff=2e-4);
generate_animation(res; stepsize=5, fps=5)