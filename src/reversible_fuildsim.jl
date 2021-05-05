# https://www.researchgate.net/profile/Jos-Stam/publication/2560062_Real-Time_Fluid_Dynamics_for_Games/links/0f317536cf9c741475000000/Real-Time-Fluid-Dynamics-for-Games.pdf

using NiLang

# grid space h = 1/Nx
## Moving Densities
@i function add_source(x, s, dt)
    @inbounds for i=1:length(x)
        x[i] += dt*s[i]
    end
end

## Gauss-Seidel relaxation to simulate diffuse
@i function diffuse(b, x::AbstractMatrix{T}, x0, diffusion::TD, gar; dt, h, gs_iter) where {T,TD}
    @routine begin
        Nx, Ny ← size(x) .- 2
        dthh ← dt/h/h
        @zeros TD a foura
        a += diffusion*dthh
        foura += 4*a
        INC(foura)
        f1 += x0[i,j] / foura
        f2 += a/foura
    end
    @inbounds for k=2:gs_iter+1, j=2:Ny+1, i=2:Nx+1
        x[i,j,k] +=  f1
        x[i,j,k] += f2 * x[i-1,j,k-1]
        x[i,j,k] += f2 * x[i+1,j,k-1]
        x[i,j,k] += f2 * x[i,j-1,k-1]
        x[i,j,k] += f2 * x[i,j+1,k-1]
    end
    ~@routine
    set_bnd(b, x[:,:,gs_iter], gar)
end

@i @inline function split_fraction(i0::Int, s1, x0, N)
    if x<1.5
        s1 += 1.5
    elseif x>N+1.5
        s1 += N+1.5
    else
        si += x
    end
    i0 += @const floor(Int, s1)
    s1 -= i0
end

@i function advect(b, d, d0, u::AbstractMatrix{T}, v, gar; dt, h) where T
    Nx, Ny = size(u) .- 2
    dt0 = dt/h
    @inbounds for j=2:Ny+1, i=2:Nx+1
        @routine begin
            @zeros T s0 s1 t0 t1 x y
            @zeros Int i0 j0
            x += i
            x -= dt0*u[i,j]
            y += j
            y -= dt0*v[i,j]
            split_fraction(i0, s1, x, Nx)
            split_fraction(j0, t1, y, Ny)
            s0 += 1 - s1
            t0 += 1 - t1
            a += t0*d0[i0,j0]
            a += t1*d0[i0,j0+1]
            b += t0*d0[i0+1,j0]
            b += t1*d0[i0+1,j0+1]
        end
        d[i,j] += s0*a + s1*b
        ~@routine
    end
    set_bnd(b, d, gar)
end

@i function dens_step(x, x0, u, v, diffusion, gar; dt, h, gs_iter)
    add_source(x, x0, dt)
    SWAP(x0, x[:,:,1])
    diffuse(0, x, x0, diffusion, gar; dt=dt, h=h, gs_iter=gs_iter)
    SWAP(x0, x[:,:,gs_iter+1])
    advect(0, x, x0, u, v; dt=dt, h=h)
end

## Evolving Velocities
@i function vel_step(u, v, u0, v0; viscosity, dt, h, gs_iter)
    add_source(u, u0, dt)
    add_source(v, v0, dt)
    SWAP(u0, u)
    diffuse(1, u, u0, viscosity; dt=dt, h=h, gs_iter=gs_iter)
    SWAP(v0, v)
    diffuse(2, v, v0, viscosity; dt=dt, h=h, gs_iter)
    project(u, v, u0, v0; h=h, gs_iter=gs_iter)
    SWAP(u0, u)
    SWAP(v0, v)
    advect(1, u, u0, u0, v0; dt=dt, h=h)
    advect(2, v, v0, u0, v0; dt=dt, h=h)
    project(u, v, u0, v0; h=h, gs_iter=gs_iter)
end

@i function project(u, v, p, div; h, gs_iter)
    Nx, Ny = size(u) .- 2
    @inbounds for j=2:Ny+1, i=2:Nx+1
        div[i,j] = -0.5*h*(u[i+1,j]-u[i-1,j] + v[i,j+1]-v[i,j-1])
    end
    set_bnd(0, div, gar1)
    @inbounds for k=2:gs_iter+1, j=2:Ny+1, i=2:Nx+1
        @routine begin
            sx ← zero(T)
            sx += p[i-1,j,k-1]+p[i+1,j,k-1]
            sx += p[i,j-1,k-1]+p[i,j+1,k-1]
            sx += div[i,j]
        end
        p[i,j,k] += sx * 0.25
        ~@routine
    end
    set_bnd(0, p, gar2)
    
    f1 = 0.5/h
    @inbounds for j=2:Ny+1, i=2:Nx+1
        dx += p[i+1,j]-p[i-1,j]
        dy += p[i,j+1]-p[i,j-1]
        u[i,j] -= f1*dx
        v[i,j] -= f1*dy
    end
    set_bnd(1, u, gar3)
    set_bnd(2, v, gar4)
end

# garbage size is 2Nx + 2Ny + 4
@i function set_bnd(b, x, garbage)
    Nx, Ny = size(x) .- 2
    @inbounds for i=2:Nx+1
        SWAP(x[i,1], garbage[2i-1])
        x[i,1] += b==2 ? -x[i,2] : x[i,2]
        SWAP(x[i,Ny+2], garbage[2i])
        x[i,Ny+2] += b==2 ? -x[i,Ny+1] : x[i,Ny+1]
    end
    @inbounds for j=2:Ny+1
        SWAP(x[1,j], garbage[2Nx+2j-1])
        x[1,j] += b==1 ? -x[2,j] : x[2,j]
        SWAP(x[Nx+2,j], garbage[2Nx+2j])
        x[Nx+2,j] += b==1 ? -x[Nx+1,j] : x[Nx+1,j]
    end

    @inbounds begin
        SWAP(x[1,j], garbage[2Nx+2j-1])
        x[1,1] += 0.5*(x[2,1]+x[1,2])
        x[1,Ny+2] += 0.5*(x[2,Ny+2]+x[1,Ny+1])
        x[Nx+2,1] += 0.5*(x[Nx+1,1]+x[Nx+2,2])
        x[Nx+2,Ny+2] += 0.5*(x[Nx+1,Ny+2]+x[Nx+2,Ny+1])
    end
end

@i function simulate!(res, u, v, u_pre, v_pre; viscosity, dt, h)
    res[:,:,1] += dens_prev
    for i=1:nstep
        @safe @show i
        #vel_step(u, v, u_prev, v_prev; viscosity=visc, dt=dt, h=h)
        dens_step(dens, dens_prev, u, v; diffusion=diff, dt=dt, h=h)
        res[:,:,i+1] += dens
    end
end

function run(Nx::Int, Ny::Int; nstep, dt, h=1.0/Nx, visc=0.0, diff)
    u, v, dens, u_prev, v_prev, dens_prev = [zeros(Nx+2, Ny+2) for i=1:6]

    dens_prev[Nx÷2:Nx÷2+5,Nx÷2:Nx÷2+5] .= 1
    u_prev .= 1.0
    v_prev .= 0.0

    res = zeros(Nx+2, Ny+2, nstep+1)
    simulate!(res, u, v, u_pre, v_pre; viscosity, dt, h)
end

using Plots
function generate_animation(tu_seq; stepsize=20, fps=5)
    ani = @animate for i=1:stepsize:size(tu_seq, 3)
        heatmap(tu_seq[2:end-1,2:end-1,i])
    end
    gif(ani, fps=fps)
end

res = run(200, 200; nstep=100, dt=0.1, h=1/200, visc=5e-4, diff=5e-4)
generate_animation(res; stepsize=5, fps=5)