# https://www.researchgate.net/profile/Jos-Stam/publication/2560062_Real-Time_Fluid_Dynamics_for_Games/links/0f317536cf9c741475000000/Real-Time-Fluid-Dynamics-for-Games.pdf

using NiLang

# grid space h = 1/Nx
# Moving Densities
@i function add_source(x, s, dt)
    for i=1:length(x)
        x[i] += dt*s[i]
    end
end

# Gauss-Seidel relaxation to simulate diffuse
@i function diffuse(b, x::AbstractArray{T,3}, x0, diffusion::TD, gars; dt, h, gs_iter) where {T,TD}
    @routine begin
        Nx, Ny ← size(x0) .- 2
        dthh ← dt/h/h
        @zeros TD a foura f1 f2
        a += diffusion*dthh
        foura += 4*a
        INC(foura)
        f1 += inv(foura)
        f2 += a/foura
    end
    @inbounds for k=2:gs_iter+1
        for j=2:Ny+1
            for i=2:Nx+1
                @routine @invcheckoff begin
                    @zeros T s m
                    s += x[i-1,j,k-1] + x[i+1,j,k-1]
                    s += x[i,j-1,k-1] + x[i,j+1,k-1]
                    m += f1 * x0[i,j]
                    m += f2 * s
                end
                x[i,j,k] += m
                ~@routine
            end
        end
        set_bnd(b, x.[:,:,gs_iter], gars.[:,k-1])
    end
    ~@routine
end

@i @inline function split_fraction(i0::Int, s1, x, N)
    @invcheckoff if x<1.5
        s1 += 1.5
    elseif x>N+1.5
        s1 += N+1.5
    else
        s1 += x
    end
    i0 += @skip! floor(Int, s1)
    s1 -= i0
end

@i function advect(b, d, d0, u::AbstractMatrix{T}, v, gar; dt, h) where T
    Nx, Ny ← size(u) .- 2
    @routine begin
        dt0 ← zero(dt)
        dt0 += dt/h
    end
    @inbounds for j=2:Ny+1
        for i=2:Nx+1
            @routine @invcheckoff begin
                @zeros T s0 s1 t0 t1 x y A B
                @zeros Int i0 j0
                x += i
                x -= dt0*u[i,j]
                y += j
                y -= dt0*v[i,j]
                split_fraction(i0, s1, x, Nx)
                split_fraction(j0, t1, y, Ny)
                s0 += 1 - s1
                t0 += 1 - t1
                A += t0*d0[i0,j0]
                A += t1*d0[i0,j0+1]
                B += t0*d0[i0+1,j0]
                B += t1*d0[i0+1,j0+1]
            end
            d[i,j] += s0*A
            d[i,j] += s1*B
            ~@routine
        end
    end
    ~@routine
    set_bnd(b, d, gar)
end

@i function dens_step(x, x0, u, v, diffusion, gars; dt, h, gs_iter)
    add_source(x.[:,:,1], x0, dt)
    SWAP.(x0, x.[:,:,1])
    diffuse(0, x, x0, diffusion, gars; dt=dt, h=h, gs_iter=gs_iter)
    SWAP.(x0, x.[:,:,gs_iter+1])
    advect(0, x.[:,:,gs_iter+1], x0, u, v, gars.[:,gs_iter+1]; dt=dt, h=h)
end

# Evolving Velocities
@i function vel_step(u, v, u0, v0, p1, div1, p2, div2, viscosity, gars; dt, h, gs_iter)
    add_source(u.[:,:,1], u0, dt)
    SWAP.(u0, u.[:,:,1])
    diffuse(1, u, u0, viscosity, gars.[:,1:gs_iter]; dt=dt, h=h, gs_iter=gs_iter)

    add_source(v.[:,:,1], v0, dt)
    SWAP.(v0, v.[:,:,1])
    diffuse(2, v, v0, viscosity, gars.[:,gs_iter+1:2*gs_iter]; dt=dt, h=h, gs_iter)

    project(u.[:,:,gs_iter+1], v.[:,:,gs_iter+1], p1, div1, gars.[:,2*gs_iter+1:3*gs_iter+2]; h=h, gs_iter=gs_iter)
    @routine @invcheckoff begin
        u0_ ← zero(u0)
        v0_ ← zero(v0)
        u0_ += u0
        v0_ += v0
    end
    advect(1, u.[:,:,gs_iter+1], u0, u0_, v0, gars.[:,3*gs_iter+3]; dt=dt, h=h)
    advect(2, v.[:,:,gs_iter+1], v0_, u0, v0, gars.[:,3*gs_iter+4]; dt=dt, h=h)
    ~@routine
    project(u.[:,:,gs_iter+1], v.[:,:,gs_iter+1], p2, div2, gars.[:,3*gs_iter+5:4*gs_iter+6]; h=h, gs_iter=gs_iter)
end

# div and p are zeros
@i function project(u, v, p::AbstractArray{T,3}, div, gars; h, gs_iter) where T
    Nx, Ny ← size(u) .- 2
    @routine begin
        @zeros Float64 f1 halfh
        f1 += 0.5/h
        halfh += 0.5*h
    end
    @inbounds for j=2:Ny+1
        for i=2:Nx+1
            div[i,j] -= halfh * u[i+1,j]
            div[i,j] += halfh * u[i-1,j]
            div[i,j] -= halfh * v[i,j+1]
            div[i,j] += halfh * v[i,j-1]
        end
    end
    @inbounds for k=2:gs_iter+1
        for j=2:Ny+1
            for i=2:Nx+1
                @routine @invcheckoff begin
                    sx ← zero(T)
                    sx += p[i-1,j,k-1]+p[i+1,j,k-1]
                    sx += p[i,j-1,k-1]+p[i,j+1,k-1]
                    sx += div[i,j]
                end
                p[i,j,k] += sx * 0.25
                ~@routine
            end
        end
        set_bnd(0, p.[:,:,k], gars.[:,k])
    end
    
    @inbounds for j=2:Ny+1
        for i=2:Nx+1
            @routine @invcheckoff begin
                @zeros T t1 t2
                t1 += p[i-1,j,gs_iter+1] - p[i+1,j,gs_iter+1]
                t2 += p[i-1,j,gs_iter+1] - p[i+1,j,gs_iter+1]
            end
            u[i,j] += f1 * t1
            v[i,j] += f1 * t2
            ~@routine
        end
    end
    ~@routine
    set_bnd(1, u, gars.[:,gs_iter+1])
    set_bnd(2, v, gars.[:,gs_iter+2])
end

# garbage size is 2Nx + 2Ny + 4
@i function set_bnd(b, x, garbage)
    Nx, Ny ← size(x) .- 2
    @inbounds for i=2:Nx+1
        SWAP(x[i,1], garbage[2i-1])
        if b==2 
            x[i,1] += -x[i,2]
        else
            x[i,1] += x[i,2]
        end
        SWAP(x[i,Ny+2], garbage[2i])
        if b==2
            x[i,Ny+2] += -x[i,Ny+1]
        else
            x[i,Ny+2] += x[i,Ny+1]
        end
    end
    @inbounds for j=2:Ny+1
        SWAP(x[1,j], garbage[2Nx+2j-1])
        x[1,j] += b==1 ? -x[2,j] : x[2,j]
        SWAP(x[Nx+2,j], garbage[2Nx+2j])
        x[Nx+2,j] += b==1 ? -x[Nx+1,j] : x[Nx+1,j]
    end

    @inbounds begin
        SWAP(x[1,1], garbage[end-3])
        x[1,1] += 0.5*x[2,1]
        x[1,1] += 0.5*x[1,2]
        SWAP(x[1,Ny+2], garbage[end-2])
        x[1,Ny+2] += 0.5*x[2,Ny+2]
        x[1,Ny+2] += 0.5*x[1,Ny+1]
        SWAP(x[Nx+2,1], garbage[end-1])
        x[Nx+2,1] += 0.5*x[Nx+1,1]
        x[Nx+2,1] += 0.5*x[Nx+2,2]
        SWAP(x[Nx+2,Ny+2], garbage[end])
        x[Nx+2,Ny+2] += 0.5*x[Nx+1,Ny+2]
        x[Nx+2,Ny+2] += 0.5*x[Nx+2,Ny+1]
    end
end

@i function simulate!(dens_out::AbstractArray{T,3}, u_out, v_out, dens_prev, u_prev, v_prev, viscosity, diffusion; dt, h, nstep, gs_iter) where T
    Nx, Ny ← size(u_prev) .- 2
    @inbounds for i=1:nstep
        @safe @show i
        dens_out.[:,:,1] += dens_prev
        u_out.[:,:,1] += u_prev
        v_out.[:,:,1] += v_prev
        @routine @invcheckoff begin
            gars1 ← zeros(T, 2Nx+2Ny+4, 4*gs_iter+6)
            @zeros u_prev div1 div2
            u ← zeros(T, size(u_prev)..., gs_iter+1)
            @zeros u p1 p2 v dens
            u.[:,:,1] += u_out.[:,:,i]
            v.[:,:,1] += v_out.[:,:,i]
            vel_step(u, v, u_prev, v_prev, p1, div1, p2, div2, viscosity, gars1; dt=dt, h=h, gs_iter=gs_iter)
            gars2 ← zeros(T, 2Nx+2Ny+4, gs_iter+1)
            dens.[:,:,1] += dens_out.[:,:,i]
            dens_step(dens, dens_prev, u.[:,:,gs_iter+1], v.[:,:,gs_iter+1], diffusion, gars2; dt=dt, h=h, gs_iter=gs_iter)
        end
        dens_out.[:,:,i+1] += dens.[:,:,gs_iter+1]
        ~@routine
    end
end

function run(Nx::Int, Ny::Int; nstep, dt, h=1.0/Nx, viscosity=0.0, diffusion, gs_iter=20)
    dens, u_prev, v_prev, dens_prev = [zeros(Nx+2, Ny+2) for i=1:6]

    u_prev .= 1.0
    v_prev .= 0.0

    dens_out = zeros(Nx+2, Ny+2, nstep+1)
    u_out = zeros(Nx+2, Ny+2, nstep+1)
    v_out = zeros(Nx+2, Ny+2, nstep+1)
    dens_prev[Nx÷2:Nx÷2+5,Nx÷2:Nx÷2+5,1] .= 1
    simulate!(dens_out, u_out, v_out, dens_prev, u_prev, v_prev, viscosity, diffusion; dt, h, nstep, gs_iter)[1]
end

using Plots
function generate_animation(tu_seq; stepsize=20, fps=5)
    ani = @animate for i=1:stepsize:size(tu_seq, 3)
        heatmap(tu_seq[2:end-1,2:end-1,i])
    end
    gif(ani, fps=fps)
end

@time res = run(200, 200; nstep=100, dt=0.1, h=1/200, viscosity=5e-4, diffusion=5e-4)
generate_animation(res; stepsize=5, fps=5)