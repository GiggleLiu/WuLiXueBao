include("bennett.jl")
include("treeverse.jl")

using Compose, Viznet
LW = 0.3mm

function showtape(n::Int, checkpoints; removed=Int[], new=Int[], y=0.5, ngrad=0, flag=true, label="")
    unit = 1/(n+1)
    BG = "#AAAAAA"
    NEW = "#000000"
    EXIST = "#3D3D3D"
    r = 0.25
    a = 0.4
    grid = nodestyle(:square, fill(BG), stroke("transparent"), r=a*unit)
    grid_red = nodestyle(:square, fill("red"), stroke("transparent"), r=a*unit)
    pebble = nodestyle(:circle, fill(EXIST); r=r*unit)
    pebble_removed = nodestyle(:circle, fill("transparent"), stroke("black"), linewidth(0.1mm); r=r*unit)
    pebble_new = nodestyle(:circle, fill(NEW); r=r*unit)
    tb_flag = textstyle(:default, fontsize(2))
    tb = textstyle(:default, fontsize(14pt), fill("white"))
    pebble >> (0.5*unit, y)
    flag && tb_flag >> (((n+0.5)*unit, y), "🚩")
    if !isempty(label)
        tb >> ((1.0, y), label)
    end
    for p in setdiff(removed, checkpoints)
        pebble_removed >> ((p+0.5)*unit, y)
    end
    for p in new
        pebble_new >> ((p+0.5)*unit, y)
    end
    for p in setdiff(checkpoints, new)
        pebble >> ((p+0.5)*unit, y)
    end
    for i = 0:n-ngrad
        grid >> ((i+0.5)*unit, y)
    end
    for i = n-ngrad+1:n
        grid_red >> ((i+0.5)*unit, y)
    end
end

function bennett_pebblegame(N::Int, k)
    x0 = P3(1.0, 0.0, 0.0)
    logger = NiLang.BennettLog()
    #NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=k, nsteps=N, logger=logger)[4]
    bennett_loss(0.0, lorentz_step!, zero(P3{Float64}), x0; Δt=3e-3, k=k, N=N, logger=logger)
    @show logger.peak_mem[]
    NY = length(logger.fcalls)÷2+1
    X = 1cm*(N+1)
    Y = 1cm*NY
    Compose.set_default_graphic_size(X, Y)
    img = canvas() do
        pebbles = Int[]
        y = 0.5/(N+1)
        dy = 1/(N+1)

        for (i, f) in enumerate(logger.fcalls[1:NY])
            @show f
            if f[3] isa NiLangCore.Inv
                k = findfirst(==(f[2]-1), pebbles)
                deleteat!(pebbles, k)
                removed = [f[2]-1]
                new=[]
            else
                push!(pebbles, f[2]-1)
                removed = []
                new=[f[2]-1]
            end
            showtape(N, pebbles; y=y, ngrad=0, flag=true, removed=removed, new=new)
            y += dy
        end
    end
    img = Compose.compose(context(0, 0, 1.0, (N+1)/NY), img)
    return img, NY
end

using Compose, Viznet
function treeverse_pebblegame(N::Int, δ)
    x0 = P3(1.0, 0.0, 0.0)
    s0 = (0.0, x0)
    g = (0.0, P3(1.0, 0.0, 0.0))
    g_tv, logger = treeverse!(step_fun, s0, g; δ=δ, N=N)
    X = 1cm*(N+1)

    actions = copy(logger.actions)
    ptr = 1
    while length(actions)>0 && ptr<length(actions)
        if actions[ptr].action == :fetch && actions[ptr+1].action == :store && actions[ptr].step == actions[ptr+1].step
            deleteat!(actions, [ptr, ptr+1])
        elseif actions[ptr].action == :call && actions[ptr+1].action == :grad
            deleteat!(actions, ptr)
        else
            ptr += 1
        end
    end

    NY = count(a->a.action == :call, actions)+1
    Y = 1cm*NY
    Compose.set_default_graphic_size(X, Y)

    img = canvas() do
        checkpoints = []
        y = 0.5/(N+1)
        dy = 1/(N+1)
        ngrad = 1
        fstep = 0
        removed = []
        for (i, act) in enumerate(actions)
            @show act
            pebbles = checkpoints
            new = []
            if act.action == :call
                pebbles = [pebbles..., act.step+1]
                push!(removed, act.step)
                new = [act.step+1]
                fstep = act.step
            elseif act.action == :store
                push!(checkpoints, act.step)
                continue
            elseif act.action == :fetch
                pebbles = [pebbles..., act.step]
                deleteat!(checkpoints, findfirst(==(act.step), checkpoints))
                push!(removed, act.step)
                continue
            elseif act.action == :grad
                ngrad += 1
                push!(removed, act.step)
                i==length(actions) || continue
            else
                error("")
            end
            showtape(N, pebbles; y=y, removed=removed, new=new, ngrad=ngrad, flag=false, label="")
            empty!(removed)
            y += dy
        end
    end
    img = Compose.compose(context(0, 0, 1.0, (N+1)/NY), img)
    return img, NY
end

function plot_all(; fname=nothing)
    NX1 = 21
    NX2 = 17
    img1, NY1 = treeverse_pebblegame(NX1-1, 3)
    img2, NY2 = bennett_pebblegame(NX2-1, 2)
    NN = max(NY1, NY2)
    Compose.set_default_graphic_size(20cm, 20cm)
    factor = 16/20
    x0 = 0.075
    y0 = 0.115
    x1 = 0.55
    y1 = 0.115
    img = Compose.compose(context(),
        (context(), line([(x0, y0), (x0+0.15, y0)]), arrow(), stroke("black")),
        (context(), text(x0+0.03, y0-0.01, "格子 × 20"), fontsize(5)),
        (context(), line([(x0, y0), (x0, y0+0.15)]), arrow(), stroke("black")),
        (context(), text(x0-0.025, y0+0.075, "步骤 × 46", hcenter, vcenter, Rotation(-π/2,x0-0.025, y0+0.075)), fontsize(5)),
        (context(), line([(x1, y1), (x1+0.15, y1)]), arrow(), stroke("black")),
        (context(), text(x1+0.03, y1-0.01, "格子 × 16"), fontsize(5)),
        (context(), line([(x1, y1), (x1, y1+0.15)]), arrow(), stroke("black")),
        (context(), text(x1-0.025, y1+0.075, "步骤 × 41", hcenter, vcenter, Rotation(-π/2,x1-0.025, y1+0.075)), fontsize(5)),
        (context(), text(0.05, 0.07, "(a)"), fontsize(5)),
        (context(), text(0.55, 0.07, "(b)"), fontsize(5)),
        (context(0.1, y0+0.02, 0.8*NX1/NN, 0.8*NY1/NN), img1),
        (context(0.58, y0+0.02, 0.8*NX2/NN, 0.8*NY2/NN), img2),
    )
    if fname !== nothing
        img |> SVG(fname * ".svg")
        run(`rsvg-convert -f pdf -o $fname.pdf $fname.svg`)
    end
    return img
end

#plot_all()
plot_all(fname=joinpath(dirname(@__DIR__), "bennett_treeverse_pebbles"))