include("bennett.jl")
include("treeverse.jl")

using Compose, Viznet
LW = 0.3mm

function showtape(n::Int, checkpoints; y=0.5, ngrad=0, flag=true, label="")
    unit = 1/(n+1)
    grid = nodestyle(:square, fill("gray"), stroke("transparent"), r=0.3unit)
    grid_red = nodestyle(:square, fill("red"), stroke("transparent"), r=0.3unit)
    pebble = nodestyle(:circle, fill("black"); r=0.14unit)
    tb = textstyle(:default, fontsize(44pt), fill("white"))
    pebble >> (0.5*unit, y)
    flag && tb >> (((n+0.5)*unit, y), "🚩")
    if !isempty(label)
        tb >> ((1.0, y), label)
    end
    for p in checkpoints
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
    logger = BennettLog()
    #NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=k, nsteps=N, logger=logger)[4]
    bennett_loss(0.0, lorentz_step!, zero(P3{Float64}), x0; Δt=3e-3, k=k, nsteps=N, logger=logger)
    img = canvas() do
    end
    Compose.compose(context(),
    (context(), img)
    )
end

using Compose, Viznet
function treeverse_pebblegame(N::Int, δ)
    x0 = P3(1.0, 0.0, 0.0)
    s0 = (0.0, x0)
    g = (0.0, P3(1.0, 0.0, 0.0))
    g_tv, log = treeverse!(step_fun, s0, g; δ=δ, N=N)
    Compose.set_default_graphic_size(25cm, 25cm*length(log.actions)/(N+1))

    img = canvas() do
        checkpoints = []
        y = 0.5/(N+1)
        dy = 1/(N+1)
        ngrad = 1
        fstep = 0
        actions = copy(log.actions)
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

        for (i, act) in enumerate(actions)
            @show act
            pebbles = checkpoints
            if act.action == :call
                pebbles = [pebbles..., act.step]
                fstep = act.step
            elseif act.action == :store
                push!(checkpoints, act.step)
                continue
            elseif act.action == :fetch
                pebbles = [pebbles..., act.step]
                deleteat!(checkpoints, findfirst(==(act.step), checkpoints))
                continue
            elseif act.action == :grad
                ngrad += 1
            else
                error("")
            end
            showtape(N, pebbles; y=y, ngrad=ngrad, flag=false, label="$(act.action)->$(act.step)")
            y += dy
        end
    end
    Compose.compose(context(0, 0, 1.0, (N+1)/length(log.actions)),
    (context(), img)
    )
end

function plot_all(; fname=nothing)
    Compose.set_default_graphic_size(20cm, 10cm)
    img1 = treeverse_finger_printing(binomial(3+5, 5), 3)
    img2 = bennett_finger_printing(4^3, 4)
    img = Compose.compose(context(),
        (context(), line([(0.075, 0.15), (0.225, 0.15)]), arrow(), stroke("black")),
        (context(), text(0.12, 0.12, "step"), fontsize(5)),
        (context(), line([(0.075, 0.15), (0.075, 0.4)]), arrow(), stroke("black")),
        (context(), text(0.05, 0.25, "time", hcenter, vcenter, Rotation(-π/2,0.05, 0.25)), fontsize(5)),
        (context(), text(0.05, 0.07, "(a)"), fontsize(5)),
        (context(), text(0.55, 0.07, "(b)"), fontsize(5)),
        (context(0.1, 0.25, 0.4, 0.5), img1),
        (context(0.6, 0.1, 0.4, 0.8), img2),
    )
    if fname !== nothing
        img |> SVG(fname * ".svg")
        run(`rsvg-convert -f pdf -o $fname.pdf $fname.svg`)
    end
    return img
end

treeverse_pebblegame(10, 3)

#plot_all()
#plot_all(fname=joinpath(dirname(@__DIR__), "bennett_treeverse_fingerprint"))