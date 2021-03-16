function bennett_finger_printing(N::Int, k)
    LW = 0.3mm
    logger = BennettLog()
    bennett(PlusEq(identity), 0.0, 0.0; logger=logger, k=k, N=N)
    fcalls = logger.fcalls[1:length(logger.fcalls)*4÷7]

    eb1 = bondstyle(:line, linewidth(LW), stroke("black"))
    eb2 = bondstyle(:line, linewidth(LW), stroke("green"))
    img = canvas() do
        for (depth, i, f) in fcalls
            if f isa Inv
                eb2 >> ((i-1.0, depth-0.0), (i*1.0, depth+0.0))
            else
                eb1 >> ((i-1.0, depth-0.0), (i*1.0, depth+0.0))
            end
        end
    end
    d = maximum(getindex.(fcalls, 1))
    Compose.compose(context(),
        (context(0.5/(N+1), 0.5/(1+d), 1/(N+1), 1/(1+d)), img),
        (context(), line([(0.65, 0.15), (0.75, 0.15)]), linewidth(LW), stroke("black")),
        (context(), text(0.8, 0.15, "forward", hleft, vcenter)),
        (context(), line([(0.65, 0.22), (0.75, 0.22)]), linewidth(LW), stroke("green")),
        (context(), text(0.8, 0.22,"backward", hleft, vcenter))
    )
end

function treeverse_finger_printing(N::Int, δ)
    LW = 0.3mm
    x0 = 0.0
    s0 = x0
    g = 0.0
    log = TreeverseLog()
    g_tv = treeverse(x->0.0, (x,z)->0.0, s0, g; δ=δ, N=N, logger=log)

    nb = nodestyle(:circle; r=1.5*LW)
    eb1 = bondstyle(:line, linewidth(LW))
    eb2 = bondstyle(:line, linewidth(LW), stroke("red"))
    img = canvas() do
        for act in filter(act->act.action == :call, log.actions)
            eb1 >> ((act.step, act.depth), (act.step+1, act.depth))
        end
        for act in filter(act->act.action == :store, log.actions)
            nb >> (act.step, act.depth)
        end
        for act in filter(act->act.action == :grad, log.actions)
            eb2 >> ((Float64(act.step), act.depth+0.3), (Float64(act.step+1), act.depth+0.3))
        end
    end
    τ = binomial_fit(N, δ)
    d = log.depth[]
    Compose.compose(context(),
        (context(1/(N+1), 0.5/(1+d), 1/(N+1), 1/(1+d)), img),
        (context(), line([(0.7, 0.75), (0.8, 0.75)]), linewidth(LW), stroke("black")),
        (context(), text(0.85, 0.75, "function", hleft, vcenter)),
        (context(), line([(0.7, 0.85), (0.8, 0.85)]), linewidth(LW), stroke("red")),
        (context(), text(0.85, 0.85,"gradient", hleft, vcenter))
    )
end


export plot_fingerprinting
function plot_fingerprinting(; fname=nothing)
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

#plot_fingerprinting()
#plot_fingerprinting(fname=joinpath(dirname(@__DIR__), "bennett_treeverse_fingerprint"))