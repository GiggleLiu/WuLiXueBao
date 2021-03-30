@i function lorentz_step!(y!::T, y::T, θ; Δt) where T
    i_ODEStep((@skip! RK4()), (@skip! lorentz!), y!, y, θ; Δt, t=0.0)
end

@i function bennett_loss(out, step, y, x, θ; kwargs...)
    bennett((@skip! step), y, x, θ; kwargs...)
    out += y.x
end

using Compose, Viznet
function bennett_finger_printing(N::Int, k)
    x0 = P3(1.0, 0.0, 0.0)
    θ = (0.0, 0.0, 0.0)
    logger = BennettLog()
    #NiLang.AD.gradient(bennett_loss, (0.0, lorentz_step!, zero(P3{Float64}), x0); iloss=1, Δt=3e-3, k=k, N=N, logger=logger)[4]
    bennett_loss(0.0, lorentz_step!, zero(P3{Float64}), x0, θ; Δt=3e-3, k=k, N=N, logger=logger)
    fcalls = logger.fcalls[1:length(logger.fcalls)*4÷7]

    eb1 = bondstyle(:line, linewidth(0.1mm), stroke("red"))
    eb2 = bondstyle(:line, linewidth(0.1mm), stroke("green"))
    Compose.set_default_graphic_size(15cm, 15cm)
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
    Compose.compose(context(0.5/(N+1), 0.5/(1+d), 1/(N+1), 1/(1+d)), img)
end

#bennett_finger_printing(4^4, 4)
#x |> SVG(fname * ".svg")
#run(`rsvg-convert -f pdf -o $fname.pdf $fname.svg`)