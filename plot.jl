### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ df10bbe8-74af-11eb-28f0-c95d4bf59abe
using Revise, Viznet, Compose, Yao, YaoPlots

# ╔═╡ 33aa3a4c-74bc-11eb-0464-b7381f7e4d87
#= html"""
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script>
    MathJax = {
      startup: {
        ready: function () {
          var math = MathJax._.core.MmlTree.MmlNodes.math.MmlMath;
          math.defaults.scriptminsize = '0px';
          MathJax.startup.defaultReady();
        }
      }
    };
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>
"""
=#
function savefig(str)
	function (x)
		fname = joinpath(@__DIR__, str)
		x |> SVG(fname * ".svg")
		run(`rsvg-convert -f pdf -o $fname.pdf $fname.svg`)
		return x
	end
end

# ╔═╡ 0d42b178-74ca-11eb-1585-b3cbfc0eb864
begin
	lb = textstyle(:math, fontsize(10), width=1.0, height=0.5)
	tb = textstyle(:default, fontsize(4.4), font("monospace"))
	tb_big = textstyle(:default, fontsize(5), fill("white"), font("monospace"))
	nb = nodestyle(:circle, fill("black"), stroke("transparent"); r=0.05)
	tri = nodestyle(:triangle, stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	
	x_x = (0.1, 0.2)
	x_y = (0.9, 0.5)
	x_z = (0.1, 0.7)
	x_sin = (0.3, 0.3)
	x_mul = (0.5, 0.5)
end

# ╔═╡ 5af89c0e-74d2-11eb-0b7b-6773678516ad
function arrow(x, y)
	mid = (x .+ y) ./ 2
	t = nodestyle(:triangle, θ=π/2-atan((y .- x)...)-1π/6)
	eb >> (x, y)
	t >> mid
end

# ╔═╡ 20c8b7c0-74b0-11eb-365b-ade866a50925
# forwarddiff
canvas() do
	Compose.set_default_graphic_size(100mm, 100mm)
	nb >> x_sin
	nb >> x_mul
	tb_big >> (x_sin, "sin")
	tb_big >> (x_mul .+ (0, 0.01), "*")
	arrow(x_sin, x_mul)
	arrow(x_x, x_sin)
	arrow(x_mul, x_y)
	arrow(x_z, x_mul)
	tb >> ((x_x .+ x_sin) ./ 2 .- (0.02, 0.06), "x+ϵˣ")
	tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.22, 0.04), "sin(x)+cos(x)*ϵˣ")
	tb >> ((x_y .+ x_mul) ./ 2 .- (-0.04, 0.055), "z*sin(x)+sin(x)*ϵᶻ\n+z*cos(x)*ϵˣ")
	tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.02), "z+ϵᶻ")
end |> savefig("forwarddiff")

# ╔═╡ f3efbf02-74cb-11eb-2ce9-89fe0320c00c
# normal
canvas() do
	Compose.set_default_graphic_size(100mm, 100mm)
	nb >> x_sin
	nb >> x_mul
	tb_big >> (x_sin, "sin")
	tb_big >> (x_mul .+ (0, 0.01), "*")
	arrow(x_sin, x_mul)
	arrow(x_x, x_sin)
	arrow(x_mul, x_y)
	arrow(x_z, x_mul)
	tb >> ((x_x .+ x_sin) ./ 2 .- (0.02, 0.06), "x")
	tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.08, 0.04), "sin(x)")
	tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.04), "z*sin(x)")
	tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.04), "z")
end

# ╔═╡ 2f7b1a04-74cd-11eb-001d-25db5400d5a5
# backward-forward
canvas() do
	Compose.set_default_graphic_size(100mm, 100mm)
	nb >> x_sin
	nb >> x_mul
	tb_big >> (x_sin, "sin")
	tb_big >> (x_mul .+ (0, 0.01), "*")
	arrow(x_sin, x_mul)
	arrow(x_x, x_sin)
	arrow(x_mul, x_y)
	arrow(x_z, x_mul)
	tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x \n push(Σ,x)")
	tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.15, 0.04), "s = sin(x) \n push(Σ,s)")
	tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.04), "y = z*sin(x)")
	tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "z\n push(Σ,z)")
end |> savefig("backward-forward")

# ╔═╡ 88fcfeba-74cd-11eb-00ba-9d75a97a3034
# backward-backward
canvas() do
	Compose.set_default_graphic_size(100mm, 100mm)
	nb >> x_sin
	nb >> x_mul
	tb_big >> (x_sin, "sin")
	tb_big >> (x_mul .+ (0, 0.01), "*")
	arrow(x_mul, x_sin)
	arrow(x_sin, x_x)
	arrow(x_y, x_mul)
	arrow(x_mul, x_z)
	tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x = pop(Σ)\nx̄ = cos(x)*s̄")
	tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.12, 0.04), "s = pop(Σ)\ns̄ = z*ȳ")
	tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.06), "y\nȳ=1")
	tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "z = pop(Σ)\nz̄ = s*ȳ")
end |> savefig("backward-backward")

# ╔═╡ 712fc8b0-74cf-11eb-3454-4fb7f1ef7b1c
# nilang-forward
let
	Compose.set_default_graphic_size(100mm, 100mm)
	c = chain(cnot(4,1,2), cnot(4,(2,3),4))
	compose(context(),
		(context(), canvas() do
				tb >> ((0.18, 0.04), "x")
				tb >> ((0.18, 0.27), "0")
				tb >> ((0.18, 0.5), "z")
				tb >> ((0.18, 0.72), "0")
				
				tb >> ((0.85, 0.27), "s")
				tb >> ((0.85, 0.04), "x")
				tb >> ((0.85, 0.5), "z")
				tb >> ((0.85, 0.72), "y")
				tb >> ((0.45, 0.88), "sin")
				tb >> ((0.68, 0.9), "*")
		end),
		(context(0.1, 0.02, 0.9, 0.9), vizcircuit(c; show_ending=false))
		)
end |> savefig("nilang-forward")

# ╔═╡ fa5830de-74dc-11eb-29a3-6de38399dd5f
# nilang-forward
let
	Compose.set_default_graphic_size(100mm, 100mm)
	c = chain(cnot(4,1,2), cnot(4,(2,3),4))
	compose(context(),
		(context(), canvas() do
				tb >> ((0.2, 0.04), "(x,z*cos(x))")
				tb >> ((0.2, 0.27), "(0,z)")
				tb >> ((0.2, 0.5), "(z,sin(x))")
				tb >> ((0.2, 0.72), "(0,1)")
				
				tb >> ((0.92, 0.27), "(s,0)")
				tb >> ((0.7, 0.27), "(s,z)")
				tb >> ((0.92, 0.04), "(x,0)")
				tb >> ((0.92, 0.5), "(z,0)")
				tb >> ((0.92, 0.72), "(y,1)")
				tb >> ((0.55, 0.88), "sin")
				tb >> ((0.78, 0.9), "*")
		end),
		(context(0.20, 0.02, 0.9, 0.9), vizcircuit(c; show_ending=false))
		)
end |> savefig("nilang-backward")

# ╔═╡ Cell order:
# ╠═df10bbe8-74af-11eb-28f0-c95d4bf59abe
# ╠═33aa3a4c-74bc-11eb-0464-b7381f7e4d87
# ╠═5af89c0e-74d2-11eb-0b7b-6773678516ad
# ╠═0d42b178-74ca-11eb-1585-b3cbfc0eb864
# ╠═20c8b7c0-74b0-11eb-365b-ade866a50925
# ╠═f3efbf02-74cb-11eb-2ce9-89fe0320c00c
# ╠═2f7b1a04-74cd-11eb-001d-25db5400d5a5
# ╠═88fcfeba-74cd-11eb-00ba-9d75a97a3034
# ╠═712fc8b0-74cf-11eb-3454-4fb7f1ef7b1c
# ╠═fa5830de-74dc-11eb-29a3-6de38399dd5f
