module Lorentz

using BenchmarkTools
using DelimitedFiles
using ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra: norm
using NiLang.AD: GVar
using NiLang
using Statistics

export P3, rk4, lorentz

include("point.jl")
include("bennett.jl")
include("treeverse.jl")
include("julia.jl")
include("reversible_programming.jl")
include("neuralode.jl")
include("benchmarking.jl")

end