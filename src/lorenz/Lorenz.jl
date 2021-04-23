module Lorenz

using ..TreeverseAndBennett: RK4, ODESolve, Glued, checkpointed_neuralode, i_ODESolve, i_ODEStep

using BenchmarkTools
using DelimitedFiles
using ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra: norm
using NiLang.AD: GVar
using NiLang
using Statistics

export P3, lorenz

include("base.jl")
include("bennett.jl")
include("treeverse.jl")
include("neuralode.jl")
include("benchmarking.jl")

end
