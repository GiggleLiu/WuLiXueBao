using Test, TreeverseAndBennett
using Compose

@testset "lorentz" begin
    include("lorentz.jl")
end

@testset "vizsualize" begin
    plot_fingerprinting() isa Compose.Context
end