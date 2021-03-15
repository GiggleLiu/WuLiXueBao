using Test, TreeverseAndBennett
using Compose

@testset "lorentz" begin
    include("lorentz.jl")
end

@testset "vizsualize" begin
    @test plot_fingerprinting() isa Compose.Context
    @test plot_pebblegame() isa Compose.Context
end