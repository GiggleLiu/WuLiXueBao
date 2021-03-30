using Test, TreeverseAndBennett
using Compose, Pkg

@testset "lorenz" begin
    include("lorenz.jl")
end

@testset "seismic" begin
    include("seismic.jl")
end

@testset "vizsualize" begin
    @test plot_fingerprinting() isa Compose.Context
    @test plot_pebblegame() isa Compose.Context
end

function isinstalled(target)
    deps = Pkg.dependencies()
    for (uuid, dep) in deps
        dep.is_direct_dep || continue
        dep.name == target && return true
    end
    return false
end

if isinstalled("KernelAbstractions")
    @testset "cuda" begin
        include("cuda.jl")
    end
end
