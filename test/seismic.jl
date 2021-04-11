using TreeverseAndBennett, Test
using TreeverseAndBennett.Seismic

@testset "demo" begin
    n = 52
    c = rand(n, n)
    f_nilang = Seismic.generate_useq(c, nstep=1000, method=:nilang)
    f_julia = Seismic.generate_useq(c, nstep=1000, method=:julia)
    f_bennett = Seismic.generate_useq(c, nstep=1000, method=:bennett, bennett_k=50)
    @test f_nilang ≈ f_julia
    @test f_bennett ≈ f_julia[:,:,end]
    g_nilang = Seismic.getgrad(c, nstep=1000, method=:nilang)
    g_treeverse = Seismic.getgrad(c, nstep=1000, method=:treeverse)
    g_bennett = Seismic.getgrad(c, nstep=1000, method=:bennett)
    @test g_nilang[1] ≈ g_treeverse[1]
    @test g_nilang[2] ≈ g_treeverse[2]
    @test g_nilang[3] ≈ g_treeverse[3]
    @test g_nilang[1] ≈ g_bennett[1]
    @test g_nilang[2] ≈ g_bennett[2]
    @test g_nilang[3] ≈ g_bennett[3]
end

@testset "detector" begin
    p1, g1 = getgrad_three_layer(method=:treeverse)
    p2, g2 = getgrad_three_layer(method=:bennett)
    @test g1[1] ≈ g2[1]
    @test g1[2] ≈ g2[2]
    @test g1[3] ≈ g2[3]
end