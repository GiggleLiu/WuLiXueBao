using TreeverseAndBennett, Test

@testset "demo" begin
    c = rand(52, 52)
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