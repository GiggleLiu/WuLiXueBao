using TreeverseAndBennett, Test, CUDA
CUDA.allowscalar(false)

@testset "demo" begin
    n = 52
    c = rand(n, n)
    g_nilang = Seismic.getgrad(c, nstep=1000, method=:nilang, usecuda=false)
    g_treeverse = Seismic.getgrad(c, nstep=1000, method=:treeverse, usecuda=true)
    g_bennett = Seismic.getgrad(c, nstep=1000, method=:bennett, usecuda=true)

    cg_treeverse = Seismic.getgrad(c, nstep=1000, method=:treeverse, usecuda=false)
    cg_bennett = Seismic.getgrad(c, nstep=1000, method=:bennett, usecuda=false)
    for i=1:3
        @test Array(g_treeverse[i]) ≈ Array(cg_treeverse[i])
    end
    for i=1:3
        @test Array(g_bennett[i]) ≈ Array(cg_bennett[i])
    end
    for i=1:3
        @test Array(g_nilang[i]) ≈ Array(g_treeverse[i])
        @test Array(g_nilang[i]) ≈ Array(g_bennett[i])
    end
end

@testset "detector" begin
    p1, g1 = getgrad_three_layer(method=:treeverse)
    p2, g2 = getgrad_three_layer(method=:treeverse, usecuda=true)
    @test g2[1] isa CuArray
    @test g1[1] ≈ Array(g2[1])
    @test g1[2] ≈ Array(g2[2])
    @test g1[3] ≈ Array(g2[3])
end

@testset "detector" begin
    p1, g1 = getgrad_three_layer(method=:bennett, usecuda=false)
    p2, g2 = getgrad_three_layer(method=:bennett, usecuda=true)
    @test g2[1] isa CuArray
    @test g1[1] ≈ Array(g2[1])
    @test g1[2] ≈ Array(g2[2])
    @test g1[3] ≈ Array(g2[3])
end