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