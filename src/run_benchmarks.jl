function run_benchmarks(cases; output_file)
    suite = BenchmarkGroup()
    for (case, f) in cases
        suite[case] = @benchmarkable $f()
    end

    tune!(suite)
    res = run(suite)

    times = zeros(length(cases))
    for (k, (case, f)) in enumerate(cases)
        times[k] = minimum(res[case].times)
    end

    println("Writing benchmark results to file: $output_file.")
    mkpath(dirname(output_file))
    writedlm(output_file, times)
end

