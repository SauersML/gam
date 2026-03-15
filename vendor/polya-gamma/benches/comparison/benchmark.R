# benchmark.R
library(BayesLogit)
library(microbenchmark)
library(ggplot2)

# Function to benchmark a single parameter set
benchmark_pg <- function(n, b, c, num_reps = 10) {
    # Warm-up
    invisible(rpg(100, b, c))

    # Benchmark
    mb <- microbenchmark(
        rpg(n, b, c),
        times = num_reps,
        unit = "us"
    )

    # Return summary
    data.frame(
        n = n,
        b = b,
        c = c,
        mean_us = mean(mb$time / 1000), # Convert to microseconds
        median_us = median(mb$time / 1000),
        min_us = min(mb$time / 1000),
        max_us = max(mb$time / 1000),
        stddev_us = sd(mb$time / 1000)
    )
}

# Function to run all benchmarks
benchmark_pg_all <- function() {
    # Define parameter combinations to test
    params <- expand.grid(
        n = 10000, # Fixed number of samples
        b = c(0.5, 1.0, 5.0, 10.0), # Shape parameter
        c = c(0.0, 0.5, 1.0, 2.0), # Tilt parameter
        stringsAsFactors = FALSE
    )

    # Run all benchmarks
    results <- do.call(rbind, lapply(1:nrow(params), function(i) {
        with(params[i, ], {
            cat(sprintf("Benchmarking n=%d, b=%.1f, c=%.1f...\n", n, b, c))
            benchmark_pg(n, b, c)
        })
    }))

    return(results)
}

# Run if called directly
if (!interactive()) {
    results <- benchmark_pg_all()
    write.csv(results, "benches/comparison/results/bayeslogit_results.csv", row.names = FALSE)
}
