"""
Bootstrap Methods Module

Pure computational functions for bootstrap sample size analysis.
No I/O, no file operations - just statistics.
"""

module BootstrapMethods

using Statistics, Random, StatsBase
using GLM, HypothesisTests
using Base.Threads
using DataFrames

export run_bootstrap_analysis, find_optimal_sample_size

"""
Run bootstrap analysis to determine optimal sample size.
Returns results dictionary with all statistics.
"""
function run_bootstrap_analysis(state, method::String, fit_method1::Function, fit_method2::Function;
                               n_iterations::Int=100, sample_sizes::Vector{Int}=Int[])
    N = nrow(state.data)
    
    # Default sample sizes based on data size
    if isempty(sample_sizes)
        sample_sizes = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000,
                       30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        sample_sizes = sort(unique(filter(s -> s <= N, sample_sizes)))
    end
    
    println("Bootstrap Analysis Configuration:")
    println("  Method: $method")
    println("  Data size: $N rows")
    println("  Sample sizes: $sample_sizes")
    println("  Iterations: $n_iterations")
    println("  Threads: $(Threads.nthreads())")
    
    results = Dict{String, Any}()
    results["method"] = method
    results["N"] = N
    results["sample_sizes"] = sample_sizes
    results["n_iterations"] = n_iterations
    results["sample_results"] = Dict{Int, Any}()
    
    for n in sample_sizes
        println("\nTesting n=$n ($(round(100*n/N, digits=1))% of data)...")
        
        sample_result = bootstrap_sample_size(
            state, n, method, fit_method1, fit_method2, n_iterations
        )
        
        results["sample_results"][n] = sample_result
        
        println("  ✓ Power: $(round(sample_result["power"], digits=3))")
        println("    Eco-power: $(round(sample_result["ecological_power"], digits=3))")
    end
    
    # Find optimal sample size
    results["optimal_n"] = find_optimal_sample_size(results)
    
    println("\n" * "="^60)
    println("OPTIMAL SAMPLE SIZE: $(results["optimal_n"])")
    println("="^60)
    
    return results
end

"""
Bootstrap iterations for a specific sample size.
"""
function bootstrap_sample_size(state, n::Int, method::String, 
                              fit_method1::Function, fit_method2::Function,
                              n_iterations::Int)
    N = nrow(state.data)
    
    # Pre-allocate results arrays
    plants_effects = Vector{Float64}(undef, n_iterations)
    vertebrates_effects = Vector{Float64}(undef, n_iterations)
    p_values = Vector{Float64}(undef, n_iterations)
    r2_improvements = Vector{Float64}(undef, n_iterations)
    
    # Thread-safe bootstrap iterations
    Threads.@threads :static for i in 1:n_iterations
        # Sample indices
        rng = MersenneTwister(42 + n*1000 + i)
        boot_idx = sample(rng, 1:N, n, replace=false)
        
        # Create bootstrap state with view (10x less memory than copying)
        boot_state = typeof(state)(
            @view(state.data[boot_idx, :]),  # Use view instead of copy
            state.mesf_columns,
            state.config,
            state.cache_dir,
            state.output_dir
        )
        
        # Fit models based on method
        if method == "method1"
            models, lrt_result = fit_method1(boot_state)
            
            # Extract coefficients from full model
            full_model = models[2].model  # method1_full
            coef_vals = coef(full_model)
            coef_names = coefnames(full_model)
            plants_idx = findfirst(x -> x == "plants_richness_log", coef_names)
            vertebrates_idx = findfirst(x -> x == "vertebrates_richness_log", coef_names)
            plants_effects[i] = coef_vals[plants_idx]
            vertebrates_effects[i] = coef_vals[vertebrates_idx]
            p_values[i] = lrt_result["p_value"]
            
            # R² improvement
            null_model = models[1].model
            r2_improvements[i] = (deviance(null_model) - deviance(full_model)) / deviance(null_model)
            
        elseif method == "method2"
            models = fit_method2(boot_state)
            
            # Use continuous interaction model
            cont_model = models[1].model  # method2_continuous
            coef_vals = coef(cont_model)
            coef_names = coefnames(cont_model)
            plants_idx = findfirst(x -> x == "plants_richness_log", coef_names)
            vertebrates_idx = findfirst(x -> x == "vertebrates_richness_log", coef_names)
            plants_effects[i] = coef_vals[plants_idx]
            vertebrates_effects[i] = coef_vals[vertebrates_idx]
            
            # Compare to method1_full for p-value
            method1_models, _ = fit_method1(boot_state)
            full_model = method1_models[2].model
            lrt = lrtest(full_model, cont_model)
            p_values[i] = lrt.pval[2]
            
            r2_improvements[i] = (deviance(full_model) - deviance(cont_model)) / deviance(full_model)
            
        else
            error("Bootstrap not implemented for method: $method")
        end
    end
    
    # Calculate statistics
    return Dict(
        "plants_effect" => Dict(
            "mean" => mean(plants_effects),
            "std" => std(plants_effects),
            "var" => var(plants_effects)
        ),
        "vertebrates_effect" => Dict(
            "mean" => mean(vertebrates_effects),
            "std" => std(vertebrates_effects),
            "var" => var(vertebrates_effects)
        ),
        "power" => mean(p_values .< 0.05),
        "ecological_power" => mean((p_values .< 0.05) .& (r2_improvements .> 0.05)),
        "r2_improvement_mean" => mean(r2_improvements),
        # Raw data for visualization
        "plants_effects" => plants_effects,
        "vertebrates_effects" => vertebrates_effects,
        "p_values" => p_values,
        "r2_improvements" => r2_improvements
    )
end

"""
Find optimal sample size based on variance stabilization and power.
"""
function find_optimal_sample_size(results::Dict)
    sample_sizes = results["sample_sizes"]
    
    # Variance stabilization
    plants_vars = [results["sample_results"][n]["plants_effect"]["var"] for n in sample_sizes]
    stable_n = sample_sizes[end]
    for i in 2:length(plants_vars)
        relative_change = abs(plants_vars[i] - plants_vars[i-1]) / plants_vars[i-1]
        if relative_change < 0.1
            stable_n = sample_sizes[i]
            break
        end
    end
    
    # Power threshold
    power_n = sample_sizes[end]
    for n in sample_sizes
        if results["sample_results"][n]["ecological_power"] >= 0.8
            power_n = n
            break
        end
    end
    
    return max(stable_n, power_n)
end

end # module