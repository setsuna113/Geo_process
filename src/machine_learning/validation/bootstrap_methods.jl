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
        if haskey(sample_result, "response_var")
            println("    Response: $(sample_result["response_var"])")
        end
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
    
    # Get response variable from config (for Method1)
    response_var = Symbol(get(get(state.config, "method1", Dict()), "response", "fungi_richness_log"))
    
    # Determine which biotic predictors to track based on response
    if response_var == :fungi_richness_log
        biotic1_name = "plants_richness_log"
        biotic2_name = "vertebrates_richness_log"
    elseif response_var == :plants_richness_log
        biotic1_name = "fungi_richness_log"
        biotic2_name = "vertebrates_richness_log"
    else  # vertebrates_richness_log
        biotic1_name = "fungi_richness_log"
        biotic2_name = "plants_richness_log"
    end
    
    # Pre-allocate results arrays (renamed for clarity)
    biotic1_effects = Vector{Float64}(undef, n_iterations)
    biotic2_effects = Vector{Float64}(undef, n_iterations)
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
            biotic1_idx = findfirst(x -> x == biotic1_name, coef_names)
            biotic2_idx = findfirst(x -> x == biotic2_name, coef_names)
            biotic1_effects[i] = coef_vals[biotic1_idx]
            biotic2_effects[i] = coef_vals[biotic2_idx]
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
            biotic1_idx = findfirst(x -> x == biotic1_name, coef_names)
            biotic2_idx = findfirst(x -> x == biotic2_name, coef_names)
            biotic1_effects[i] = coef_vals[biotic1_idx]
            biotic2_effects[i] = coef_vals[biotic2_idx]
            
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
    
    # Calculate statistics - ONE TRUE FORMAT
    return Dict(
        "biotic1_name" => biotic1_name,
        "biotic2_name" => biotic2_name,
        "response_var" => String(response_var),
        "biotic1_effect" => Dict(
            "mean" => mean(biotic1_effects),
            "std" => std(biotic1_effects),
            "var" => var(biotic1_effects)
        ),
        "biotic2_effect" => Dict(
            "mean" => mean(biotic2_effects),
            "std" => std(biotic2_effects),
            "var" => var(biotic2_effects)
        ),
        "power" => mean(p_values .< 0.05),
        "ecological_power" => mean((p_values .< 0.05) .& (r2_improvements .> 0.05)),
        "r2_improvement_mean" => mean(r2_improvements),
        # Raw data for visualization
        "biotic1_effects" => biotic1_effects,
        "biotic2_effects" => biotic2_effects,
        "p_values" => p_values,
        "r2_improvements" => r2_improvements
    )
end

"""
Find optimal sample size based on variance stabilization and power.
"""
function find_optimal_sample_size(results::Dict)
    sample_sizes = results["sample_sizes"]
    
    # Use the first biotic predictor for variance stabilization
    # (it exists regardless of which response variable is used)
    first_sample_result = results["sample_results"][sample_sizes[1]]
    biotic1_key = haskey(first_sample_result, "biotic1_effect") ? "biotic1_effect" : "plants_effect"
    
    # Variance stabilization
    biotic1_vars = [results["sample_results"][n][biotic1_key]["var"] for n in sample_sizes]
    stable_n = sample_sizes[end]
    for i in 2:length(biotic1_vars)
        relative_change = abs(biotic1_vars[i] - biotic1_vars[i-1]) / biotic1_vars[i-1]
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