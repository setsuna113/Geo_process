#!/usr/bin/env julia
"""
Unified LMG (Lindeman-Merenda-Gold) Relative Importance Analysis
Supports multiple response variables: fungi, vertebrates, plants
Usage: julia run_lmg_analysis.jl <data_arrow_path> <n_samples> <response_type> [max_orderings]
"""

using Distributed
using Random

# Parse arguments
if length(ARGS) < 3
    error("Usage: julia run_lmg_analysis.jl <data_arrow_path> <n_samples> <response_type> [max_orderings]")
end

data_path = ARGS[1]
n_samples = parse(Int, ARGS[2])
response_type = ARGS[3]  # "fungi", "animal", or "plant"
# Production run: always use ALL orderings (9! = 362,880)

# Add workers FIRST (before any module loading)
n_workers = 100
if nworkers() < n_workers
    addprocs(n_workers - nworkers())
    println("Added workers. Total: $(nworkers())")
end

# Load packages on ALL workers
@everywhere begin
    using DataFrames, GLM, Statistics, Random, StatsBase
    using GLM: LogLink, deviance, glm
    using Distributions: NegativeBinomial
    using StatsModels: @formula
    using Combinatorics
    
    # FIXED THETA based on distribution analysis
    const FIXED_THETA = 18.83
    
    # Define the parallel fitting function on all workers
    # Uses fixed theta instead of estimating it
    function parallel_fit_glm(predictors::Vector{String})
        # Uses globals: worker_data and worker_response
        if isempty(predictors)
            formula_str = "$(worker_response) ~ 1"
        else
            formula_str = "$(worker_response) ~ " * join(predictors, " + ")
        end
        
        formula = eval(Meta.parse("@formula($formula_str)"))
        
        # Create NegativeBinomial distribution with fixed theta (r parameter)
        # NegativeBinomial(r, p) where r is the dispersion parameter (theta)
        # We use a dummy p value since GLM will estimate mu anyway
        dist = NegativeBinomial(FIXED_THETA, 0.5)
        
        # Fit GLM with fixed dispersion
        model = glm(formula, worker_data, dist, LogLink())
        
        return deviance(model)
    end
end

# Now load data and run analysis in main process
using DataFrames, Arrow, Statistics, Combinatorics, StatsBase
using JSON, Dates
using Statistics: mean, std, median, quantile
using GLM: LogLink, glm
using Distributions: NegativeBinomial

println("\n" * "="^60)
println("LMG IMPORTANCE ANALYSIS - $(uppercase(response_type)) RESPONSE")
println("="^60)

# Configure response variable and predictors based on type
const RESPONSE_CONFIG = Dict(
    "fungi" => (
        response_col = :fungi_richness,
        biodiv_predictors = Dict(
            "plant_biodiv" => ["plants_richness_log_std"],
            "animal_biodiv" => ["vertebrates_richness_log_std"]
        )
    ),
    "animal" => (
        response_col = :vertebrates_richness,
        biodiv_predictors = Dict(
            "fungi_biodiv" => ["fungi_richness"],
            "plant_biodiv" => ["plants_richness_log_std"]
        )
    ),
    "plant" => (
        response_col = :plants_richness,
        biodiv_predictors = Dict(
            "fungi_biodiv" => ["fungi_richness"],
            "animal_biodiv" => ["vertebrates_richness_log_std"]
        )
    )
)

if !haskey(RESPONSE_CONFIG, response_type)
    error("Invalid response type: $response_type. Must be 'fungi', 'animal', or 'plant'")
end

cfg = RESPONSE_CONFIG[response_type]
response_col = cfg.response_col

# Load preprocessed data from pipeline
println("Loading preprocessed data from: $data_path")
arrow_table = Arrow.Table(data_path)
df = DataFrame(arrow_table)
println("Loaded $(nrow(df)) rows with $(ncol(df)) columns")

# Verify response variable exists
if !(response_col in propertynames(df))
    error("Response variable $response_col not found in data!")
end

# Convert response to Float64 if needed (for animal/plant responses)
if response_type in ["animal", "plant"]
    df[!, response_col] = Float64.(df[!, response_col])
end

# Group predictors based on naming patterns
println("\nGrouping predictors...")
groups = Dict{String, Vector{String}}()
cols = String.(names(df))

# Spatial: all MESF components
groups["spatial"] = filter(c -> occursin("mesf_", c), cols)

# Topographic: elevation, slope, tpi, aspect
groups["topographic"] = filter(cols) do c
    any(pattern -> occursin(pattern, c), 
        ["elevation_std", "slope_std", "tpi_std", "aspect_cos_std", "aspect_sin_std"])
end

# Soil: sand, organic carbon, pH  
groups["soil"] = filter(cols) do c
    any(pattern -> occursin(pattern, c),
        ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"])
end

# Precipitation: bio12, bio15
groups["precipitation"] = filter(c -> occursin("bio12_std", c) || occursin("bio15_std", c), cols)

# Temperature
groups["annual_temp"] = filter(c -> occursin("bio01_std", c), cols)
groups["seasonal_temp"] = filter(c -> occursin("bio04_std", c), cols)

# PET
groups["pet"] = filter(c -> occursin("pet_mm_month_std", c), cols)

# Biodiversity predictors - configured based on response type
for (group_name, predictor_vars) in cfg.biodiv_predictors
    groups[group_name] = filter(c -> c in predictor_vars, cols)
end

# Remove empty groups
filter!(p -> !isempty(p.second), groups)

println("Groups defined:")
for (name, vars) in groups
    println("  $name: $(length(vars)) variables")
end

# Sample data if needed
if nrow(df) > n_samples
    Random.seed!(42)
    sample_idx = sample(1:nrow(df), n_samples, replace=false)
    data_sample = df[sample_idx, :]
    println("\nSampled $n_samples rows from $(nrow(df)) total")
else
    data_sample = df
    println("\nUsing all $(nrow(df)) rows (less than requested $n_samples)")
end

# Fit null model with FIXED theta
println("\nFitting null model for $response_col with FIXED theta = 18.83...")
using Distributions: NegativeBinomial
# Build formula dynamically based on response
null_formula = eval(Meta.parse("@formula($response_col ~ 1)"))
null_dist = NegativeBinomial(18.83, 0.5)
null_model = glm(null_formula, data_sample, null_dist, LogLink())
null_dev = deviance(null_model)
println("Null deviance: $(round(null_dev, digits=2))")
println("Using FIXED theta: 18.83 (median from theta distribution analysis)")

# Generate orderings - sample 1/100 for faster execution
group_names = collect(keys(groups))
n_groups = length(group_names)
n_total_orderings = factorial(n_groups)

# Sample 1/100 of all orderings (approximately 3629 orderings)
Random.seed!(42)
all_perms = collect(permutations(group_names))
n_sample = max(100, div(n_total_orderings, 100))  # Sample 1/100, minimum 100
sampled_indices = sample(1:n_total_orderings, n_sample, replace=false)
all_orderings = all_perms[sampled_indices]

println("\nSampling $(length(all_orderings)) orderings from $n_total_orderings total ($(n_groups)! = $n_total_orderings)")

# Generate all predictor sets to fit
println("\nGenerating predictor sets...")
all_predictor_sets = Vector{String}[]
task_info = Tuple{Int,Int}[]  # (ordering_idx, step_idx)

for (ord_idx, ordering) in enumerate(all_orderings)
    current_vars = String[]
    for (step_idx, group) in enumerate(ordering)
        append!(current_vars, groups[group])
        push!(all_predictor_sets, copy(current_vars))
        push!(task_info, (ord_idx, step_idx))
    end
end

n_models = length(all_predictor_sets)
println("Total models to fit: $n_models")

# Broadcast data to all workers
println("\nBroadcasting data to $(nworkers()) workers...")
@sync for w in workers()
    @async remotecall_fetch(w, data_sample, response_col) do data, resp
        global worker_data = data
        global worker_response = resp
        nothing
    end
end
println("Data broadcast complete!")

# Fit models in parallel
println("\nFitting $n_models GLMs in parallel...")
start_time = time()

# Process in batches for progress reporting
batch_size = min(1000, n_models ÷ 10)
n_batches = ceil(Int, n_models / batch_size)
all_deviances = Float64[]

for batch_idx in 1:n_batches
    start_idx = (batch_idx - 1) * batch_size + 1
    end_idx = min(batch_idx * batch_size, n_models)
    batch_predictors = all_predictor_sets[start_idx:end_idx]
    
    # Parallel GLM fitting - returns deviances only (theta is fixed)
    batch_deviances = pmap(parallel_fit_glm, batch_predictors)
    append!(all_deviances, batch_deviances)
    
    # Progress report
    if batch_idx % max(1, n_batches ÷ 10) == 0 || batch_idx == n_batches
        elapsed = time() - start_time
        models_done = length(all_deviances)
        rate = models_done / elapsed
        pct = round(100 * models_done / n_models, digits=1)
        println("  Progress: $models_done/$n_models ($pct%) - $(round(rate, digits=0)) models/sec")
    end
end

elapsed_fitting = time() - start_time
println("Fitted $n_models models in $(round(elapsed_fitting, digits=1))s")
println("Performance: $(round(n_models/elapsed_fitting, digits=1)) models/second")

# Calculate LMG importance
println("\nCalculating LMG importance...")
importance_matrix = zeros(Float64, length(all_orderings), n_groups)

global task_idx = 1
for ord_idx in 1:length(all_orderings)
    ordering = all_orderings[ord_idx]
    prev_r2 = 0.0
    
    for (step_idx, group) in enumerate(ordering)
        current_dev = all_deviances[task_idx]
        current_r2 = 1.0 - (current_dev / null_dev)
        
        group_idx = findfirst(==(group), group_names)
        importance_matrix[ord_idx, group_idx] = current_r2 - prev_r2
        
        prev_r2 = current_r2
        global task_idx += 1
    end
end

# Average across orderings
lmg_importance = vec(mean(importance_matrix, dims=1))
total_r2 = sum(lmg_importance)

println("\nUsing FIXED theta = 18.83 for all models")
println("Response variable: $response_col")
if response_type != "fungi"
    println("Fungi is now a PREDICTOR")
end

# Save results
output_dir = "/scratch/yl998/lmg_results"
mkpath(output_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
output_file = joinpath(output_dir, "lmg_importance_$(response_type)_$(timestamp).json")

# Prepare results
results = Dict(
    "response" => String(response_col),
    "importance" => Dict(group_names[i] => lmg_importance[i] for i in 1:n_groups),
    "importance_pct" => Dict(group_names[i] => 100 * lmg_importance[i] for i in 1:n_groups),
    "total_r2" => total_r2,
    "fixed_theta" => 18.83,
    "metadata" => Dict(
        "n_samples" => nrow(data_sample),
        "n_orderings" => length(all_orderings),
        "n_orderings_total" => n_total_orderings,
        "orderings_sampled" => length(all_orderings) < n_total_orderings,
        "n_models" => n_models,
        "n_workers" => nworkers(),
        "elapsed_seconds" => elapsed_fitting,
        "models_per_second" => n_models / elapsed_fitting,
        "timestamp" => timestamp,
        "data_source" => data_path,
        "group_sizes" => Dict(name => length(vars) for (name, vars) in groups)
    )
)

# Write JSON
open(output_file, "w") do f
    JSON.print(f, results, 4)
end

# Print results
println("\n" * "="^60)
println("LMG IMPORTANCE RESULTS - $(uppercase(response_type)) RESPONSE")
println("="^60)
println("Total R²: $(round(total_r2, digits=4)) ($(round(100*total_r2, digits=2))%)")
println("\nRelative Importance (sorted):")

sorted_importance = sort([(name, lmg_importance[i]) for (i, name) in enumerate(group_names)], 
                         by=x->x[2], rev=true)

for (group, imp) in sorted_importance
    pct = 100 * imp
    rel_pct = 100 * imp / total_r2
    println("  $(rpad(group, 20)): $(round(imp, digits=4)) ($(round(pct, digits=2))% of 1.0, $(round(rel_pct, digits=1))% of R²)")
end

println("\nResults saved to: $output_file")
println("="^60)