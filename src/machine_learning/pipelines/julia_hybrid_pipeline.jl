#!/usr/bin/env julia
# Output structure: see docs/julia_pipeline_output_structure.md

using DataFrames, Arrow, Statistics, Distributions, JSON, StatsBase
using LinearAlgebra, YAML, JLD2, Dates, NPZ, Random
using StatsModels, HypothesisTests
using NPZ  # For saving binary numpy files
using Distributions: Normal, Chisq, cdf
using LinearAlgebra: dot, inv, cond, BLAS
using Random: MersenneTwister

# Use RidgeNegBin instead of GLM for negative binomial models
include("RidgeNegBin.jl")
using .RidgeNegBin  # Provides negbin() with ridge option
# Import GLM functions that we still need for diagnostics and LRT
using GLM: LogLink, lrtest, predict, response, modelmatrix,
           coefnames, stderror, vcov, dof_residual, nulldeviance, nullloglikelihood,
           dispersion, coeftable, confint, gvif

# Import post-processing module - separation of concerns
include("post_process.jl")
using .Main: extract_diagnostics_and_save, process_dml_results,
             save_glm_results, save_bootstrap_results, dump_analysis_data, save_dml_results

# Configure BLAS threads from config or use default
config_path = "src/config/ml/ml_config.yml"
if isfile(config_path)
    config = YAML.load_file(config_path)
    num_threads = get(get(config, "julia", Dict()), "num_threads", 120)
    ENV["JULIA_NUM_THREADS"] = string(num_threads)
else
    num_threads = 120
end
# Set BLAS to 1 to avoid competition with Julia's own parallelism
BLAS.set_num_threads(1)
println("Julia configured with $num_threads threads, BLAS threads set to: $(BLAS.get_num_threads())")

# MESF is computed by Python and cached as NPY files
# No Julia MESF module needed - just load from cache

# ============================================================================
# DATA STRUCTURES - No globals, no special cases
# ============================================================================

struct PipelineState
    data::DataFrame
    mesf_columns::Vector{String}
    config::Dict{String, Any}
    cache_dir::String
    output_dir::String
end

struct ModelResult
    name::String
    model::Any  # GLM model object
    diagnostics::Dict{String, Any}
end

# ============================================================================
# DATA LOADING - Simple and direct
# ============================================================================

# Process DataFrame with MESF - expects already sampled data
function process_with_mesf(df::DataFrame, method::String, config::Dict{String, Any})
    # Data is already sampled at load time if needed
    
    # Load MESF from Python cache BEFORE removing NaN
    # (cache is for full sampled data with NaN)
    println("Loading MESF from Python cache...")
    
    # Generate same cache key as Python: mesf_n{n}_k{enum}
    n = nrow(df)
    enum = 200  # Default from Python
    cache_key = "mesf_n$(n)_k$(enum)"
    cache_path = "/scratch/yl998/.cache/mesf/$(cache_key).npy"
    
    # Load the NPY file
    if !isfile(cache_path)
        error("MESF cache not found at: $cache_path. Run Python MESF computation first!")
    end
    
    pca_components = npzread(cache_path)  # NPZ package already imported
    println("MESF loaded from cache: $(size(pca_components)) matrix")
    
    # Check dimensions
    if size(pca_components, 1) != nrow(df)
        error("MESF dimensions mismatch: $(size(pca_components, 1)) vs $(nrow(df)) data rows")
    end
    
    # Remove rows with NaN values (keep only complete cases)
    println("Removing rows with NaN values...")
    n_before = nrow(df)
    
    # Check for NaN in all numeric columns (completecases only checks missing, not NaN!)
    complete_rows = trues(nrow(df))
    for col in propertynames(df)
        if eltype(df[!, col]) <: Union{Number, Union{Missing, <:Number}}
            # Skip missing values but check for NaN
            complete_rows .&= (ismissing.(df[!, col]) .| .!isnan.(coalesce.(df[!, col], 0.0)))
        end
    end
    
    # Apply to both DataFrame and MESF components
    df = df[complete_rows, :]
    pca_components = pca_components[complete_rows, :]  # Filter MESF to match!
    
    n_after = nrow(df)
    println("Kept $(n_after) complete rows out of $(n_before) ($(round(100*n_after/n_before, digits=1))%)")
    
    # First preprocess data (without MESF since we don't have it yet)
    println("Preprocessing data...")
    df = preprocess_data(df, method, String[], config)
    
    # Use ALL MESF components from cache (no limiting)
    n_mesf_available = size(pca_components, 2)
    
    # Add ALL MESF columns to dataframe (already standardized Float32 from cache)
    mesf_columns = String[]
    for i in 1:n_mesf_available
        col_name = "mesf_$i"
        df[!, Symbol(col_name)] = pca_components[:, i]  # Direct assignment, already Float32
        push!(mesf_columns, col_name)
    end
    
    println("Added ALL $n_mesf_available standardized MESF components from cache")
    
    return df, mesf_columns
end

function preprocess_data(df::DataFrame, method::String, mesf_columns::Vector{String}, config::Dict{String, Any})
    # Don't use dropmissing - it's too aggressive
    # We'll handle NaN values per column as needed
    
    # Convert all numeric columns to Float32 using broadcast
    println("Converting columns to Float32...")
    for col in propertynames(df)
        if eltype(df[!, col]) <: Number
            df[!, col] = Float32.(df[!, col])
        end
    end
    
    # Handle column naming: terrestrial_richness might be called vertebrates_richness
    if :terrestrial_richness in propertynames(df) && !(:vertebrates_richness in propertynames(df))
        df.vertebrates_richness = df.terrestrial_richness
        println("Using terrestrial_richness as vertebrates_richness")
    end
    
    # Create fungi_richness
    df.fungi_richness = df.am_fungi_richness .+ df.ecm_fungi_richness
    
    # Standardize all environmental variables (2SD method) using broadcast operations
    # EXACT column names from MOD44B data:
    env_vars = [:annual_temp, :temp_seasonality, :annual_precip, :precip_seasonality,
                :elevation, :pet, :sand_content_pct, :soil_organic_carbon, 
                :soil_ph, :slope, :tpi, :modis_tree_cover_pct, 
                :modis_nontree_veg_pct, :modis_bare_ground_pct]
    
    for var in env_vars
        col_std = Symbol(string(var) * "_std")
        col_data = df[!, var]
        μ = mean(col_data)
        σ = std(col_data)
        df[!, col_std] = @. (col_data - μ) / (2σ)
    end
    
    # Aspect is already decomposed into cos/sin, standardize those
    for var in [:aspect_cos, :aspect_sin]
        col_std = Symbol(string(var) * "_std")
        col_data = df[!, var]
        μ = mean(col_data)
        σ = std(col_data)
        df[!, col_std] = @. (col_data - μ) / (2σ)
    end
    
    # NOTE: MESF columns handled by MESF module (already standardized)
    # No need to standardize them here
    
    # Add elevation squared
    df.elevation_std_sq = df.elevation_std .^ 2
    
    # Method-specific transforms
    if method in ["method3", "dml"]
        # Keep raw counts for DML
        println("Keeping raw counts for DML")
        
        # Add interaction term for DML (plants × vertebrates)
        # This aligns with the paper's recommendation for 2 treatments
        interaction = df.plants_richness .* df.vertebrates_richness
        # Standardize the interaction term
        df.plants_x_vertebrates_std = (interaction .- mean(interaction)) ./ (2 .* std(interaction))
        # Keep unstandardized for backward compatibility
        df.plants_x_vertebrates = interaction
        println("Added and standardized interaction term: plants_x_vertebrates_std for DML")
    else
        # GLM needs integers for response variables
        # Create integer versions (for use as response)
        df.fungi_richness_int = round.(Int, df.fungi_richness)
        df.plants_richness_int = round.(Int, df.plants_richness)
        df.vertebrates_richness_int = round.(Int, df.vertebrates_richness)
        
        # Create standardized versions (for use as predictors)
        # Single function to handle transform + standardize (no special cases)
        transform_and_standardize(data, should_log) = begin
            transformed = should_log ? log.(data .+ 1) : data
            (transformed .- mean(transformed)) ./ (2 .* std(transformed))
        end
        
        # Check if we should log-transform before standardizing
        log_biodiv = get(get(config, "method1", Dict()), "log_biodiversity", false)
        log_biodiv && println("Using log-transformed biodiversity predictors")
        
        # Apply transformation uniformly (no branching)
        df.fungi_richness_std = transform_and_standardize(df.fungi_richness, log_biodiv)
        df.plants_richness_std = transform_and_standardize(df.plants_richness, log_biodiv)
        df.vertebrates_richness_std = transform_and_standardize(df.vertebrates_richness, log_biodiv)
        
        println("Created integer and standardized versions of biodiversity variables")
    end
    
    return df
end

# ============================================================================
# MODEL FITTING - Clean and simple
# ============================================================================

# NOTE: extract_diagnostics moved to post_process.jl for separation of concerns

function fit_method1(state::PipelineState; use_ridge::Bool=true, theta::Union{Float64, Nothing}=nothing)
    results = ModelResult[]
    
    # Get response type from config (Method1-specific)
    method1_config = get(state.config, "method1", Dict())
    response_type = get(method1_config, "response", "fungi")
    
    # Configure response variable and biotic predictors based on type
    if response_type == "fungi"
        response_var = :fungi_richness_int  # Integer version for response
        biotic = ["plants_richness_std", "vertebrates_richness_std"]  # Standardized for predictors
    elseif response_type == "plants"
        response_var = :plants_richness_int
        biotic = ["fungi_richness_std", "vertebrates_richness_std"]
    elseif response_type == "vertebrates"
        response_var = :vertebrates_richness_int
        biotic = ["fungi_richness_std", "plants_richness_std"]
    else
        error("Invalid response type for Method 1: $response_type. Must be 'fungi', 'plants', or 'vertebrates'")
    end
    
    println("Method 1: Using response variable $response_var")
    
    # Features - now includes all environmental variables
    climate = ["annual_temp_std", "temp_seasonality_std", "annual_precip_std", "precip_seasonality_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std", 
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_std"]
    vegetation = ["modis_tree_cover_pct_std", "modis_nontree_veg_pct_std", "modis_bare_ground_pct_std"]
    
    # Combine all abiotic features
    base = vcat(climate, soil, terrain, water, vegetation, state.mesf_columns)
    
    # Null model - use ridge when we have 200+ MESF components
    formula_null = Term(response_var) ~ sum(Term.(Symbol.(base)))
    model_null = negbin(formula_null, state.data, LogLink(), ridge=use_ridge, theta=theta)
    push!(results, ModelResult("method1_null", model_null, extract_diagnostics_and_save(model_null, "method1_null", state)))
    
    # Full model - use ridge when we have 200+ MESF components
    formula_full = Term(response_var) ~ sum(Term.(Symbol.(vcat(base, biotic))))
    model_full = negbin(formula_full, state.data, LogLink(), ridge=use_ridge, theta=theta)
    push!(results, ModelResult("method1_full", model_full, extract_diagnostics_and_save(model_full, "method1_full", state)))
    
    # LRT
    lrt = lrtest(model_null, model_full)
    lrt_result = Dict(
        "lr_statistic" => lrt.deviance[1] - lrt.deviance[2],  # Correct: difference in deviances
        "df" => lrt.dof[2],
        "p_value" => lrt.pval[2],
        "test_type" => "Likelihood Ratio Test",
        "response" => String(response_var)
    )
    
    println("Method 1: AIC null=$(round(aic(model_null), digits=2)), full=$(round(aic(model_full), digits=2))")
    println("LRT p-value: $(round(lrt_result["p_value"], digits=6))")
    
    return results, lrt_result
end

function fit_method2(state::PipelineState; use_ridge::Bool=true, theta::Union{Float64, Nothing}=nothing)
    results = ModelResult[]
    
    # Add categorical features
    df = state.data
    median_bio4 = median(df.temp_seasonality_std)
    df.temperate = Int.(df.temp_seasonality_std .> median_bio4)
    
    # Use standardized plants for categorical split
    median_plants = median(df.plants_richness_std)
    df.plant_high = Int.(df.plants_richness_std .> median_plants)
    
    df.temperate_x_plant_high = df.temperate .* df.plant_high
    df.temp_seasonality_std_x_plants_std = df.temp_seasonality_std .* df.plants_richness_std
    
    # Features - now includes all environmental variables
    climate = ["annual_temp_std", "temp_seasonality_std", "annual_precip_std", "precip_seasonality_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std",
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_std"]
    vegetation = ["modis_tree_cover_pct_std", "modis_nontree_veg_pct_std", "modis_bare_ground_pct_std"]
    biodiv = ["plants_richness_std", "vertebrates_richness_std"]
    
    # Combine all features
    base = vcat(climate, soil, terrain, water, vegetation, biodiv, state.mesf_columns)
    
    # Continuous interaction - use ridge when we have 200+ MESF components
    cont_features = vcat(base, ["temp_seasonality_std_x_plants_std"])
    formula_cont = Term(:fungi_richness_int) ~ sum(Term.(Symbol.(cont_features)))
    model_cont = negbin(formula_cont, df, LogLink(), ridge=use_ridge, theta=theta)
    push!(results, ModelResult("method2_continuous", model_cont, extract_diagnostics_and_save(model_cont, "method2_continuous", state)))
    
    # Categorical interaction - replace continuous bio04 and plants with categorical versions
    # Remove temp_seasonality_std and plants_richness_std from base, add categorical versions
    climate_cat = ["annual_temp_std", "annual_precip_std", "precip_seasonality_std"]  # Removed temp_seasonality_std
    biodiv_cat = ["vertebrates_richness_std"]  # Removed plants_richness_std
    
    base_cat = vcat(climate_cat, soil, terrain, water, vegetation, biodiv_cat, state.mesf_columns)
    
    # Add categorical main effects and interaction
    cat_features = vcat(
        base_cat,
        ["temperate", "plant_high", "temperate_x_plant_high"]  # Categorical main effects + interaction
    )
    formula_cat = Term(:fungi_richness_int) ~ sum(Term.(Symbol.(cat_features)))
    model_cat = negbin(formula_cat, df, LogLink(), ridge=use_ridge, theta=theta)
    push!(results, ModelResult("method2_categorical", model_cat, extract_diagnostics_and_save(model_cat, "method2_categorical", state)))
    
    println("Method 2: AIC cont=$(round(aic(model_cont), digits=2)), cat=$(round(aic(model_cat), digits=2))")
    
    return results
end

function fit_method3_dml(state::PipelineState)
    # Load DML module - use the FINAL version with proper config flow
    if !isdefined(Main, :MultivariableDML)
        include(joinpath(@__DIR__, "multivariable_dml_final.jl"))
    end
    
    # DML uses raw counts (not _int versions)
    # These are created in preprocessing for DML
    required = ["fungi_richness", "plants_richness", "vertebrates_richness"]
    for col in required
        if !(Symbol(col) in propertynames(state.data))
            error("DML requires raw counts. Reload with method='method3'")
        end
    end
    
    # Setup - now includes all environmental variables as confounders
    climate = ["annual_temp_std", "temp_seasonality_std", "annual_precip_std", "precip_seasonality_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std",
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_std"]
    vegetation = ["modis_tree_cover_pct_std", "modis_nontree_veg_pct_std", "modis_bare_ground_pct_std"]
    
    confounders = vcat(climate, soil, terrain, water, vegetation, state.mesf_columns)
    
    # Include interaction term as per DML paper recommendation for 2 treatments
    # The paper notation: A = [a, b], A° = [a×b]
    treatments = Dict(
        "plants" => "plants_richness",
        "vertebrates" => "vertebrates_richness",
        "plants_x_vertebrates" => "plants_x_vertebrates_std"  # Standardized interaction term
    )
    
    # Fit DML
    dml = MultivariableDML(n_folds=5, config_path="src/config/ml/ml_config.yml", verbose=true)
    fit_dml!(dml, state.data, confounders, treatments, "fungi_richness")
    
    # Use clean I/O module to process results
    result = process_dml_results(dml)
    
    println("Method 3 DML completed")
    
    return result
end

# NOTE: save_bootstrap_results and dump_for_r_dharma moved to post_process.jl

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

function run_pipeline(method::String;
                     mode::String="normal", bootstrap_iterations::Int=100, dump_for_dharma::Bool=false,
                     compute_lmg::Bool=false)
    # Load config
    config_path = "src/config/ml/ml_config.yml"
    config = Dict{String, Any}(string(k) => v for (k, v) in YAML.load_file(config_path))
    
    # ALWAYS read data_path and spatial_step from config
    regression_config = get(config, "regression_analysis", Dict())
    data_config = get(regression_config, "data", Dict())
    data_path = get(data_config, "data_path", nothing)
    if isnothing(data_path)
        error("data_path not found in config at regression_analysis.data.data_path!")
    end
    spatial_step = get(data_config, "spatial_grid_step", 10)
    println("Using from config: data=$data_path, step=$spatial_step")
    
    # Load data
    println("="^60)
    println("Loading data for $method")
    println("="^60)
    
    # Load data once at the pipeline level
    println("Loading data from: $data_path")
    
    # For Arrow files, we can't filter rows during read, but we can use memory mapping
    arrow_table = Arrow.Table(data_path)
    
    # If sampling needed, create row indices first
    if spatial_step > 1
        # Use first column to get row count (length(arrow_table) returns column count!)
        total_rows = length(arrow_table[1])
        row_indices = 1:spatial_step:total_rows
        println("Loading $(length(row_indices)) of $(total_rows) rows (step=$spatial_step)")
        # Create DataFrame with only selected rows
        df_raw = DataFrame(arrow_table)[row_indices, :]
    else
        total_rows = length(arrow_table[1])
        println("Loading all $(total_rows) rows")
        df_raw = DataFrame(arrow_table, copycols=false)
    end
    
    # Process with MESF
    data, mesf_columns = process_with_mesf(df_raw, method, config)
    
    # Create state
    state = PipelineState(
        data,
        mesf_columns,
        config,
        "/scratch/yl998/.cache/mesf",
        "/scratch/yl998/julia"  # Use scratch for large outputs
    )
    
    # Always dump analysis data - function handles early exit for DHARMa mode
    dump_result = dump_analysis_data(state, method, dharma_only=dump_for_dharma)
    !isnothing(dump_result) && return dump_result  # Exit if DHARMa mode
    
    # LMG IMPORTANCE ANALYSIS MODE
    if compute_lmg
        println("\n" * "="^60)
        println("RUNNING LMG IMPORTANCE ANALYSIS")
        println("="^60)
        
        # Save preprocessed data to temporary Arrow file
        temp_path = "/tmp/lmg_temp_$(round(Int, time() * 1000)).arrow"
        println("Saving preprocessed data to: $temp_path")
        
        Arrow.write(temp_path, state.data)
        println("Data saved: $(nrow(state.data)) rows, $(ncol(state.data)) columns")
        
        # Call unified LMG script - it will read all params from config
        println("Calling LMG analysis script...")
        
        lmg_script = joinpath(@__DIR__, "run_lmg_analysis.jl")
        
        println("LMG script will read all parameters from config")
        
        try
            # Only pass the preprocessed data path
            run(`julia $lmg_script $temp_path`)
            println("LMG analysis completed successfully!")
        catch e
            println("ERROR in LMG analysis: $e")
            # Clean up temp file even on error
            isfile(temp_path) && rm(temp_path)
            rethrow(e)
        end
        
        # Clean up temporary file
        rm(temp_path)
        println("Temporary data file cleaned up.")
        
        return Dict("status" => "lmg_complete")
    end
    
    # BOOTSTRAP MODE
    if mode == "bootstrap"
        println("\n" * "="^60)
        println("RUNNING BOOTSTRAP ANALYSIS MODE")
        println("="^60)
        
        # Set output directory for bootstrap results
        output_dir = joinpath(@__DIR__, "..", "..", "..", "results", "bootstrap")
        mkpath(output_dir)
        println("Bootstrap results will be saved to: $output_dir")
        
        # Load bootstrap methods module
        include("../validation/bootstrap_methods.jl")
        
        # Run bootstrap analysis using the module (use Base.invokelatest for world age issue)
        bootstrap_results = Base.invokelatest(
            Main.BootstrapMethods.run_bootstrap_analysis,
            state, method, fit_method1, fit_method2,
            n_iterations=bootstrap_iterations
        )
        
        # Save bootstrap results (pipeline handles I/O)
        save_bootstrap_results(bootstrap_results, output_dir, method)
        
        return bootstrap_results
    end
    
    # NORMAL MODE - Fit models
    all_results = Dict{String, Any}()
    all_models = Dict{String, Any}()
    
    # Get ridge configuration from YAML
    ridge_config = get(state.config, "ridge", Dict())
    use_ridge = get(ridge_config, "enabled", false)
    
    # Get negative binomial configuration from YAML
    negbin_config = get(state.config, "negbin", Dict())
    fixed_theta = get(negbin_config, "fixed_theta", nothing)
    
    if use_ridge
        println("Using Ridge regularization (automatic lambda selection via CV)")
        println("  MESF components: $(length(state.mesf_columns))")
    else
        println("Using standard GLM (no regularization)")
    end
    
    if !isnothing(fixed_theta)
        println("Using fixed theta = $fixed_theta for all models (ensures comparable pseudo R²)")
    else
        println("Estimating theta separately for each model")
    end
    
    if method == "method1"
        models, lrt = fit_method1(state, use_ridge=use_ridge, theta=fixed_theta)
        for m in models
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        all_results["method1_lr_test"] = lrt
        
    elseif method == "method2"
        # First fit method1 for comparisons
        models1, _ = fit_method1(state, use_ridge=use_ridge, theta=fixed_theta)
        for m in models1
            all_models[m.name] = m.model
        end
        
        models2 = fit_method2(state, use_ridge=use_ridge, theta=fixed_theta)
        for m in models2
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        
        # LRT comparisons - both models are now properly nested
        if haskey(all_models, "method1_full")
            # Compare continuous interaction model (nested - adds interaction term)
            lrt_a = lrtest(all_models["method1_full"], all_models["method2_continuous"])
            all_results["lrt_method1_vs_method2_continuous"] = Dict(
                "lr_statistic" => lrt_a.deviance[1] - lrt_a.deviance[2],  # Correct: difference in deviances
                "df" => lrt_a.dof[2],
                "p_value" => lrt_a.pval[2],
                "comparison" => "method1_full vs method2_continuous (adds continuous interaction)"
            )
            
            # Cannot compare categorical model - not nested (different parameterization)
            # Method2_categorical replaces continuous vars with categorical, so LRT is invalid
        end
        
    elseif method in ["method3", "dml"]
        all_results["method3_dml"] = fit_method3_dml(state)
        
    else
        error("Unknown method: $method")
    end
    
    # Save results using appropriate function based on method
    if method in ["method3", "dml"]
        save_dml_results(all_results["method3_dml"], state.output_dir)
    else
        save_glm_results(all_results, state)
    end
    
    return all_results
end

# NOTE: save_results removed - using specific save functions from post_process.jl

# ============================================================================
# CLI
# ============================================================================

if length(ARGS) > 0
    method = ARGS[1]
    
    # Check for LMG importance analysis mode
    if method == "lmg"
        # LMG mode: just verify config exists and run
        println("Running LMG importance analysis from config...")
        
        # Load config to get model and data path for preprocessing
        config_path = "src/config/ml/ml_config.yml"
        config = YAML.load_file(config_path)
        
        # Only need model for preprocessing
        # Everything else will be read from config
        if !haskey(config, "lmg_model")
            error("Missing required LMG config parameter: lmg_model")
        end
        
        model = config["lmg_model"]
        println("  Preprocessing with model: $model")
        println("  (All parameters from config)")
        
        # Pass compute_lmg=true, script will read all params from config
        run_pipeline(model, compute_lmg=true)
    
    
    # Check for dharma mode
    elseif method == "dharma"
        # DHARMa dump mode - ALL from config
        
        println("Dumping preprocessed data for DHARMa diagnostics...")
        println("  (All parameters from config)")
        
        # Run for method1/2 (they share preprocessing)
        run_pipeline("method1", dump_for_dharma=true)
        
    # Check for bootstrap mode
    elseif method == "bootstrap"
        # Bootstrap mode: bootstrap <model> <data_path> [iterations]
        if length(ARGS) < 3
            println("Usage: julia $PROGRAM_FILE bootstrap <model> <data_path> [iterations]")
            println("Models: method1, method2")
            exit(1)
        end
        
        model = ARGS[2]
        iterations = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 100
        
        println("Running bootstrap analysis...")
        println("  Model: $model")
        println("  Iterations: $iterations")
        println("  (Data path and spatial step from config)")
        
        # Bootstrap mode - ALL from config
        run_pipeline(model,
                    mode="bootstrap", bootstrap_iterations=iterations)
        
    elseif method in ["method1", "method2", "method3", "dml"]
        # Normal mode - ALL parameters from config
        
        # For method3/dml, load the DML module early
        if method in ["method3", "dml"]
            include(joinpath(@__DIR__, "multivariable_dml_final.jl"))
        end
        
        run_pipeline(method)
        
    else
        println("Usage:")
        println("  Normal: julia $PROGRAM_FILE <method>")
        println("  Bootstrap: julia $PROGRAM_FILE bootstrap <model> [iterations]")
        println("  DHARMa: julia $PROGRAM_FILE dharma")
        println("  LMG: julia $PROGRAM_FILE lmg")
        println("Methods: method1, method2, method3/dml")
        println("Note: ALL parameters (data_path, spatial_step, num_threads) from config")
    end
else
    println("Usage:")
    println("  Normal: julia $PROGRAM_FILE <method>")
    println("  Bootstrap: julia $PROGRAM_FILE bootstrap <model> [iterations]")
    println("  DHARMa: julia $PROGRAM_FILE dharma")
    println("  LMG: julia $PROGRAM_FILE lmg")
    println("Note: ALL parameters (data_path, spatial_step, num_threads) from config")
end