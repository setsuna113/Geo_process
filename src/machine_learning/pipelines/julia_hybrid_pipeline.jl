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

# Configure BLAS threads for optimal performance
# Set to 1 to avoid competition with Julia's own parallelism
BLAS.set_num_threads(1)
println("BLAS threads set to: $(BLAS.get_num_threads())")

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
function process_with_mesf(df::DataFrame, method::String)
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
    
    # Now remove rows with NaN values in any column (keep only complete cases)
    println("Removing rows with NaN values...")
    n_before = nrow(df)
    
    # Check each column for NaN and keep only rows without any NaN
    mask = trues(n_before)
    for col in propertynames(df)
        col_data = df[!, col]
        # Handle both regular numbers and Union{Missing, Number} types
        if eltype(col_data) <: Number || eltype(col_data) <: Union{Missing, <:Number}
            # Use a more robust NaN check that handles Union types
            for i in 1:length(col_data)
                val = col_data[i]
                if !ismissing(val) && isnan(val)
                    mask[i] = false
                end
            end
        end
    end
    
    # Apply mask to both DataFrame and MESF components
    df = df[mask, :]
    pca_components = pca_components[mask, :]  # Filter MESF to match!
    
    n_after = nrow(df)
    println("Kept $(n_after) complete rows out of $(n_before) ($(round(100*n_after/n_before, digits=1))%)")
    
    # First preprocess data (without MESF since we don't have it yet)
    println("Preprocessing data...")
    df = preprocess_data(df, method, String[])
    
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

function preprocess_data(df::DataFrame, method::String, mesf_columns::Vector{String})
    # Don't use dropmissing - it's too aggressive
    # We'll handle NaN values per column as needed
    
    # Convert all numeric columns to Float32 using broadcast (more efficient)
    println("Converting columns to Float32...")
    for col in propertynames(df)
        if eltype(df[!, col]) <: Number
            # Use in-place broadcast for better memory efficiency
            df[!, col] = Float32.(df[!, col])
        end
    end
    
    # Create fungi_richness (no NaN handling needed - already removed)
    df.fungi_richness = df.am_fungi_richness .+ df.ecm_fungi_richness
    
    # Standardize naming
    if :plant_richness in propertynames(df)
        df.plants_richness = df.plant_richness
    end
    if :animal_richness in propertynames(df)
        df.vertebrates_richness = df.animal_richness
    elseif :terrestrial_richness in propertynames(df)
        df.vertebrates_richness = df.terrestrial_richness
    end
    
    # Standardize all environmental variables (2SD method) using broadcast operations
    env_vars = [:bio01, :bio04, :bio12, :bio15, :elevation,
                :pet_mm_month, :sand_content_pct, :soil_organic_carbon, 
                :soil_ph, :slope, :tpi]
    
    for var in env_vars
        if var in propertynames(df)
            col_std = Symbol(string(var) * "_std")
            col_data = df[!, var]
            μ = mean(col_data)  # No NaN - already removed
            σ = std(col_data)
            # Single broadcast operation for standardization
            df[!, col_std] = @. (col_data - μ) / (2σ)
        end
    end
    
    # Aspect is already decomposed into cos/sin, standardize those
    for var in [:aspect_cos, :aspect_sin]
        if var in propertynames(df)
            col_std = Symbol(string(var) * "_std")
            col_data = df[!, var]
            μ = mean(col_data)
            σ = std(col_data)
            df[!, col_std] = @. (col_data - μ) / (2σ)
        end
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
        # GLM needs integers and log transforms
        df.fungi_richness = round.(Int, df.fungi_richness)
        
        # Log transform first
        plants_log = log.(df.plants_richness .+ 1)
        vertebrates_log = log.(df.vertebrates_richness .+ 1)
        
        # Then standardize using 2SD method for fair comparison
        df.plants_richness_log_std = (plants_log .- mean(plants_log)) ./ (2 .* std(plants_log))
        df.vertebrates_richness_log_std = (vertebrates_log .- mean(vertebrates_log)) ./ (2 .* std(vertebrates_log))
        
        # Keep unstandardized versions for backward compatibility
        df.plants_richness_log = plants_log
        df.vertebrates_richness_log = vertebrates_log
        
        println("Log-transformed and standardized biotic predictors (2SD method)")
    end
    
    return df
end

# ============================================================================
# MODEL FITTING - Clean and simple
# ============================================================================

function extract_diagnostics(model, model_name::String, state::PipelineState)
    # Handle both standard GLM and RidgeNegBinModel wrapper
    if isa(model, RidgeNegBinModel)
        actual_model = model.model
        θ = actual_model.model.rr.d.r
    else
        actual_model = model
        θ = model.model.rr.d.r
    end
    
    # Calculate residuals but DON'T put in JSON
    y = response(actual_model)
    μ = predict(actual_model)
    # θ already extracted above
    
    response_resid = y .- μ
    deviance_resid = sign.(y .- μ) .* sqrt.(2 .* (
        y .* log.((y .+ 1e-10) ./ μ) .- (y .+ θ) .* log.((y .+ θ) ./ (μ .+ θ))
    ))
    
    # Save residuals to binary file (NPY format)
    residuals_file = joinpath(state.output_dir, "$(model_name)_residuals.npz")
    npzwrite(residuals_file, Dict(
        "y_values" => Float32.(y),
        "fitted_values" => Float32.(μ),
        "response_residuals" => Float32.(response_resid),
        "deviance_residuals" => Float32.(deviance_resid)
    ))
    
    # Only return summary stats in JSON
    diagnostics = Dict(
        # Core stats (keep these - they're small)
        "coefficients" => coef(actual_model),
        "coef_names" => coefnames(actual_model),
        "std_errors" => stderror(actual_model),
        "vcov" => vcov(actual_model),
        "aic" => aic(model),  # Use wrapper for AIC (handles ridge)
        "bic" => bic(model),  # Use wrapper for BIC (handles ridge)
        "loglik" => loglikelihood(actual_model),
        "deviance" => deviance(actual_model),
        "df_residual" => dof_residual(actual_model),
        "n_obs" => nobs(actual_model),
        "theta" => θ,
        
        # Residual summaries only (not full arrays)
        "residuals_file" => residuals_file,
        "residuals_summary" => Dict(
            "mean_response" => mean(response_resid),
            "std_response" => std(response_resid),
            "mean_deviance" => mean(deviance_resid),
            "std_deviance" => std(deviance_resid),
            "min_deviance" => minimum(deviance_resid),
            "max_deviance" => maximum(deviance_resid)
        ),
        
        # Extended stats (NEW - prefixed)
        "null_deviance" => nulldeviance(actual_model),
        "null_loglik" => nullloglikelihood(actual_model),
        "dispersion" => dispersion(actual_model.model)  # Need the actual GLM, not wrapper
    )
    
    # Inference (NEW - flat with prefixes)
    try
        ct = coeftable(actual_model)
        diagnostics["coef_z_values"] = ct.cols[3]
        diagnostics["coef_p_values"] = ct.cols[4]
        diagnostics["confint_95"] = confint(actual_model, 0.95)
    catch e
        # If inference fails, just skip those fields
        println("Note: Inference stats not available: $e")
    end
    
    # Multicollinearity (NEW - flat with prefixes)
    try
        vif_values = gvif(actual_model)
        diagnostics["gvif"] = vif_values
        diagnostics["gvif_max"] = maximum(vif_values)
        if maximum(vif_values) > 10
            println("⚠ High multicollinearity: max GVIF = $(maximum(vif_values))")
        end
    catch e
        # If GVIF fails, skip
        println("Note: GVIF not available: $e")
    end
    
    # Diagnostic tests (NEW - flat with prefixes)
    try
        # Use the deviance_resid we already calculated above
        dev_resid = deviance_resid
        
        # Jarque-Bera normality test
        jb = JarqueBeraTest(dev_resid)
        diagnostics["test_jarque_bera_stat"] = jb.JB
        diagnostics["test_jarque_bera_p"] = pvalue(jb)
        
        # Outlier detection (for NB GLM: |deviance residual| > 3)
        outlier_threshold = 3.0
        n_outliers = sum(abs.(dev_resid) .> outlier_threshold)
        diagnostics["test_outliers_count"] = n_outliers
        diagnostics["test_outliers_percent"] = 100.0 * n_outliers / length(dev_resid)
        diagnostics["test_outliers_threshold"] = outlier_threshold
        
        # Heteroscedasticity check via residual variance in bins
        # For NB GLM: check if variance changes with fitted values
        n_bins = min(10, Int(ceil(sqrt(length(μ)))))
        sorted_idx = sortperm(μ)
        bin_size = length(μ) ÷ n_bins
        bin_variances = Float64[]
        for i in 1:n_bins
            start_idx = (i-1) * bin_size + 1
            end_idx = i == n_bins ? length(μ) : i * bin_size
            bin_resids = dev_resid[sorted_idx[start_idx:end_idx]]
            push!(bin_variances, var(bin_resids))
        end
        # Coefficient of variation of bin variances (should be low for homoscedasticity)
        heterosced_cv = std(bin_variances) / mean(bin_variances)
        diagnostics["test_heteroscedasticity_cv"] = heterosced_cv
        diagnostics["test_heteroscedasticity_high"] = heterosced_cv > 0.5
        
        # Skewness and Kurtosis of deviance residuals
        # For well-fitted NB GLM, deviance residuals should be approximately normal
        resid_skewness = StatsBase.skewness(dev_resid)
        resid_kurtosis = StatsBase.kurtosis(dev_resid)  # Excess kurtosis (0 for normal)
        diagnostics["test_skewness"] = resid_skewness
        diagnostics["test_kurtosis_excess"] = resid_kurtosis
        diagnostics["test_skewness_high"] = abs(resid_skewness) > 1.0
        diagnostics["test_kurtosis_high"] = abs(resid_kurtosis) > 3.0
    catch e
        # If tests fail, skip
        println("Note: Diagnostic tests not available: $e")
    end
    
    return diagnostics
end

function fit_method1(state::PipelineState; use_ridge::Bool=true)
    results = ModelResult[]
    
    # Features - now includes all environmental variables
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std", 
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_mm_month_std"]
    
    # Combine all abiotic features
    base = vcat(climate, soil, terrain, water, state.mesf_columns)
    biotic = ["plants_richness_log_std", "vertebrates_richness_log_std"]
    
    # Null model - use ridge when we have 200+ MESF components
    formula_null = Term(:fungi_richness) ~ sum(Term.(Symbol.(base)))
    model_null = negbin(formula_null, state.data, LogLink(), ridge=use_ridge)
    push!(results, ModelResult("method1_null", model_null, extract_diagnostics(model_null, "method1_null", state)))
    
    # Full model - use ridge when we have 200+ MESF components
    formula_full = Term(:fungi_richness) ~ sum(Term.(Symbol.(vcat(base, biotic))))
    model_full = negbin(formula_full, state.data, LogLink(), ridge=use_ridge)
    push!(results, ModelResult("method1_full", model_full, extract_diagnostics(model_full, "method1_full", state)))
    
    # LRT
    lrt = lrtest(model_null, model_full)
    lrt_result = Dict(
        "lr_statistic" => lrt.deviance[2],
        "df" => lrt.dof[2],
        "p_value" => lrt.pval[2],
        "test_type" => "Likelihood Ratio Test"
    )
    
    println("Method 1: AIC null=$(round(aic(model_null), digits=2)), full=$(round(aic(model_full), digits=2))")
    println("LRT p-value: $(round(lrt_result["p_value"], digits=6))")
    
    return results, lrt_result
end

function fit_method2(state::PipelineState; use_ridge::Bool=true)
    results = ModelResult[]
    
    # Add categorical features
    df = state.data
    median_bio4 = median(df.bio04_std)
    df.temperate = Int.(df.bio04_std .> median_bio4)
    
    # FIXED: Use log-transformed plants to match continuous model
    median_plants = median(df.plants_richness_log_std)
    df.plant_high = Int.(df.plants_richness_log_std .> median_plants)
    
    df.temperate_x_plant_high = df.temperate .* df.plant_high
    df.bio04_std_x_plants_log_std = df.bio04_std .* df.plants_richness_log_std
    
    # Features - now includes all environmental variables
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std",
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_mm_month_std"]
    biodiv = ["plants_richness_log_std", "vertebrates_richness_log_std"]
    
    # Combine all features
    base = vcat(climate, soil, terrain, water, biodiv, state.mesf_columns)
    
    # Continuous interaction - use ridge when we have 200+ MESF components
    cont_features = vcat(base, ["bio04_std_x_plants_log_std"])
    formula_cont = Term(:fungi_richness) ~ sum(Term.(Symbol.(cont_features)))
    model_cont = negbin(formula_cont, df, LogLink(), ridge=use_ridge)
    push!(results, ModelResult("method2_continuous", model_cont, extract_diagnostics(model_cont, "method2_continuous", state)))
    
    # Categorical interaction - replace continuous bio04 and plants with categorical versions
    # Remove bio04_std and plants_richness_log from base, add categorical versions
    climate_cat = ["bio01_std", "bio12_std", "bio15_std"]  # Removed bio04_std
    biodiv_cat = ["vertebrates_richness_log_std"]  # Removed plants_richness_log
    
    base_cat = vcat(climate_cat, soil, terrain, water, biodiv_cat, state.mesf_columns)
    
    # Add categorical main effects and interaction
    cat_features = vcat(
        base_cat,
        ["temperate", "plant_high", "temperate_x_plant_high"]  # Categorical main effects + interaction
    )
    formula_cat = Term(:fungi_richness) ~ sum(Term.(Symbol.(cat_features)))
    model_cat = negbin(formula_cat, df, LogLink(), ridge=use_ridge)
    push!(results, ModelResult("method2_categorical", model_cat, extract_diagnostics(model_cat, "method2_categorical", state)))
    
    println("Method 2: AIC cont=$(round(aic(model_cont), digits=2)), cat=$(round(aic(model_cat), digits=2))")
    
    return results
end

function fit_method3_dml(state::PipelineState)
    # Load DML module - use the FINAL version with proper config flow
    if !isdefined(Main, :MultivariableDML)
        include(joinpath(@__DIR__, "multivariable_dml_final.jl"))
    end
    
    # Verify raw counts
    required = ["fungi_richness", "plants_richness", "vertebrates_richness"]
    for col in required
        if !(Symbol(col) in propertynames(state.data))
            error("DML requires raw counts. Reload with method='method3'")
        end
    end
    
    # Setup - now includes all environmental variables as confounders
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    soil = ["sand_content_pct_std", "soil_organic_carbon_std", "soil_ph_std"]
    terrain = ["elevation_std", "elevation_std_sq", "slope_std", "tpi_std",
               "aspect_cos_std", "aspect_sin_std"]
    water = ["pet_mm_month_std"]
    
    confounders = vcat(climate, soil, terrain, water, state.mesf_columns)
    
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
    
    # Compute comprehensive statistics
    
    # Initialize result with basic info
    result = Dict(
        "coefficients" => dml.coefficients,
        "coef_names" => dml.treatment_names,
        "std_errors" => dml.std_errors,
        "vcov" => dml.vcov,
        "n_obs" => dml.n_obs,
        "method_note" => "DML with cross-fitting and LightGBM",
        "bdml_stats" => dml.bdml_stats,  # Include BDML statistics directly
        "treatments" => Dict{String, Any}()
    )
    
    # Compute detailed statistics for each treatment
    for (i, name) in enumerate(dml.treatment_names)
        coef = dml.coefficients[i]
        se = dml.std_errors[i]
        z = coef / se
        p = 2 * (1 - cdf(Normal(), abs(z)))
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        
        result["treatments"][name] = Dict(
            "coefficient" => coef,
            "std_error" => se,
            "z_statistic" => z,
            "p_value" => p,
            "ci_95" => [ci_lower, ci_upper],
            "statistically_significant" => p < 0.05
        )
    end
    
    # Overall model statistics
    result["model_stats"] = Dict(
        "condition_number" => cond(dml.vcov),
        "joint_test" => Dict(
            "chi2_statistic" => dot(dml.coefficients, inv(dml.vcov) * dml.coefficients),
            "df" => length(dml.coefficients),
            "p_value" => 1 - cdf(Chisq(length(dml.coefficients)), 
                                 dot(dml.coefficients, inv(dml.vcov) * dml.coefficients))
        )
    )
    
    println("Method 3 DML completed")
    
    return result
end

function save_bootstrap_results(results::Dict, output_dir::String, method::String)
    """Save bootstrap results in NPZ format for Python visualization."""
    mkpath(output_dir)
    
    npz_path = joinpath(output_dir, "bootstrap_$(method).npz")
    
    # Prepare data for NPZ
    npz_data = Dict{String, Any}()
    sample_sizes = results["sample_sizes"]
    
    # Metadata
    npz_data["sample_sizes"] = sample_sizes
    npz_data["optimal_n"] = results["optimal_n"]
    npz_data["N_total"] = results["N"]
    npz_data["n_iterations"] = results["n_iterations"]
    
    # Results for each sample size
    for n in sample_sizes
        sr = results["sample_results"][n]
        npz_data["n$(n)_plants_effects"] = sr["plants_effects"]
        npz_data["n$(n)_vertebrates_effects"] = sr["vertebrates_effects"]
        npz_data["n$(n)_p_values"] = sr["p_values"]
        npz_data["n$(n)_r2_improvements"] = sr["r2_improvements"]
        npz_data["n$(n)_power"] = sr["power"]
        npz_data["n$(n)_eco_power"] = sr["ecological_power"]
        npz_data["n$(n)_plants_mean"] = sr["plants_effect"]["mean"]
        npz_data["n$(n)_plants_std"] = sr["plants_effect"]["std"]
    end
    
    npzwrite(npz_path, npz_data)
    println("\nBootstrap results saved to: $npz_path")
end

# ============================================================================
# DHARMA DIAGNOSTIC SUPPORT (minimal, clean)
# ============================================================================

function dump_for_r_dharma(state::PipelineState, method::String; sample_frac::Float64=0.1)
    """Dump sampled preprocessed data for R DHARMa diagnostics."""
    cache_dir = "/scratch/yl998/.cache/dharma"
    mkpath(cache_dir)
    
    # Sample data (DHARMa doesn't need full dataset)
    n_rows = nrow(state.data)
    n_sample = max(10000, Int(floor(n_rows * sample_frac)))  # At least 10k rows
    n_sample = min(n_sample, 100000)  # Cap at 100k for efficiency
    Random.seed!(42)  # Reproducible sampling
    sample_idx = sort(sample(1:n_rows, n_sample, replace=false))
    
    df_sample = state.data[sample_idx, :]
    
    # Use Arrow for efficient R reading
    arrow_file = joinpath(cache_dir, "dharma_data.arrow")
    Arrow.write(arrow_file, df_sample)
    
    # Save minimal metadata (just data info, no formulas)
    metadata = Dict(
        "data_file" => arrow_file,
        "n_sampled" => n_sample,
        "n_total" => n_rows,
        "sample_frac" => n_sample / n_rows,
        "mesf_columns" => state.mesf_columns,  # R needs to know which are MESF
        "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    )
    
    json_file = joinpath(cache_dir, "dharma_meta.json")
    open(json_file, "w") do f
        JSON.print(f, metadata, 2)
    end
    
    println("DHARMa cache saved:")
    println("  Data: $arrow_file ($(round(filesize(arrow_file)/1e6, digits=1)) MB)")
    println("  Meta: $json_file")
    println("  Sampled: $n_sample / $n_rows rows ($(round(100*n_sample/n_rows, digits=1))%)")
end

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

function run_pipeline(method::String, data_path::String, spatial_step::Int=1000;
                     mode::String="normal", bootstrap_iterations::Int=100, dump_for_dharma::Bool=false)
    # Load config
    config_path = "src/config/ml/ml_config.yml"
    config = YAML.load_file(config_path)
    
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
    data, mesf_columns = process_with_mesf(df_raw, method)  # Already sampled if needed
    
    # Create state
    state = PipelineState(
        data,
        mesf_columns,
        config,
        "/scratch/yl998/.cache/mesf",
        "/scratch/yl998/julia"  # Use scratch for large outputs
    )
    
    # DHARMA preprocessing dump (minimal, clean)
    if dump_for_dharma
        dump_for_r_dharma(state, method)
        println("Data dumped for DHARMa diagnostics. Exiting.")
        return Dict("status" => "dharma_cache_saved")
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
    
    if use_ridge
        println("Using Ridge regularization (automatic lambda selection via CV)")
        println("  MESF components: $(length(state.mesf_columns))")
    else
        println("Using standard GLM (no regularization)")
    end
    
    if method == "method1"
        models, lrt = fit_method1(state, use_ridge=use_ridge)
        for m in models
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        all_results["method1_lr_test"] = lrt
        
    elseif method == "method2"
        # First fit method1 for comparisons
        models1, _ = fit_method1(state, use_ridge=use_ridge)
        for m in models1
            all_models[m.name] = m.model
        end
        
        models2 = fit_method2(state, use_ridge=use_ridge)
        for m in models2
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        
        # LRT comparisons - both models are now properly nested
        if haskey(all_models, "method1_full")
            # Compare continuous interaction model (nested - adds interaction term)
            lrt_a = lrtest(all_models["method1_full"], all_models["method2_continuous"])
            all_results["lrt_method1_vs_method2_continuous"] = Dict(
                "lr_statistic" => lrt_a.deviance[2],
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
    
    # Save results
    save_results(all_results, state)
    
    return all_results
end

function save_results(results::Dict, state::PipelineState)
    mkpath(state.output_dir)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Prepare output
    output = Dict(
        "models" => results,
        "metadata" => Dict(
            "n_rows" => nrow(state.data),
            "n_cols" => ncol(state.data),
            "mesf_count" => length(state.mesf_columns)
        ),
        "timestamp" => timestamp,
        "julia_version" => string(VERSION)
    )
    
    # Save JSON
    json_file = joinpath(state.output_dir, "julia_results_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, output, 4)
    end
    println("Results saved: $json_file")
    
    # Save JLD2
    jld2_file = joinpath(state.output_dir, "julia_results_$(timestamp).jld2")
    @save jld2_file output
    
    # Save BDML statistics separately if DML was run
    if haskey(results, "method3_dml") && haskey(results["method3_dml"], "bdml_stats")
        bdml_stats = results["method3_dml"]["bdml_stats"]
        if bdml_stats !== nothing
            # Save as NPZ with same timestamp for easy matching
            bdml_file = joinpath(state.output_dir, "bdml_stats_$(timestamp).npz")
            
            # NPZ can't save strings, so save only numeric arrays
            bdml_numeric = Dict(
                "covariance_matrix" => bdml_stats["covariance_matrix"],
                "n_obs" => bdml_stats["n_obs"],
                "n_treatments" => bdml_stats["n_treatments"],
                "residual_means" => bdml_stats["residual_means"],
                "residual_stds" => bdml_stats["residual_stds"],
                "correlation_matrix" => bdml_stats["correlation_matrix"],
                "condition_number" => [bdml_stats["condition_number"]]  # Wrap scalar in array
            )
            npzwrite(bdml_file, bdml_numeric)
            
            # Save treatment names separately in JSON for reference
            bdml_meta_file = joinpath(state.output_dir, "bdml_meta_$(timestamp).json")
            open(bdml_meta_file, "w") do f
                JSON.print(f, Dict(
                    "treatment_names" => bdml_stats["treatment_names"],
                    "is_positive_definite" => bdml_stats["is_positive_definite"],
                    "bdml_stats_file" => bdml_file
                ), 2)
            end
            
            println("BDML statistics saved: $bdml_file ($(round(filesize(bdml_file)/1024, digits=1)) KB)")
            println("BDML metadata saved: $bdml_meta_file")
            
            # Add references to main results
            results["method3_dml"]["bdml_stats_file"] = bdml_file
            results["method3_dml"]["bdml_meta_file"] = bdml_meta_file
        end
    end
    
    # Update symlink
    latest = joinpath(state.output_dir, "julia_results_latest.json")
    rm(latest, force=true)
    symlink(basename(json_file), latest)
end

# ============================================================================
# CLI
# ============================================================================

if length(ARGS) > 0
    method = ARGS[1]
    
    # Check for dharma mode
    if method == "dharma"
        # DHARMa dump mode: dharma <data_path> [spatial_step]
        if length(ARGS) < 2
            println("Usage: julia $PROGRAM_FILE dharma <data_path> [spatial_step]")
            exit(1)
        end
        
        data_path = ARGS[2]
        spatial_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000
        
        println("Dumping preprocessed data for DHARMa diagnostics...")
        println("  Data: $data_path")
        println("  Spatial step: $spatial_step")
        
        # Run for method1/2 (they share preprocessing)
        run_pipeline("method1", data_path, spatial_step, dump_for_dharma=true)
        
    # Check for bootstrap mode
    elseif method == "bootstrap"
        # Bootstrap mode: bootstrap <model> <data_path> [iterations]
        if length(ARGS) < 3
            println("Usage: julia $PROGRAM_FILE bootstrap <model> <data_path> [iterations]")
            println("Models: method1, method2")
            exit(1)
        end
        
        model = ARGS[2]
        data_path = ARGS[3]
        iterations = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 100
        
        println("Running bootstrap analysis...")
        println("  Model: $model")
        println("  Data: $data_path")
        println("  Spatial step: 1 (HARDCODED - uses full dataset)")
        println("  Iterations: $iterations")
        
        # HARDCODE spatial_step=1 for bootstrap - always use full dataset
        run_pipeline(model, data_path, 1, 
                    mode="bootstrap", bootstrap_iterations=iterations)
        
    elseif method in ["method1", "method2", "method3", "dml"]
        # Normal mode
        data_path = length(ARGS) > 1 ? ARGS[2] : error("Data path required")
        spatial_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000
        
        # For method3/dml, load the DML module early
        if method in ["method3", "dml"]
            include(joinpath(@__DIR__, "multivariable_dml_final.jl"))
        end
        
        run_pipeline(method, data_path, spatial_step)
        
    else
        println("Usage:")
        println("  Normal: julia $PROGRAM_FILE <method> <data_path> [spatial_step]")
        println("  Bootstrap: julia $PROGRAM_FILE bootstrap <model> <data_path> [iterations]")
        println("  DHARMa: julia $PROGRAM_FILE dharma <data_path> [spatial_step]")
        println("Methods: method1, method2, method3/dml")
    end
else
    println("Usage:")
    println("  Normal: julia $PROGRAM_FILE <method> <data_path> [spatial_step]")
    println("  Bootstrap: julia $PROGRAM_FILE bootstrap <model> <data_path> [iterations]")
    println("  DHARMa: julia $PROGRAM_FILE dharma <data_path> [spatial_step]")
end