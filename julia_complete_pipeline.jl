#!/usr/bin/env julia
"""
Complete Julia Pipeline for Biodiversity Analysis
Handles all data processing, MESF computation, and model fitting.
Only returns results to Python for hypothesis testing and reporting.
"""

# Package dependencies are assumed to be installed:
# DataFrames, Arrow, GLM, Distributions, JSON, StatsBase, Distances

using DataFrames, Arrow, Statistics, GLM, Distributions, JSON, StatsBase
using LinearAlgebra, Distances, Random
using YAML, JLD2, SHA, Dates

# ============================================================================
# GLOBAL DATA STORAGE
# ============================================================================
global current_data = nothing
global data_id = ""
global mesf_columns = String[]
global fitted_models = Dict{String, Any}()
global config = Dict{String, Any}()
global cache_dir = "/home/yl998/.cache/mesf"
global output_dir = "results/julia_models"

# ============================================================================
# MESF COMPUTATION (Ported from Python JAX implementation)
# ============================================================================

"""
    haversine_distance(lon1, lat1, lon2, lat2)

Compute haversine distance between two points in kilometers.
"""
function haversine_distance(lon1, lat1, lon2, lat2)
    R = 6371.0  # Earth radius in km
    
    # Convert to radians
    lon1_r, lat1_r = deg2rad(lon1), deg2rad(lat1)
    lon2_r, lat2_r = deg2rad(lon2), deg2rad(lat2)
    
    # Haversine formula
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    
    a = sin(dlat/2)^2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)^2
    c = 2 * asin(sqrt(min(a, 1.0)))
    
    return R * c
end

"""
    compute_spatial_weights(coords, threshold_km)

Compute spatial weights matrix using distance threshold.
"""
function compute_spatial_weights(coords::Matrix{Float64}, threshold_km::Float64)
    n = size(coords, 1)
    W = zeros(n, n)
    
    # Compute pairwise distances
    for i in 1:n
        for j in i+1:n
            dist = haversine_distance(coords[i,1], coords[i,2], coords[j,1], coords[j,2])
            if dist <= threshold_km
                W[i,j] = 1.0
                W[j,i] = 1.0
            end
        end
    end
    
    # Row standardize
    row_sums = sum(W, dims=2)
    row_sums[row_sums .== 0] .= 1.0  # Avoid division by zero
    W = W ./ row_sums
    
    return W
end

"""
    compute_mesf_nystrom(df, n_eigenvectors=10, distance_threshold=100.0)

Compute MESF eigenvectors using Nystrom approximation for large datasets.
"""
function compute_mesf_nystrom(df::DataFrame, n_eigenvectors::Int=10, distance_threshold::Float64=100.0)
    n_total = nrow(df)
    
    # Calculate Nystrom parameters (matching Python implementation)
    n_landmark = min(Int(round(4 * sqrt(n_total))), 50000)  # Cap at 50k for memory
    n_sample = min(Int(round(200 * sqrt(n_total))), 1000000)  # Cap at 1M
    
    println("MESF Nystrom: $n_total total points, $n_landmark landmarks, $n_sample samples")
    
    # Extract coordinates
    coords = Matrix(df[:, [:x, :y]])
    
    # Select landmark indices (random sampling)
    Random.seed!(42)
    landmark_indices = randperm(n_total)[1:min(n_landmark, n_total)]
    coords_landmark = coords[landmark_indices, :]
    
    # Compute weights matrix for landmarks
    println("Computing spatial weights for landmarks...")
    W_landmark = compute_spatial_weights(coords_landmark, distance_threshold)
    
    # Compute eigendecomposition
    println("Computing eigendecomposition...")
    eigenvals, eigenvecs = eigen(W_landmark)
    
    # Select largest eigenvalues (skip trivial)
    idx = sortperm(eigenvals, rev=true)[2:min(n_eigenvectors+1, length(eigenvals))]
    selected_eigenvecs = eigenvecs[:, idx]
    
    # Interpolate to full dataset using k-NN
    println("Interpolating eigenvectors to full dataset...")
    mesf_full = zeros(n_total, n_eigenvectors)
    
    # Simple nearest neighbor interpolation
    for i in 1:n_total
        # Find nearest landmark
        distances = [haversine_distance(coords[i,1], coords[i,2], 
                                       coords_landmark[j,1], coords_landmark[j,2]) 
                    for j in 1:n_landmark]
        nearest_idx = argmin(distances)
        
        # Assign eigenvector values from nearest landmark
        mesf_full[i, :] = selected_eigenvecs[nearest_idx, 1:n_eigenvectors]
    end
    
    # Add MESF columns to dataframe
    for i in 1:n_eigenvectors
        df[!, Symbol("mesf_$i")] = mesf_full[:, i]
    end
    
    # Update global mesf_columns
    global mesf_columns = ["mesf_$i" for i in 1:n_eigenvectors]
    
    println("✓ Added $n_eigenvectors MESF eigenvectors")
    
    return df
end

# ============================================================================
# CONFIGURATION AND CACHING
# ============================================================================

"""
    load_config(config_path::String)

Load configuration from YAML file.
"""
function load_config(config_path::String)
    global config
    if isfile(config_path)
        println("Loading configuration from: $config_path")
        config = YAML.load_file(config_path)
    else
        println("Config file not found, using defaults")
        config = Dict(
            "preprocessing" => Dict(
                "standardization" => "2sd",
                "log_transform" => true
            ),
            "spatial" => Dict(
                "mesf" => Dict(
                    "max_eigenvectors" => 10,
                    "distance_threshold" => 100.0
                )
            ),
            "models" => Dict(
                "glm" => Dict(
                    "family" => "negative_binomial",
                    "maxiter" => 100
                )
            )
        )
    end
    return config
end

"""
    get_cache_key(identifier::String, params::Dict)

Generate a SHA256 cache key from identifier and parameters.
"""
function get_cache_key(identifier::String, params::Dict)
    key_string = "$identifier-$(JSON.json(params, 2))"
    return bytes2hex(sha256(key_string))
end

"""
    load_from_cache(cache_key::String, cache_type::String="mesf")

Load cached data if available.
"""
function load_from_cache(cache_key::String, cache_type::String="mesf")
    cache_file = joinpath(cache_dir, cache_type, "$cache_key.jld2")
    if isfile(cache_file)
        println("Loading from cache: $cache_file")
        @load cache_file data metadata
        return data, metadata
    end
    return nothing, nothing
end

"""
    save_to_cache(cache_key::String, data, metadata::Dict, cache_type::String="mesf")

Save data to cache with metadata.
"""
function save_to_cache(cache_key::String, data, metadata::Dict, cache_type::String="mesf")
    cache_path = joinpath(cache_dir, cache_type)
    mkpath(cache_path)
    cache_file = joinpath(cache_path, "$cache_key.jld2")
    @save cache_file data metadata
    println("Saved to cache: $cache_file")
    return cache_file
end

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

"""
    load_and_preprocess(data_path, spatial_grid_step=1, compute_mesf=true)

Load data from Arrow/Parquet and perform all preprocessing.
"""
function load_and_preprocess(data_path::String, spatial_grid_step::Int=1, 
                           compute_mesf::Bool=true, n_mesf::Int=10)
    global current_data, data_id, mesf_columns
    
    println("="^60)
    println("LOADING AND PREPROCESSING DATA")
    println("="^60)
    
    # Load data
    println("Loading data from: $data_path")
    if endswith(data_path, ".arrow")
        df = DataFrame(Arrow.Table(data_path))
    elseif endswith(data_path, ".parquet")
        # Parquet support disabled - use Arrow format instead
        error("Parquet files not supported due to compatibility issues. Please convert to Arrow format.")
    else
        error("Unsupported file format. Use .arrow or .parquet")
    end
    
    println("Loaded $(nrow(df)) rows, $(ncol(df)) columns")
    
    # Spatial sampling
    if spatial_grid_step > 1
        indices = 1:spatial_grid_step:nrow(df)
        df = df[indices, :]
        println("Sampled to $(nrow(df)) rows (step=$spatial_grid_step)")
    end
    
    # Remove missing values
    initial_rows = nrow(df)
    dropmissing!(df)
    println("Removed $(initial_rows - nrow(df)) rows with missing values")
    
    # Create fungi_richness (always needed for our models)
    if :am_fungi_richness in propertynames(df) && :ecm_fungi_richness in propertynames(df)
        # Convert to Int, handling any Float values
        df.fungi_richness = round.(Int, df.am_fungi_richness .+ df.ecm_fungi_richness)
        println("Created fungi_richness from AM + ECM")
    elseif !(:fungi_richness in propertynames(df))
        error("Cannot create fungi_richness: am_fungi_richness or ecm_fungi_richness missing. Available columns: $(names(df))")
    end
    
    # Standardize climate variables (2SD method)
    climate_vars = [:bio01, :bio04, :bio12, :bio15, :elevation]
    for var in climate_vars
        if var in propertynames(df)
            col_std = Symbol(string(var) * "_std")
            mean_val = mean(skipmissing(df[!, var]))
            std_val = std(skipmissing(df[!, var]))
            df[!, col_std] = (df[!, var] .- mean_val) ./ (2 * std_val)
        end
    end
    println("✓ Standardized climate variables (2SD method)")
    
    # Add elevation squared
    if :elevation_std in propertynames(df)
        df.elevation_std_sq = df.elevation_std .^ 2
        println("✓ Added elevation_std_sq")
    end
    
    # Log transform biodiversity variables with proper naming
    # Handle different naming conventions in the data
    if :plant_richness in propertynames(df)
        df.plants_richness_log = log.(df.plant_richness .+ 1)
        println("  Created plants_richness_log from plant_richness")
    elseif :plants_richness in propertynames(df) 
        df.plants_richness_log = log.(df.plants_richness .+ 1)
        println("  Created plants_richness_log from plants_richness")
    end
    
    if :animal_richness in propertynames(df)
        df.vertebrates_richness_log = log.(df.animal_richness .+ 1)
        println("  Created vertebrates_richness_log from animal_richness")
    elseif :vertebrates_richness in propertynames(df)
        df.vertebrates_richness_log = log.(df.vertebrates_richness .+ 1)
        println("  Created vertebrates_richness_log from vertebrates_richness")
    end
    
    println("✓ Log-transformed biodiversity variables")
    
    # Compute MESF eigenvectors if requested
    if compute_mesf && nrow(df) > 1000  # Only for reasonably sized datasets
        # Try to load from cache first
        cache_params = Dict(
            "n_rows" => nrow(df),
            "n_mesf" => n_mesf,
            "spatial_step" => spatial_grid_step
        )
        cache_key = get_cache_key(data_path, cache_params)
        
        cached_mesf, cache_meta = load_from_cache(cache_key, "mesf")
        if !isnothing(cached_mesf)
            # Add cached MESF columns to dataframe
            for (i, col) in enumerate(eachcol(cached_mesf))
                df[!, Symbol("mesf_$i")] = col
            end
            mesf_columns = ["mesf_$i" for i in 1:ncol(cached_mesf)]
            println("✓ Loaded $(length(mesf_columns)) MESF eigenvectors from cache")
        else
            # Compute and cache
            df = compute_mesf_nystrom(df, n_mesf, 100.0)
            # Extract MESF columns for caching
            mesf_cols = [col for col in names(df) if startswith(string(col), "mesf_")]
            if length(mesf_cols) > 0
                mesf_data = df[:, mesf_cols]
                cache_meta = Dict("n_eigenvectors" => length(mesf_cols))
                save_to_cache(cache_key, mesf_data, cache_meta, "mesf")
            end
        end
    elseif compute_mesf
        println("Skipping MESF: dataset too small ($(nrow(df)) rows)")
        mesf_columns = String[]
    else
        # Check for existing MESF columns
        mesf_columns = String[string(col) for col in names(df) if startswith(string(col), "mesf_")]
        println("Found $(length(mesf_columns)) existing MESF columns")
    end
    
    # Store processed data
    current_data = df
    data_id = string(hash(data_path) * spatial_grid_step)
    
    println("\nFinal dataset: $(nrow(df)) rows × $(ncol(df)) columns")
    println("="^60)
    
    # Return metadata
    return Dict(
        "data_id" => data_id,
        "n_rows" => nrow(df),
        "n_cols" => ncol(df),
        "columns" => names(df),
        "mesf_count" => length(mesf_columns)
    )
end

# ============================================================================
# METHOD 1: H0-H1 TESTING (GLM)
# ============================================================================

"""
    fit_method1_models()

Fit Method 1 models: null (climate only) vs full (climate + biodiversity).
"""
function fit_method1_models()
    global current_data, mesf_columns, fitted_models
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 1: H0-H1 TESTING")
    println("="^60)
    
    # Define feature sets
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    base_features = vcat(climate_features, elevation_features, mesf_columns)
    biotic_features = ["plants_richness_log", "vertebrates_richness_log"]
    
    # Null model (H0): Climate + spatial only
    println("\nFitting null model (climate + spatial)...")
    formula_null = Term(:fungi_richness) ~ sum(Term.(Symbol.(base_features)))
    model_null = negbin(formula_null, current_data, LogLink())
    
    # Full model (H1): Climate + spatial + biodiversity
    println("Fitting full model (climate + spatial + biodiversity)...")
    full_features = vcat(base_features, biotic_features)
    formula_full = Term(:fungi_richness) ~ sum(Term.(Symbol.(full_features)))
    model_full = negbin(formula_full, current_data, LogLink())
    
    # Store results - negbin returns a TableRegressionModel
    # Extract the underlying GLM model
    glm_null = model_null.model
    fitted_models["method1_null"] = Dict(
        "coefficients" => coef(model_null),
        "coef_names" => coefnames(model_null),
        "std_errors" => stderror(model_null),
        "vcov" => vcov(model_null),
        "aic" => aic(model_null),
        "bic" => bic(model_null),
        "loglik" => loglikelihood(model_null),
        "deviance" => deviance(model_null),
        "df_residual" => dof_residual(model_null),
        "n_obs" => nrow(current_data),
        "theta" => glm_null.rr.d.r  # r parameter in NegativeBinomial (1/dispersion)
    )
    
    glm_full = model_full.model
    fitted_models["method1_full"] = Dict(
        "coefficients" => coef(model_full),
        "coef_names" => coefnames(model_full),
        "std_errors" => stderror(model_full),
        "vcov" => vcov(model_full),
        "aic" => aic(model_full),
        "bic" => bic(model_full),
        "loglik" => loglikelihood(model_full),
        "deviance" => deviance(model_full),
        "df_residual" => dof_residual(model_full),
        "n_obs" => nrow(current_data),
        "theta" => glm_full.rr.d.r  # r parameter in NegativeBinomial (1/dispersion)
    )
    
    # Likelihood ratio test
    lr_stat = -2 * (loglikelihood(model_null) - loglikelihood(model_full))
    df_diff = length(biotic_features)
    
    fitted_models["method1_lr_test"] = Dict(
        "lr_statistic" => lr_stat,
        "df" => df_diff,
        "p_value" => 1 - cdf(Chisq(df_diff), lr_stat)
    )
    
    println("\n✓ Method 1 models fitted successfully")
    println("  Null AIC: $(round(aic(model_null), digits=2))")
    println("  Full AIC: $(round(aic(model_full), digits=2))")
    println("  LR test p-value: $(round(fitted_models["method1_lr_test"]["p_value"], digits=6))")
    
    return fitted_models
end

# ============================================================================
# METHOD 2: H2 TESTING (CATEGORICAL)
# ============================================================================

"""
    add_categorical_features()

Add categorical features for Method 2.
"""
function add_categorical_features()
    global current_data
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    df = current_data
    
    # Create temperate category (bio04_std > median)
    if :bio04_std in propertynames(df)
        median_bio4 = median(df.bio04_std)
        df.temperate = Int.(df.bio04_std .> median_bio4)
        println("Created temperate category: $(sum(df.temperate))/$(nrow(df)) temperate")
    end
    
    # Create plant diversity category (plants_richness > median)
    # Check for original plant richness columns first
    if :plant_richness in propertynames(df)
        median_plants = median(df.plant_richness)
        df.plant_high = Int.(df.plant_richness .> median_plants)
        println("Created plant_high category: $(sum(df.plant_high))/$(nrow(df)) high diversity")
    elseif :plants_richness in propertynames(df)
        median_plants = median(df.plants_richness)
        df.plant_high = Int.(df.plants_richness .> median_plants)
        println("Created plant_high category: $(sum(df.plant_high))/$(nrow(df)) high diversity")
    elseif :plants_richness_log in propertynames(df)
        # Fallback: use log-transformed values if originals not available
        median_plants_log = median(df.plants_richness_log)
        df.plant_high = Int.(df.plants_richness_log .> median_plants_log)
        println("Created plant_high category from log values: $(sum(df.plant_high))/$(nrow(df)) high diversity")
    else
        println("Warning: Could not create plant_high category - no plant richness columns found")
    end
    
    # Create interaction terms
    if :temperate in propertynames(df) && :plant_high in propertynames(df)
        df.temperate_x_plant_high = df.temperate .* df.plant_high
        println("Created temperate × plant_high interaction term")
    elseif :temperate in propertynames(df)
        # If we only have temperate but not plant_high, create a dummy plant_high
        df.plant_high = zeros(Int, nrow(df))
        df.temperate_x_plant_high = zeros(Int, nrow(df))
        println("Warning: Created dummy plant_high and interaction (no plant richness data)")
    end
    
    # Create continuous interaction
    if :bio04_std in propertynames(df) && :plants_richness_log in propertynames(df)
        df.bio04_std_x_plants_log = df.bio04_std .* df.plants_richness_log
    end
    
    current_data = df
    
    return Dict(
        "categorical_features_added" => true,
        "features" => ["temperate", "plant_high", "temperate_x_plant_high", "bio04_std_x_plants_log"]
    )
end

"""
    fit_method2_models()

Fit Method 2 models: continuous and categorical interactions.
"""
function fit_method2_models()
    global current_data, mesf_columns, fitted_models
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 2: H2 TESTING (INTERACTIONS)")
    println("="^60)
    
    # Always add categorical features for Method 2
    add_categorical_features()
    
    # Define feature sets
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    biodiv_features = ["plants_richness_log", "vertebrates_richness_log"]
    base_features = vcat(climate_features, elevation_features, biodiv_features, mesf_columns)
    
    # Model 3a: Continuous interaction
    println("\nFitting Model 3a (continuous interaction)...")
    # Create the continuous interaction term first
    if :bio04_std in propertynames(current_data) && :plants_richness_log in propertynames(current_data)
        current_data.bio04_std_x_plants_log = current_data.bio04_std .* current_data.plants_richness_log
    end
    continuous_features = vcat(base_features, ["bio04_std_x_plants_log"])
    formula_3a = Term(:fungi_richness) ~ sum(Term.(Symbol.(continuous_features)))
    model_3a = negbin(formula_3a, current_data, LogLink())
    
    # Model 3b: Categorical interaction
    println("Fitting Model 3b (categorical interaction)...")
    categorical_features = vcat(base_features, ["temperate", "plant_high", "temperate_x_plant_high"])
    # Remove bio04_std and plants_richness_log as they're represented by categories
    categorical_features = filter(x -> !(x in ["bio04_std", "plants_richness_log"]), categorical_features)
    formula_3b = Term(:fungi_richness) ~ sum(Term.(Symbol.(categorical_features)))
    model_3b = negbin(formula_3b, current_data, LogLink())
    
    # Store results - extract underlying GLM models
    glm_3a = model_3a.model
    glm_3b = model_3b.model
    
    fitted_models["method2_continuous"] = Dict(
        "coefficients" => coef(model_3a),
        "coef_names" => coefnames(model_3a),
        "std_errors" => stderror(model_3a),
        "vcov" => vcov(model_3a),
        "aic" => aic(model_3a),
        "bic" => bic(model_3a),
        "loglik" => loglikelihood(model_3a),
        "deviance" => deviance(model_3a),
        "df_residual" => dof_residual(model_3a),
        "n_obs" => nrow(current_data),
        "theta" => glm_3a.rr.d.r
    )
    
    fitted_models["method2_categorical"] = Dict(
        "coefficients" => coef(model_3b),
        "coef_names" => coefnames(model_3b),
        "std_errors" => stderror(model_3b),
        "vcov" => vcov(model_3b),
        "aic" => aic(model_3b),
        "bic" => bic(model_3b),
        "loglik" => loglikelihood(model_3b),
        "deviance" => deviance(model_3b),
        "df_residual" => dof_residual(model_3b),
        "n_obs" => nrow(current_data),
        "theta" => glm_3b.rr.d.r
    )
    
    println("\n✓ Method 2 models fitted successfully")
    println("  Model 3a AIC: $(round(aic(model_3a), digits=2))")
    println("  Model 3b AIC: $(round(aic(model_3b), digits=2))")
    
    return fitted_models
end

# ============================================================================
# METHOD 3: DML (DOUBLE MACHINE LEARNING)
# ============================================================================

"""
    fit_method3_dml()

Fit Method 3: Double Machine Learning for causal inference.
Note: Simplified version - full DML requires more complex implementation.
"""
function fit_method3_dml()
    global current_data, mesf_columns, fitted_models
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 3: DOUBLE MACHINE LEARNING")
    println("="^60)
    
    # For now, implement a simplified version using residualization
    # Full DML would require cross-fitting with GBM
    
    # Define variables
    outcome = :fungi_richness
    treatments = [:plants_richness_log, :vertebrates_richness_log]
    
    # Confounders (everything else)
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    confounders = vcat(climate_features, elevation_features, mesf_columns)
    
    println("Implementing simplified DML with residualization...")
    
    # Convert all numeric columns to Float64 to avoid type issues
    for col in names(current_data)
        if eltype(current_data[!, col]) <: Number && !(eltype(current_data[!, col]) <: Integer)
            current_data[!, col] = Float64.(current_data[!, col])
        end
    end
    
    # Step 1: Residualize outcome on confounders
    formula_y = Term(outcome) ~ sum(Term.(Symbol.(confounders)))
    model_y = lm(formula_y, current_data)
    residuals_y = Float64.(residuals(model_y))  # Ensure Float64
    
    # Step 2: Residualize each treatment on confounders
    residuals_treatments = Dict()
    for treatment in treatments
        formula_t = Term(treatment) ~ sum(Term.(Symbol.(confounders)))
        model_t = lm(formula_t, current_data)
        residuals_treatments[treatment] = Float64.(residuals(model_t))  # Ensure Float64
    end
    
    # Step 3: Regress residualized outcome on residualized treatments
    df_residuals = DataFrame(
        y_resid = residuals_y,
        plants_resid = residuals_treatments[:plants_richness_log],
        vertebrates_resid = residuals_treatments[:vertebrates_richness_log]
    )
    
    model_dml = lm(@formula(y_resid ~ plants_resid + vertebrates_resid), df_residuals)
    
    # Store results
    fitted_models["method3_dml"] = Dict(
        "coefficients" => coef(model_dml),
        "coef_names" => ["intercept", "plants_richness_effect", "vertebrates_richness_effect"],
        "std_errors" => stderror(model_dml),
        "vcov" => vcov(model_dml),
        "r2" => r2(model_dml),
        "n_obs" => nrow(current_data),
        "method_note" => "Simplified DML using residualization"
    )
    
    println("\n✓ Method 3 DML fitted successfully (simplified version)")
    println("  Plants effect: $(round(coef(model_dml)[2], digits=4))")
    println("  Vertebrates effect: $(round(coef(model_dml)[3], digits=4))")
    
    return fitted_models
end

# ============================================================================
# RESULTS EXPORT
# ============================================================================

"""
    save_results_to_disk(results::Dict, metadata::Dict)

Save all model results and metadata to disk for Python to load.
"""
function save_results_to_disk(results::Dict, metadata::Dict)
    global output_dir
    
    # Create output directory
    mkpath(output_dir)
    
    # Create a new dictionary with all results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    full_results = Dict(
        "models" => results["models"],
        "metadata" => results["metadata"],
        "config" => get(results, "config", Dict()),
        "timestamp" => timestamp,
        "julia_version" => string(VERSION)
    )
    
    # Save as JSON for Python compatibility
    json_file = joinpath(output_dir, "julia_results_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, full_results, 4)
    end
    println("\n✅ Results saved to: $json_file")
    
    # Also save as JLD2 for Julia (preserves full precision)
    jld2_file = joinpath(output_dir, "julia_results_$(timestamp).jld2")
    @save jld2_file full_results metadata
    println("✅ Full precision data saved to: $jld2_file")
    
    # Save a "latest" symlink for easy access
    latest_json = joinpath(output_dir, "julia_results_latest.json")
    if isfile(latest_json) || islink(latest_json)
        rm(latest_json)
    end
    # Use just the filename for the symlink target, not the full path
    symlink(basename(json_file), latest_json)
    
    return json_file
end

# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

"""
    run_complete_pipeline(data_path, spatial_grid_step=1, compute_mesf=true, n_mesf=10)

Run the complete analysis pipeline.
"""
function run_complete_pipeline(data_path::String; 
                              spatial_grid_step::Int=1,
                              compute_mesf::Bool=true,
                              n_mesf::Int=10,
                              config_path::String="",
                              save_dir::String="")
    
    global output_dir, config
    
    println("\n" * "="^70)
    println("COMPLETE JULIA PIPELINE FOR BIODIVERSITY ANALYSIS")
    println("="^70)
    
    # Load configuration if provided
    if !isempty(config_path)
        load_config(config_path)
    end
    
    # Set output directory if provided
    if !isempty(save_dir)
        output_dir = save_dir
    end
    
    # Load and preprocess data
    metadata = load_and_preprocess(data_path, spatial_grid_step, compute_mesf, n_mesf)
    
    # Fit all methods
    fit_method1_models()
    fit_method2_models()
    fit_method3_dml()
    
    # Prepare results for Python
    results = Dict(
        "metadata" => metadata,
        "models" => fitted_models,
        "config" => config
    )
    
    # Save results to disk
    saved_file = save_results_to_disk(results, metadata)
    
    println("\n" * "="^70)
    println("✅ PIPELINE COMPLETED SUCCESSFULLY")
    println("="^70)
    
    return results
end

"""
    get_current_data_summary()

Get summary of currently loaded data.
"""
function get_current_data_summary()
    global current_data, mesf_columns
    
    if isnothing(current_data)
        return Dict("error" => "No data loaded")
    end
    
    return Dict(
        "n_rows" => nrow(current_data),
        "n_cols" => ncol(current_data),
        "columns" => names(current_data),
        "mesf_columns" => mesf_columns,
        "has_categorical" => "temperate" in names(current_data)
    )
end

"""
    get_model_results(model_name)

Get results for a specific model.
"""
function get_model_results(model_name::String)
    global fitted_models
    
    if haskey(fitted_models, model_name)
        return fitted_models[model_name]
    else
        return Dict("error" => "Model $model_name not found")
    end
end

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if length(ARGS) > 0
    command = ARGS[1]
    
    if command == "run_pipeline"
        # Full pipeline execution
        data_path = ARGS[2]
        spatial_grid_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
        compute_mesf = length(ARGS) > 3 ? parse(Bool, ARGS[4]) : true
        n_mesf = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 10
        config_path = length(ARGS) > 5 ? ARGS[6] : ""
        save_dir = length(ARGS) > 6 ? ARGS[7] : ""
        
        results = run_complete_pipeline(data_path, 
                                       spatial_grid_step=spatial_grid_step,
                                       compute_mesf=compute_mesf,
                                       n_mesf=n_mesf,
                                       config_path=config_path,
                                       save_dir=save_dir)
        # Don't print full results to stdout when saving to disk
        if !isempty(save_dir)
            println("Results saved to disk")
        else
            println(JSON.json(results))
        end
        
    elseif command == "load_data"
        # Just load and preprocess
        data_path = ARGS[2]
        spatial_grid_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
        metadata = load_and_preprocess(data_path, spatial_grid_step)
        println(JSON.json(metadata))
        
    elseif command == "fit_method1"
        results = fit_method1_models()
        println(JSON.json(results))
        
    elseif command == "fit_method2"
        results = fit_method2_models()
        println(JSON.json(results))
        
    elseif command == "fit_method3"
        results = fit_method3_dml()
        println(JSON.json(results))
        
    elseif command == "get_summary"
        summary = get_current_data_summary()
        println(JSON.json(summary))
        
    else
        println("Unknown command: $command")
        println("Available commands: run_pipeline, load_data, fit_method1, fit_method2, fit_method3, get_summary")
    end
else
    println("Julia Complete Pipeline for Biodiversity Analysis")
    println("Usage: julia julia_complete_pipeline.jl <command> [args...]")
    println("\nCommands:")
    println("  run_pipeline <data_path> [spatial_step] [compute_mesf] [n_mesf]")
    println("  load_data <data_path> [spatial_step]")
    println("  fit_method1")
    println("  fit_method2")
    println("  fit_method3")
    println("  get_summary")
end