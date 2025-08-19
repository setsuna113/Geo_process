#!/usr/bin/env julia
"""
Hybrid Julia Pipeline for Biodiversity Analysis
Uses Python's optimized MESF computation and Julia's GLM fitting.
This is the pragmatic solution that leverages existing optimizations.

EXPORT STRUCTURE for each model:
=====================================
Each fitted model exports the following structured dictionary:

{
    "core_stats": {           # ORIGINAL - Core model statistics
        "coefficients",       # Model coefficients
        "coef_names",        # Coefficient names
        "std_errors",        # Standard errors
        "vcov",              # Variance-covariance matrix
        "aic", "bic",        # Information criteria
        "loglik",            # Log-likelihood
        "deviance",          # Model deviance
        "df_residual",       # Residual degrees of freedom
        "n_obs",             # Number of observations
        "theta"              # NB dispersion parameter
    },
    
    "residuals": {           # NEW - For diagnostic plots
        "y_values",          # Actual response values
        "fitted_values",     # Model predictions
        "response_residuals", # Simple residuals (y - μ)
        "deviance_residuals" # Deviance residuals (for diagnostics)
    },
    
    "extended_stats": {      # NEW - Additional model metrics
        "null_deviance",     # Deviance of null model
        "null_loglik",       # Log-likelihood of null model
        "dispersion"         # Dispersion parameter
    },
    
    "inference": {           # NEW - For hypothesis testing
        "coef_z_values",     # Z-statistics for coefficients
        "coef_p_values",     # P-values for coefficients
        "confint_95"         # 95% confidence intervals
    },
    
    "multicollinearity": {   # NEW - Collinearity diagnostics
        "gvif",              # Generalized VIF values
        "max_gvif",          # Maximum GVIF
        "high_collinearity"  # Boolean flag if max > 10
    },
    
    "diagnostic_tests": {    # NEW - Statistical tests
        "jarque_bera": {     # Normality test
            "statistic",
            "p_value",
            "normal_residuals"
        },
        "anderson_darling": { # Alternative normality test
            "statistic",
            "p_value",
            "normal_residuals"
        },
        "durbin_watson": {    # Autocorrelation test
            "statistic",
            "p_value",
            "autocorrelation_present"
        }
    }
}

Additionally, likelihood ratio tests between models are stored separately:
- "method1_lr_test": LRT between null and full models
- "lrt_method1_vs_method2a": LRT between method1_full and method2_continuous
- "lrt_method1_vs_method2b": LRT between method1_full and method2_categorical
"""

using DataFrames, Arrow, Statistics, GLM, Distributions, JSON, StatsBase
using LinearAlgebra, Random
using YAML, JLD2, SHA, Dates
using NPZ  # For loading numpy arrays
using StatsModels  # For lrtest, coeftable, gvif
using HypothesisTests  # For normality tests, Durbin-Watson

# ============================================================================
# GLOBAL DATA STORAGE
# ============================================================================
global current_data = nothing
global data_id = ""
global mesf_columns = String[]
global fitted_models = Dict{String, Any}()
global model_objects = Dict{String, Any}()  # Store actual model objects for LRT
global config = Dict{String, Any}()
global cache_dir = "/scratch/yl998/.cache/mesf"
global output_dir = "results/julia_models"

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    load_config(config_path::String)

Load configuration from YAML file.
"""
function load_config(config_path::String)
    global config
    if !isfile(config_path)
        error("Config file not found: $config_path - Configuration is required!")
    end
    println("Loading configuration from: $config_path")
    config = YAML.load_file(config_path)
    return config
end

# ============================================================================
# DATA LOADING WITH PYTHON MESF INTEGRATION
# ============================================================================

"""
    load_with_python_mesf(data_path, mesf_path, spatial_grid_step=1; method="method1")

Load data and integrate pre-computed Python MESF eigenvectors.
This is the core of the hybrid approach - use Python's optimized MESF.

Args:
    data_path: Path to data file
    mesf_path: Path to MESF eigenvectors
    spatial_grid_step: Spatial sampling step
    method: "method1"/"method2" for GLM (log transform), "method3"/"dml" for DML (raw counts)
"""
function load_with_python_mesf(data_path::String, mesf_path::String, spatial_grid_step::Int=1; method::String="method1")
    global current_data, data_id, mesf_columns
    
    println("="^60)
    println("LOADING DATA WITH PYTHON MESF")
    println("="^60)
    
    # Load main data
    println("Loading data from: $data_path")
    if endswith(data_path, ".arrow")
        df = DataFrame(Arrow.Table(data_path), copycols=true)
    elseif endswith(data_path, ".parquet")
        # For parquet, we'll use Arrow format instead
        error("Please convert to Arrow format first. Parquet has compatibility issues.")
    else
        error("Unsupported file format. Use .arrow")
    end
    
    println("Loaded $(nrow(df)) rows, $(ncol(df)) columns")
    
    # Apply spatial sampling to match Python
    if spatial_grid_step > 1
        indices = 1:spatial_grid_step:nrow(df)
        df = df[indices, :]
        println("Sampled to $(nrow(df)) rows (step=$spatial_grid_step)")
    end
    
    # Convert all numeric columns to Float32 for memory efficiency
    println("Converting numeric columns to Float32...")
    for col in names(df)
        if eltype(df[!, col]) <: Number && !(eltype(df[!, col]) <: AbstractString)
            df[!, col] = Float32.(df[!, col])
        end
    end
    
    # Handle MESF eigenvectors
    # First check for cached MESF file with matching spatial_grid_step
    cached_mesf_path = joinpath(cache_dir, "mesf_step$(spatial_grid_step)_n10.npy")
    
    if isfile(cached_mesf_path)
        # Found cached MESF for this spatial_grid_step
        println("Found cached MESF for step=$spatial_grid_step: $cached_mesf_path")
        mesf_path = cached_mesf_path
    elseif !isempty(mesf_path) && !isfile(mesf_path)
        println("Warning: Provided MESF path doesn't exist: $mesf_path")
        mesf_path = ""
    end
    
    if !isempty(mesf_path) && isfile(mesf_path)
        # Load MESF (either from cache or provided path)
        println("Loading MESF eigenvectors from: $mesf_path")
        
        if endswith(mesf_path, ".npy")
            # Load numpy array
            mesf_array = npzread(mesf_path)
            n_mesf = size(mesf_array, 2)
            
            # Verify dimensions match
            if size(mesf_array, 1) != nrow(df)
                error("MESF dimensions mismatch: $(size(mesf_array, 1)) vs $(nrow(df)) data rows")
            end
            
            # Add MESF columns to dataframe (as Float32)
            for i in 1:n_mesf
                df[!, Symbol("mesf_$i")] = Float32.(mesf_array[:, i])
            end
            mesf_columns = ["mesf_$i" for i in 1:n_mesf]
            println("✓ Added $n_mesf MESF eigenvectors from Python")
            
        elseif endswith(mesf_path, ".arrow")
            # Load Arrow format MESF
            mesf_df = DataFrame(Arrow.Table(mesf_path))
            
            # Find MESF columns
            mesf_cols = [string(col) for col in names(mesf_df) if startswith(string(col), "mesf_")]
            
            # Add to main dataframe
            for col in mesf_cols
                df[!, Symbol(col)] = mesf_df[!, Symbol(col)]
            end
            mesf_columns = mesf_cols
            println("✓ Added $(length(mesf_cols)) MESF eigenvectors from Arrow")
            
        else
            error("Unsupported MESF format. Use .npy or .arrow")
        end
    else
        # MESF not provided or doesn't exist - compute it
        println("MESF file not found. Computing MESF eigenvectors...")
        
        # Save current dataframe temporarily for MESF computation (as Arrow)
        temp_data_path = "/tmp/temp_for_mesf_$(hash(data_path))_$(spatial_grid_step).arrow"
        Arrow.write(temp_data_path, df)
        
        # Compute MESF using Python (data already sampled in Julia)
        mesf_file = compute_mesf_with_python(
            temp_data_path,
            spatial_grid_step=spatial_grid_step,  # For filename only
            n_mesf=10
        )
        
        # Load the computed MESF
        if isfile(mesf_file)
            mesf_array = npzread(mesf_file)
            n_mesf = size(mesf_array, 2)
            
            # Add MESF columns to dataframe (as Float32)
            for i in 1:n_mesf
                df[!, Symbol("mesf_$i")] = Float32.(mesf_array[:, i])
            end
            mesf_columns = ["mesf_$i" for i in 1:n_mesf]
            println("✓ Computed and added $n_mesf MESF eigenvectors")
        else
            println("Warning: MESF computation failed, proceeding without spatial features")
            mesf_columns = String[]
        end
        
        # Clean up temp file
        rm(temp_data_path, force=true)
    end
    
    # Preprocess the data based on the method
    df = preprocess_data!(df, method=method)
    
    # Store processed data
    current_data = df
    data_id = string(hash(data_path) * spatial_grid_step)
    
    println("\nFinal dataset: $(nrow(df)) rows × $(ncol(df)) columns")
    println("MESF columns: $(length(mesf_columns))")
    println("="^60)
    
    return Dict(
        "data_id" => data_id,
        "n_rows" => nrow(df),
        "n_cols" => ncol(df),
        "columns" => names(df),
        "mesf_count" => length(mesf_columns)
    )
end

"""
    preprocess_data!(df::DataFrame; method::String="method1")

Preprocess the dataframe with standardization and transformations.

Args:
    df: DataFrame to preprocess
    method: Determines preprocessing - "method1"/"method2" use log transform,
            "method3"/"dml" keep raw counts
"""
function preprocess_data!(df::DataFrame; method::String="method1")
    # Remove missing values
    initial_rows = nrow(df)
    dropmissing!(df)
    println("Removed $(initial_rows - nrow(df)) rows with missing values")
    
    # Create fungi_richness (always from AM + ECM)
    if :am_fungi_richness in propertynames(df) && :ecm_fungi_richness in propertynames(df)
        df.fungi_richness = df.am_fungi_richness .+ df.ecm_fungi_richness  # Will be converted to Float32
        println("Created fungi_richness from AM + ECM")
    elseif !(:fungi_richness in propertynames(df))
        error("Cannot create fungi_richness: required columns missing")
    end
    
    # Ensure we have raw richness columns (before any transformations)
    # Handle different possible column names
    if :plant_richness in propertynames(df)
        df.plants_richness = df.plant_richness  # Standardize naming
    elseif !(:plants_richness in propertynames(df))
        error("Cannot find plants richness data")
    end
    
    if :animal_richness in propertynames(df)
        df.vertebrates_richness = df.animal_richness  # Standardize naming
    elseif :terrestrial_richness in propertynames(df)
        df.vertebrates_richness = df.terrestrial_richness
    elseif !(:vertebrates_richness in propertynames(df))
        error("Cannot find vertebrates/animal richness data")
    end
    
    # Standardize climate variables (2SD method) - same for all methods
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
    
    # Add elevation squared - same for all methods
    if :elevation_std in propertynames(df)
        df.elevation_std_sq = df.elevation_std .^ 2
        println("✓ Added elevation_std_sq")
    end
    
    # Method-specific transformations
    if method in ["method3", "dml"]
        # Method 3 (DML): Keep raw counts as Float64 for causal interpretation
        println("✓ Keeping raw counts as Float64 for DML (Method 3)")
        println("  Plants: $(minimum(df.plants_richness))-$(maximum(df.plants_richness)) species")
        println("  Vertebrates: $(minimum(df.vertebrates_richness))-$(maximum(df.vertebrates_richness)) species")
        println("  Fungi: $(minimum(df.fungi_richness))-$(maximum(df.fungi_richness)) species")
    else
        # Methods 1-2: GLM requires integer counts for negative binomial
        # Round fungi to nearest integer for count models
        df.fungi_richness = round.(Int, df.fungi_richness)
        
        # Log transform predictors for GLM
        df.plants_richness_log = log.(df.plants_richness .+ 1)
        df.vertebrates_richness_log = log.(df.vertebrates_richness .+ 1)
        println("✓ Rounded fungi to integers and log-transformed biodiversity predictors for GLM (Methods 1-2)")
    end
    
    return df
end

# ============================================================================
# PYTHON MESF COMPUTATION CALLER
# ============================================================================

"""
    compute_mesf_with_python(data_path, spatial_grid_step=1, n_mesf=10)

Call Python to compute MESF eigenvectors using the optimized implementation.
Returns the path to the saved MESF file.
"""
function compute_mesf_with_python(data_path::String; 
                                 spatial_grid_step::Int=1,
                                 n_mesf::Int=10,
                                 output_format::String="npy")
    println("="^60)
    println("COMPUTING MESF WITH PYTHON")
    println("="^60)
    
    # Call mesf_processor directly via CLI
    println("Calling MESF processor...")
    
    # Build command with arguments  
    mesf_processor_path = joinpath(dirname(@__FILE__), "mesf_processor.py")
    
    # Get n_jobs from config (lightgbm.num_threads)
    n_jobs = config["regression_analysis"]["models"]["lightgbm"]["num_threads"]
    
    # Use taskset to hard-limit CPU cores - no environment variables needed
    # Libraries will auto-detect available cores within taskset limits
    cpu_list = "0-$(n_jobs-1)"  # Use cores 0 to n_jobs-1
    
    # Build command with taskset only
    python_cmd = `taskset -c $(cpu_list) $(homedir())/anaconda3/envs/geo/bin/python $(mesf_processor_path) 
        $(data_path) 
        --spatial-step $(spatial_grid_step) 
        --n-eigenvectors $(n_mesf) 
        --n-jobs $(n_jobs) 
        --output-dir $(cache_dir)`
    
    # Run the command
    result = read(python_cmd, String)
    println(result)
    
    # Extract output file path from last line
    lines = split(strip(result), '\n')
    mesf_file = String(strip(lines[end]))
    
    if !isfile(mesf_file)
        error("MESF computation failed - output file not found")
    end
    
    println("✓ MESF computation complete")
    return mesf_file
end

# ============================================================================
# MODEL FITTING (Same as before)
# ============================================================================

"""
    calculate_deviance_residuals(model)

Calculate deviance residuals for a negative binomial GLM.
"""
function calculate_deviance_residuals(model)
    y = response(model)
    μ = predict(model)
    θ = model.model.rr.d.r  # Extract theta parameter
    
    # Calculate deviance residuals
    dist = NegativeBinomial(θ)
    dev_resid = sign.(y .- μ) .* sqrt.(devresid.(dist, y, μ))
    
    return dev_resid
end

"""
    extract_model_diagnostics(model, model_name::String)

Extract comprehensive diagnostics from a fitted GLM model.
Returns a structured dictionary with clear sections.
"""
function extract_model_diagnostics(model, model_name::String)
    # Structure output in clear sections
    diagnostics = Dict(
        # SECTION 1: Core model statistics (ORIGINAL - always exported)
        "core_stats" => Dict(
            "coefficients" => coef(model),
            "coef_names" => coefnames(model),
            "std_errors" => stderror(model),
            "vcov" => vcov(model),
            "aic" => aic(model),
            "bic" => bic(model),
            "loglik" => loglikelihood(model),
            "deviance" => deviance(model),
            "df_residual" => dof_residual(model),
            "n_obs" => nobs(model),
            "theta" => model.model.rr.d.r
        ),
        
        # SECTION 2: Residuals for diagnostics (NEW - for diagnostic modules)
        "residuals" => Dict(
            "y_values" => response(model),
            "fitted_values" => predict(model),
            "response_residuals" => residuals(model),
            "deviance_residuals" => calculate_deviance_residuals(model)
        ),
        
        # SECTION 3: Extended model statistics (NEW)
        "extended_stats" => Dict(
            "null_deviance" => nulldeviance(model),
            "null_loglik" => nullloglikelihood(model),
            "dispersion" => dispersion(model)
        )
    )
    
    # SECTION 4: Inference statistics (NEW - for hypothesis testing)
    diagnostics["inference"] = Dict()
    
    # Coefficient inference
    try
        ct = coeftable(model)
        # Store as arrays since coeftable might not serialize well
        diagnostics["inference"]["coef_z_values"] = ct.cols[3]  # z-values column
        diagnostics["inference"]["coef_p_values"] = ct.cols[4]  # p-values column
    catch e
        println("Warning: Could not extract coeftable for $model_name: $e")
    end
    
    # Confidence intervals
    try
        ci = confint(model, 0.95)
        diagnostics["inference"]["confint_95"] = ci
    catch e
        println("Warning: Could not extract confidence intervals for $model_name: $e")
    end
    
    # SECTION 5: Multicollinearity diagnostics (NEW)
    diagnostics["multicollinearity"] = Dict()
    try
        vif_values = gvif(model)
        diagnostics["multicollinearity"]["gvif"] = vif_values
        diagnostics["multicollinearity"]["max_gvif"] = maximum(vif_values)
        diagnostics["multicollinearity"]["high_collinearity"] = maximum(vif_values) > 10
        
        if maximum(vif_values) > 10
            println("⚠ Warning: High multicollinearity detected (max GVIF = $(maximum(vif_values)))")
        end
    catch e
        println("Note: GVIF calculation not available for $model_name: $e")
        diagnostics["multicollinearity"]["available"] = false
    end
    
    # SECTION 6: Residual diagnostic tests (NEW)
    diagnostics["diagnostic_tests"] = Dict()
    
    # Normality tests on deviance residuals
    try
        dev_resid = diagnostics["residuals"]["deviance_residuals"]
        
        # Jarque-Bera test
        jb_test = JarqueBeraTest(dev_resid)
        diagnostics["diagnostic_tests"]["jarque_bera"] = Dict(
            "statistic" => jb_test.JB,
            "p_value" => pvalue(jb_test),
            "normal_residuals" => pvalue(jb_test) > 0.05
        )
        
        # Anderson-Darling test
        ad_test = OneSampleADTest(dev_resid, Normal(0, 1))
        diagnostics["diagnostic_tests"]["anderson_darling"] = Dict(
            "statistic" => ad_test.A²,
            "p_value" => pvalue(ad_test),
            "normal_residuals" => pvalue(ad_test) > 0.05
        )
    catch e
        println("Note: Normality tests not available for $model_name: $e")
        diagnostics["diagnostic_tests"]["normality_tests_available"] = false
    end
    
    # Durbin-Watson test for autocorrelation
    try
        dw_test = DurbinWatsonTest(diagnostics["residuals"]["deviance_residuals"])
        diagnostics["diagnostic_tests"]["durbin_watson"] = Dict(
            "statistic" => dw_test.DW,
            "p_value" => pvalue(dw_test),
            "autocorrelation_present" => dw_test.DW < 1.5 || dw_test.DW > 2.5
        )
    catch e
        println("Note: Durbin-Watson test not available for $model_name: $e")
        diagnostics["diagnostic_tests"]["durbin_watson_available"] = false
    end
    
    return diagnostics
end

"""
    fit_method1_models()

Fit Method 1 models: null (climate only) vs full (climate + biodiversity).
"""
function fit_method1_models()
    global current_data, mesf_columns, fitted_models, model_objects
    
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
    
    # Store model objects for later LRT comparisons
    model_objects["method1_null"] = model_null
    model_objects["method1_full"] = model_full
    
    # Store results with comprehensive diagnostics
    fitted_models["method1_null"] = extract_model_diagnostics(model_null, "method1_null")
    fitted_models["method1_full"] = extract_model_diagnostics(model_full, "method1_full")
    
    # Use lrtest instead of manual calculation
    try
        lrt_result = lrtest(model_null, model_full)
        fitted_models["method1_lr_test"] = Dict(
            "lr_statistic" => lrt_result.deviance[2],  # Difference in deviance
            "df" => lrt_result.dof[2],  # Difference in df
            "p_value" => lrt_result.pval[2],  # p-value for comparison
            "test_type" => "Likelihood Ratio Test (lrtest)"
        )
    catch e
        # Fallback to manual calculation if lrtest fails
        println("Warning: lrtest failed, using manual calculation: $e")
        lr_stat = -2 * (loglikelihood(model_null) - loglikelihood(model_full))
        df_diff = length(biotic_features)
        fitted_models["method1_lr_test"] = Dict(
            "lr_statistic" => lr_stat,
            "df" => df_diff,
            "p_value" => 1 - cdf(Chisq(df_diff), lr_stat),
            "test_type" => "Likelihood Ratio Test (manual)"
        )
    end
    
    println("\n✓ Method 1 models fitted successfully")
    println("  Null AIC: $(round(aic(model_null), digits=2))")
    println("  Full AIC: $(round(aic(model_full), digits=2))")
    println("  LR test p-value: $(round(fitted_models["method1_lr_test"]["p_value"], digits=6))")
    
    return fitted_models
end

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
    
    # Create temperate category
    if :bio04_std in propertynames(df)
        median_bio4 = median(df.bio04_std)
        df.temperate = Int.(df.bio04_std .> median_bio4)
        println("Created temperate category: $(sum(df.temperate))/$(nrow(df)) temperate")
    end
    
    # Create plant diversity category
    if :plant_richness in propertynames(df)
        median_plants = median(df.plant_richness)
        df.plant_high = Int.(df.plant_richness .> median_plants)
    elseif :plants_richness in propertynames(df)
        median_plants = median(df.plants_richness)
        df.plant_high = Int.(df.plants_richness .> median_plants)
    elseif :plants_richness_log in propertynames(df)
        median_plants_log = median(df.plants_richness_log)
        df.plant_high = Int.(df.plants_richness_log .> median_plants_log)
    end
    
    # Create interaction terms
    if :temperate in propertynames(df) && :plant_high in propertynames(df)
        df.temperate_x_plant_high = df.temperate .* df.plant_high
        println("Created temperate × plant_high interaction term")
    end
    
    # Create continuous interaction
    if :bio04_std in propertynames(df) && :plants_richness_log in propertynames(df)
        df.bio04_std_x_plants_log = df.bio04_std .* df.plants_richness_log
    end
    
    current_data = df
end

"""
    fit_method2_models()

Fit Method 2 models: continuous and categorical interactions.
"""
function fit_method2_models()
    global current_data, mesf_columns, fitted_models, model_objects
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 2: H2 TESTING (INTERACTIONS)")
    println("="^60)
    
    # Add categorical features
    add_categorical_features()
    
    # Define feature sets
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    biodiv_features = ["plants_richness_log", "vertebrates_richness_log"]
    base_features = vcat(climate_features, elevation_features, biodiv_features, mesf_columns)
    
    # Model 3a: Continuous interaction
    println("\nFitting Model 3a (continuous interaction)...")
    continuous_features = vcat(base_features, ["bio04_std_x_plants_log"])
    formula_3a = Term(:fungi_richness) ~ sum(Term.(Symbol.(continuous_features)))
    model_3a = negbin(formula_3a, current_data, LogLink())
    
    # Model 3b: Categorical interaction
    println("Fitting Model 3b (categorical interaction)...")
    categorical_features = vcat(base_features, ["temperate", "plant_high", "temperate_x_plant_high"])
    categorical_features = filter(x -> !(x in ["bio04_std", "plants_richness_log"]), categorical_features)
    formula_3b = Term(:fungi_richness) ~ sum(Term.(Symbol.(categorical_features)))
    model_3b = negbin(formula_3b, current_data, LogLink())
    
    # Store model objects
    model_objects["method2_continuous"] = model_3a
    model_objects["method2_categorical"] = model_3b
    
    # Store results with comprehensive diagnostics
    fitted_models["method2_continuous"] = extract_model_diagnostics(model_3a, "method2_continuous")
    fitted_models["method2_categorical"] = extract_model_diagnostics(model_3b, "method2_categorical")
    
    # Add LRT comparisons between models
    if haskey(model_objects, "method1_full")
        # Compare Method 1 full vs Method 2 models
        try
            println("\nPerforming LRT comparisons between models...")
            
            # Method 1 full vs Method 2 continuous
            lrt_full_vs_3a = lrtest(model_objects["method1_full"], model_3a)
            fitted_models["lrt_method1_vs_method2a"] = Dict(
                "lr_statistic" => lrt_full_vs_3a.deviance[2],
                "df" => lrt_full_vs_3a.dof[2],
                "p_value" => lrt_full_vs_3a.pval[2],
                "comparison" => "method1_full vs method2_continuous"
            )
            
            # Method 1 full vs Method 2 categorical
            lrt_full_vs_3b = lrtest(model_objects["method1_full"], model_3b)
            fitted_models["lrt_method1_vs_method2b"] = Dict(
                "lr_statistic" => lrt_full_vs_3b.deviance[2],
                "df" => lrt_full_vs_3b.dof[2],
                "p_value" => lrt_full_vs_3b.pval[2],
                "comparison" => "method1_full vs method2_categorical"
            )
            
            println("  ✓ LRT comparisons completed")
        catch e
            println("Warning: LRT comparisons failed: $e")
        end
    end
    
    println("\n✓ Method 2 models fitted successfully")
    println("  Model 3a AIC: $(round(aic(model_3a), digits=2))")
    println("  Model 3b AIC: $(round(aic(model_3b), digits=2))")
    
    return fitted_models
end

"""
    prepare_data_for_dml()

Prepare data specifically for DML (Method 3) with proper preprocessing.
Returns a properly formatted DataFrame for DML.
"""
function prepare_data_for_dml()
    global current_data, mesf_columns
    
    # Check if we need to reload with DML-specific preprocessing
    if !(:plants_richness in propertynames(current_data)) || 
       !(:vertebrates_richness in propertynames(current_data))
        println("⚠ Data needs DML-specific preprocessing, reloading...")
        
        # We need to reload the original data and preprocess for DML
        # This is a bit inefficient but ensures correctness
        # In production, you'd want to cache both versions
        return nothing  # Signal that we need special handling
    end
    
    # Data already has raw counts, we can use it
    return current_data
end

"""
    fit_method3_dml()

Fit Method 3: TRUE Double Machine Learning using multivariable_dml.jl
"""
function fit_method3_dml()
    global current_data, mesf_columns, fitted_models
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 3: DOUBLE MACHINE LEARNING")
    println("="^60)
    
    # Load the proper DML implementation (check if not already loaded)
    if !isdefined(Main, :MultivariableDML)
        include(joinpath(@__DIR__, "multivariable_dml.jl"))
    end
    
    # Check if data has raw counts (from DML preprocessing)
    dml_data = prepare_data_for_dml()
    
    if isnothing(dml_data)
        # Data was preprocessed for GLM (log-transformed), not DML
        error("Cannot run DML: data was preprocessed with log transformation for GLM. " *
              "Please run: julia julia_hybrid_pipeline.jl method3 <data_path> " *
              "to load data with raw counts for DML.")
    end
    
    # Verify we have all required columns
    required_cols = ["fungi_richness", "plants_richness", "vertebrates_richness",
                     "bio01_std", "bio04_std", "bio12_std", "bio15_std",
                     "elevation_std", "elevation_std_sq"]
    
    for col in required_cols
        if !(Symbol(col) in propertynames(dml_data))
            error("Missing required column for DML: $col")
        end
    end
    
    # Verify MESF columns exist
    for mesf_col in mesf_columns
        if !(Symbol(mesf_col) in propertynames(dml_data))
            error("Missing MESF column: $mesf_col")
        end
    end
    
    # Define confounders and treatments
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    confounders = vcat(climate_features, elevation_features, mesf_columns)
    
    treatments = Dict(
        "plants" => "plants_richness",       # Raw counts for causal interpretation
        "vertebrates" => "vertebrates_richness"   # Raw counts
    )
    
    outcome = "fungi_richness"  # Raw counts (created in preprocessing)
    
    # Initialize and fit TRUE DML with cross-fitting and LightGBM
    println("\nRunning TRUE DML with:")
    println("  Outcome: $outcome (raw counts)")
    println("  Treatments: plants_richness, vertebrates_richness (raw counts)")
    println("  Confounders: $(length(confounders)) variables including $(length(mesf_columns)) MESF")
    println("  Method: Cross-fitting with 5 folds, LightGBM nuisance functions")
    
    # Use ML config for DML parameters
    ml_config_path = "src/config/ml/ml_config.yml"
    dml = MultivariableDML(n_folds=5, config_path=ml_config_path, verbose=true)
    
    try
        fit_dml!(dml, dml_data, confounders, treatments, outcome)
        
        # Store results in consistent format
        fitted_models["method3_dml"] = Dict(
            "coefficients" => dml.coefficients,
            "coef_names" => dml.treatment_names,
            "std_errors" => dml.std_errors,
            "vcov" => dml.vcov,
            "n_obs" => dml.n_obs,
            "method_note" => "TRUE DML with cross-fitting and LightGBM"
        )
        
        # Save detailed results
        mkpath("results/julia_models")
        save_results(dml, "results/julia_models/method3_dml_results.json")
        
        println("\n✓ Method 3 TRUE DML completed successfully")
        
    catch e
        println("\n⚠ DML failed: $e")
        println("Stack trace:")
        println(stacktrace())
        
        # Store error information
        fitted_models["method3_dml"] = Dict(
            "error" => string(e),
            "method_note" => "DML failed - check data preprocessing"
        )
    end
    
    return fitted_models
end

# ============================================================================
# RESULTS EXPORT
# ============================================================================

"""
    save_results_to_disk(results::Dict)

Save all model results to disk for Python to load.
"""
function save_results_to_disk(results::Dict)
    global output_dir
    
    mkpath(output_dir)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    full_results = Dict(
        "models" => results["models"],
        "metadata" => results["metadata"],
        "config" => get(results, "config", Dict()),
        "timestamp" => timestamp,
        "julia_version" => string(VERSION)
    )
    
    # Save as JSON
    json_file = joinpath(output_dir, "julia_results_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, full_results, 4)
    end
    println("\n✅ Results saved to: $json_file")
    
    # Save as JLD2
    jld2_file = joinpath(output_dir, "julia_results_$(timestamp).jld2")
    @save jld2_file full_results
    println("✅ Full precision data saved to: $jld2_file")
    
    # Create latest symlink
    latest_json = joinpath(output_dir, "julia_results_latest.json")
    if isfile(latest_json) || islink(latest_json)
        rm(latest_json)
    end
    symlink(basename(json_file), latest_json)
    
    return json_file
end

# ============================================================================
# MAIN HYBRID PIPELINE
# ============================================================================

"""
    run_hybrid_pipeline(data_path; kwargs...)

Run the complete hybrid pipeline with Python MESF + Julia GLM.
Note: data_path should be parquet for MESF computation, but we'll convert to Arrow for Julia.
"""
function run_hybrid_pipeline(data_path::String;
                           spatial_grid_step::Int=1,
                           compute_mesf::Bool=true,
                           n_mesf::Int=10,
                           mesf_path::String="",
                           config_path::String="",
                           save_dir::String="")
    
    global output_dir, config, fitted_models
    
    println("\n" * "="^70)
    println("HYBRID PIPELINE: PYTHON MESF + JULIA GLM")
    println("="^70)
    
    # Load configuration
    if !isempty(config_path)
        load_config(config_path)
    end
    
    # Set output directory
    if !isempty(save_dir)
        output_dir = save_dir
    end
    
    # Step 1: Compute or load MESF
    if compute_mesf && isempty(mesf_path)
        println("\nStep 1: Computing MESF with Python...")
        # Use parquet file for Python MESF computation
        mesf_data_path = replace(data_path, ".arrow" => ".parquet")
        if !isfile(mesf_data_path)
            mesf_data_path = data_path  # Fallback to original if no parquet
        end
        mesf_path = compute_mesf_with_python(
            mesf_data_path,
            spatial_grid_step=spatial_grid_step,
            n_mesf=n_mesf,
            output_format="npy"
        )
    elseif !isempty(mesf_path)
        println("\nStep 1: Using provided MESF file: $mesf_path")
    else
        println("\nStep 1: Skipping MESF computation")
        mesf_path = ""
    end
    
    # Step 2: Load data with MESF
    println("\nStep 2: Loading data with MESF integration...")
    metadata = load_with_python_mesf(data_path, mesf_path, spatial_grid_step)
    
    # Step 3: Fit all models
    println("\nStep 3: Fitting statistical models...")
    
    # Methods 1-2 use log-transformed data (already preprocessed)
    fit_method1_models()
    fit_method2_models()
    
    # Method 3 needs raw counts - check and handle
    fit_method3_dml()
    
    # Prepare results
    results = Dict(
        "metadata" => metadata,
        "models" => fitted_models,
        "config" => config
    )
    
    # Save results
    saved_file = save_results_to_disk(results)
    
    println("\n" * "="^70)
    println("✅ HYBRID PIPELINE COMPLETED SUCCESSFULLY")
    println("="^70)
    
    return results
end

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if length(ARGS) > 0
    command = ARGS[1]
    
    if command == "run"
        # Full hybrid pipeline
        data_path = ARGS[2]
        spatial_grid_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
        compute_mesf = length(ARGS) > 3 ? parse(Bool, ARGS[4]) : true
        n_mesf = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 10
        mesf_path = length(ARGS) > 5 ? ARGS[6] : ""
        config_path = length(ARGS) > 6 ? ARGS[7] : ""
        save_dir = length(ARGS) > 7 ? ARGS[8] : "results/julia_models"
        
        results = run_hybrid_pipeline(
            data_path,
            spatial_grid_step=spatial_grid_step,
            compute_mesf=compute_mesf,
            n_mesf=n_mesf,
            mesf_path=mesf_path,
            config_path=config_path,
            save_dir=save_dir
        )
        
        println("\n✅ Pipeline complete. Results saved to: $save_dir")
        
    elseif command == "compute_mesf"
        # Just compute MESF
        data_path = ARGS[2]
        spatial_grid_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
        n_mesf = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 10
        
        mesf_file = compute_mesf_with_python(
            data_path,
            spatial_grid_step=spatial_grid_step,
            n_mesf=n_mesf
        )
        println("MESF saved to: $mesf_file")
        
    elseif command == "method1"
        # Run Method 1 only (H0 vs H1)
        data_path = ARGS[2]
        
        # Load config to get spatial_grid_step
        config_path = joinpath(dirname(@__DIR__), "..", "config", "ml", "ml_config.yml")
        config = load_config(config_path)
        spatial_grid_step = config["regression_analysis"]["data"]["spatial_grid_step"]
        
        mesf_path = length(ARGS) > 2 ? ARGS[3] : ""
        save_dir = length(ARGS) > 3 ? ARGS[4] : "results/julia_models"
        
        println("\n" * "="^70)
        println("RUNNING METHOD 1 ONLY (H0-H1 TESTING)")
        println("="^70)
        
        # Load data with log transforms for GLM
        metadata = load_with_python_mesf(data_path, mesf_path, spatial_grid_step, method="method1")
        
        # Run Method 1 only
        fit_method1_models()
        
        # Save results
        results = Dict(
            "metadata" => metadata,
            "models" => fitted_models,
            "config" => config
        )
        saved_file = save_results_to_disk(results)
        
        println("\n✅ Method 1 analysis complete. Results saved to: $save_dir")
        
    elseif command == "method2"
        # Run Method 2 only (Interactions)
        data_path = ARGS[2]
        
        # Load config to get spatial_grid_step
        config_path = joinpath(dirname(@__DIR__), "..", "config", "ml", "ml_config.yml")
        config = load_config(config_path)
        spatial_grid_step = config["regression_analysis"]["data"]["spatial_grid_step"]
        
        mesf_path = length(ARGS) > 2 ? ARGS[3] : ""
        save_dir = length(ARGS) > 3 ? ARGS[4] : "results/julia_models"
        
        println("\n" * "="^70)
        println("RUNNING METHOD 2 ONLY (INTERACTIONS)")
        println("="^70)
        
        # Load data with log transforms for GLM
        metadata = load_with_python_mesf(data_path, mesf_path, spatial_grid_step, method="method2")
        
        # Run Method 2 only
        fit_method2_models()
        
        # Save results
        results = Dict(
            "metadata" => metadata,
            "models" => fitted_models,
            "config" => config
        )
        saved_file = save_results_to_disk(results)
        
        println("\n✅ Method 2 analysis complete. Results saved to: $save_dir")
        
    elseif command == "method3" || command == "dml"
        # Run Method 3 only (DML)
        data_path = ARGS[2]
        
        # Load config to get spatial_grid_step
        config_path = joinpath(dirname(@__DIR__), "..", "config", "ml", "ml_config.yml")
        config = load_config(config_path)
        spatial_grid_step = config["regression_analysis"]["data"]["spatial_grid_step"]
        
        mesf_path = length(ARGS) > 2 ? ARGS[3] : ""
        save_dir = length(ARGS) > 3 ? ARGS[4] : "results/julia_models"
        
        # Load multivariable_dml.jl at top level to avoid world age issues
        include(joinpath(@__DIR__, "multivariable_dml.jl"))
        
        println("\n" * "="^70)
        println("RUNNING METHOD 3 ONLY (DML)")
        println("="^70)
        
        # Load and preprocess data for DML (raw counts)
        metadata = load_with_python_mesf(data_path, mesf_path, spatial_grid_step, method="method3")
        
        # Run Method 3 DML only
        fit_method3_dml()
        
        # Save results
        results = Dict(
            "metadata" => metadata,
            "models" => fitted_models,
            "config" => config
        )
        saved_file = save_results_to_disk(results)
        
        println("\n✅ Method 3 (DML) analysis complete. Results saved to: $save_dir")
        
    else
        println("Unknown command: $command")
        println("Available commands: run, method1, method2, method3/dml, compute_mesf")
    end
else
    println("Hybrid Julia Pipeline for Biodiversity Analysis")
    println("Usage: julia julia_hybrid_pipeline.jl <command> [args...]")
    println("\nCommands:")
    println("  run <data_path> [spatial_step] [compute_mesf] [n_mesf] [mesf_path] [config_path] [save_dir]")
    println("       Run all methods (1, 2, and 3)")
    println()
    println("  method1 <data_path> [spatial_step] [mesf_path] [save_dir]")
    println("       Run Method 1 only: H0 vs H1 hypothesis testing")
    println()
    println("  method2 <data_path> [spatial_step] [mesf_path] [save_dir]")
    println("       Run Method 2 only: Interaction analysis")
    println()
    println("  method3 <data_path> [spatial_step] [mesf_path] [save_dir]")
    println("  dml <data_path> [spatial_step] [mesf_path] [save_dir]")
    println("       Run Method 3 only: Double Machine Learning (DML)")
    println()
    println("  compute_mesf <data_path> [spatial_step] [n_mesf]")
    println("       Compute MESF eigenvectors only")
    println()
    println("Default spatial_step: 1000 (for individual methods)")
    println("Default save_dir: results/julia_models")
end