#!/usr/bin/env julia
"""
Hybrid Julia Pipeline for Biodiversity Analysis
Uses Python's optimized MESF computation and Julia's GLM fitting.
This is the pragmatic solution that leverages existing optimizations.
"""

using DataFrames, Arrow, Statistics, GLM, Distributions, JSON, StatsBase
using LinearAlgebra, Random
using YAML, JLD2, SHA, Dates
using NPZ  # For loading numpy arrays

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
# CONFIGURATION
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

# ============================================================================
# DATA LOADING WITH PYTHON MESF INTEGRATION
# ============================================================================

"""
    load_with_python_mesf(data_path, mesf_path, spatial_grid_step=1)

Load data and integrate pre-computed Python MESF eigenvectors.
This is the core of the hybrid approach - use Python's optimized MESF.
"""
function load_with_python_mesf(data_path::String, mesf_path::String, spatial_grid_step::Int=1)
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
    
    # Load MESF eigenvectors from Python
    if isfile(mesf_path)
        println("Loading MESF eigenvectors from: $mesf_path")
        
        if endswith(mesf_path, ".npy")
            # Load numpy array
            mesf_array = npzread(mesf_path)
            n_mesf = size(mesf_array, 2)
            
            # Verify dimensions match
            if size(mesf_array, 1) != nrow(df)
                error("MESF dimensions mismatch: $(size(mesf_array, 1)) vs $(nrow(df)) data rows")
            end
            
            # Add MESF columns to dataframe
            for i in 1:n_mesf
                df[!, Symbol("mesf_$i")] = mesf_array[:, i]
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
        println("Warning: MESF file not found, proceeding without spatial features")
        mesf_columns = String[]
    end
    
    # Preprocess the data
    df = preprocess_data!(df)
    
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
    preprocess_data!(df::DataFrame)

Preprocess the dataframe with standardization and transformations.
"""
function preprocess_data!(df::DataFrame)
    # Remove missing values
    initial_rows = nrow(df)
    dropmissing!(df)
    println("Removed $(initial_rows - nrow(df)) rows with missing values")
    
    # Create fungi_richness
    if :am_fungi_richness in propertynames(df) && :ecm_fungi_richness in propertynames(df)
        df.fungi_richness = round.(Int, df.am_fungi_richness .+ df.ecm_fungi_richness)
        println("Created fungi_richness from AM + ECM")
    elseif !(:fungi_richness in propertynames(df))
        error("Cannot create fungi_richness: required columns missing")
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
    
    # Log transform biodiversity variables
    if :plant_richness in propertynames(df)
        df.plants_richness_log = log.(df.plant_richness .+ 1)
    elseif :plants_richness in propertynames(df)
        df.plants_richness_log = log.(df.plants_richness .+ 1)
    end
    
    if :animal_richness in propertynames(df)
        df.vertebrates_richness_log = log.(df.animal_richness .+ 1)
    elseif :vertebrates_richness in propertynames(df)
        df.vertebrates_richness_log = log.(df.vertebrates_richness .+ 1)
    end
    
    println("✓ Log-transformed biodiversity variables")
    
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
    python_cmd = `$(homedir())/anaconda3/envs/geo/bin/python $(mesf_processor_path) 
        $(data_path) 
        --spatial-step $(spatial_grid_step) 
        --n-eigenvectors $(n_mesf) 
        --n-jobs 120 
        --output-dir results/mesf_cache`
    
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
    
    # Store results
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
        "theta" => glm_null.rr.d.r
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
        "theta" => glm_full.rr.d.r
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
    global current_data, mesf_columns, fitted_models
    
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
    
    # Store results
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

"""
    fit_method3_dml()

Fit Method 3: Simplified Double Machine Learning.
"""
function fit_method3_dml()
    global current_data, mesf_columns, fitted_models
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    println("\n" * "="^60)
    println("METHOD 3: DOUBLE MACHINE LEARNING")
    println("="^60)
    
    # Define variables
    outcome = :fungi_richness
    treatments = [:plants_richness_log, :vertebrates_richness_log]
    
    # Confounders
    climate_features = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation_features = ["elevation_std", "elevation_std_sq"]
    confounders = vcat(climate_features, elevation_features, mesf_columns)
    
    println("Implementing simplified DML with residualization...")
    
    # Ensure numeric columns are Float64
    for col in names(current_data)
        if eltype(current_data[!, col]) <: Number && !(eltype(current_data[!, col]) <: Integer)
            current_data[!, col] = Float64.(current_data[!, col])
        end
    end
    
    # Step 1: Residualize outcome
    formula_y = Term(outcome) ~ sum(Term.(Symbol.(confounders)))
    model_y = lm(formula_y, current_data)
    residuals_y = Float64.(residuals(model_y))
    
    # Step 2: Residualize treatments
    residuals_treatments = Dict()
    for treatment in treatments
        formula_t = Term(treatment) ~ sum(Term.(Symbol.(confounders)))
        model_t = lm(formula_t, current_data)
        residuals_treatments[treatment] = Float64.(residuals(model_t))
    end
    
    # Step 3: Final regression
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
    
    println("\n✓ Method 3 DML fitted successfully")
    println("  Plants effect: $(round(coef(model_dml)[2], digits=4))")
    println("  Vertebrates effect: $(round(coef(model_dml)[3], digits=4))")
    
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
    fit_method1_models()
    fit_method2_models()
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
        
    else
        println("Unknown command: $command")
        println("Available commands: run, compute_mesf")
    end
else
    println("Hybrid Julia Pipeline for Biodiversity Analysis")
    println("Usage: julia julia_hybrid_pipeline.jl <command> [args...]")
    println("\nCommands:")
    println("  run <data_path> [spatial_step] [compute_mesf] [n_mesf] [mesf_path] [config_path] [save_dir]")
    println("  compute_mesf <data_path> [spatial_step] [n_mesf]")
end