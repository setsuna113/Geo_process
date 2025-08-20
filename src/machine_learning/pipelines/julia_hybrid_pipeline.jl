#!/usr/bin/env julia
# Output structure: see docs/julia_pipeline_output_structure.md

using DataFrames, Arrow, Statistics, GLM, Distributions, JSON, StatsBase
using LinearAlgebra, YAML, JLD2, Dates, NPZ
using StatsModels, HypothesisTests
using NPZ  # For saving binary numpy files
using Distributions: Normal, Chisq, cdf
using LinearAlgebra: dot, inv, cond

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

function load_data_with_mesf(data_path::String, spatial_step::Int, method::String)
    # Load data
    println("Loading data from: $data_path")
    df = DataFrame(Arrow.Table(data_path), copycols=false)  # Don't copy, save memory
    println("Loaded $(nrow(df)) rows")
    
    # Sample if needed
    if spatial_step > 1
        df = df[1:spatial_step:end, :]
        println("Sampled to $(nrow(df)) rows")
    end
    
    # Only convert necessary columns to Float32
    println("Converting columns to Float32...")
    important_cols = [:bio01, :bio04, :bio12, :bio15, :elevation,
                      :am_fungi_richness, :ecm_fungi_richness,
                      :plants_richness, :plant_richness,
                      :vertebrates_richness, :animal_richness, :terrestrial_richness]
    
    for col in important_cols
        if col in propertynames(df) && eltype(df[!, col]) <: Number
            df[!, col] = Float32.(df[!, col])
        end
    end
    
    # Load or compute MESF
    cache_dir = "/scratch/yl998/.cache/mesf"
    mesf_path = joinpath(cache_dir, "mesf_step$(spatial_step)_n10.npy")
    mesf_columns = String[]
    
    if isfile(mesf_path)
        println("Loading cached MESF: $mesf_path ($(round(filesize(mesf_path)/1e9, digits=2)) GB)")
        mesf_array = npzread(mesf_path)
        n_mesf = size(mesf_array, 2)
        println("Adding $n_mesf MESF columns...")
        
        # Check dimensions match
        if size(mesf_array, 1) != nrow(df)
            error("MESF dimensions mismatch: $(size(mesf_array, 1)) vs $(nrow(df)) data rows")
        end
        
        for i in 1:n_mesf
            df[!, Symbol("mesf_$i")] = Float32.(mesf_array[:, i])
            push!(mesf_columns, "mesf_$i")
        end
        println("MESF columns added")
    else
        println("MESF not found at: $mesf_path")
        # Compute MESF if needed - simplified
        mesf_columns = compute_mesf(df, spatial_step)
    end
    
    # Preprocess based on method
    df = preprocess_data(df, method)
    
    return df, mesf_columns
end

function preprocess_data(df::DataFrame, method::String)
    # Remove missing - need to copy for mutation if Arrow-backed
    if any(ismissing, Iterators.flatten(eachcol(df)))
        df = dropmissing(df)  # Creates a copy without missing
    end
    
    # Create fungi_richness
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
    
    # Standardize climate (2SD method)
    for var in [:bio01, :bio04, :bio12, :bio15, :elevation]
        if var in propertynames(df)
            col_std = Symbol(string(var) * "_std")
            μ = mean(skipmissing(df[!, var]))
            σ = std(skipmissing(df[!, var]))
            df[!, col_std] = (df[!, var] .- μ) ./ (2σ)
        end
    end
    
    # Add elevation squared
    df.elevation_std_sq = df.elevation_std .^ 2
    
    # Method-specific transforms
    if method in ["method3", "dml"]
        # Keep raw counts for DML
        println("Keeping raw counts for DML")
        
        # Add interaction term for DML (plants × vertebrates)
        # This aligns with the paper's recommendation for 2 treatments
        df.plants_x_vertebrates = df.plants_richness .* df.vertebrates_richness
        println("Added interaction term: plants_x_vertebrates for DML")
    else
        # GLM needs integers and log transforms
        df.fungi_richness = round.(Int, df.fungi_richness)
        df.plants_richness_log = log.(df.plants_richness .+ 1)
        df.vertebrates_richness_log = log.(df.vertebrates_richness .+ 1)
    end
    
    return df
end

function compute_mesf(df::DataFrame, spatial_step::Int)
    # Call Python MESF processor
    temp_path = "/tmp/temp_mesf_$(hash(df))_$(spatial_step).arrow"
    Arrow.write(temp_path, df)
    
    mesf_processor = joinpath(@__DIR__, "mesf_processor.py")
    cache_dir = "/scratch/yl998/.cache/mesf"
    
    # Simple command execution
    cmd = `$(homedir())/anaconda3/envs/geo/bin/python $mesf_processor 
           $temp_path --spatial-step $spatial_step --n-eigenvectors 10 
           --n-jobs 120 --output-dir $cache_dir`
    
    output = read(cmd, String)
    rm(temp_path, force=true)
    
    # Parse output to get file path
    lines = split(strip(output), '\n')
    mesf_file = strip(lines[end])
    
    # Load MESF
    mesf_columns = String[]
    if isfile(mesf_file)
        mesf_array = npzread(mesf_file)
        for i in 1:size(mesf_array, 2)
            df[!, Symbol("mesf_$i")] = Float32.(mesf_array[:, i])
            push!(mesf_columns, "mesf_$i")
        end
    end
    
    return mesf_columns
end

# ============================================================================
# MODEL FITTING - Clean and simple
# ============================================================================

function extract_diagnostics(model, model_name::String, state::PipelineState)
    # Calculate residuals but DON'T put in JSON
    y = response(model)
    μ = predict(model)
    θ = model.model.rr.d.r
    
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
        "theta" => model.model.rr.d.r,
        
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
        "null_deviance" => nulldeviance(model),
        "null_loglik" => nullloglikelihood(model),
        "dispersion" => dispersion(model.model)  # Need the actual GLM, not wrapper
    )
    
    # Inference (NEW - flat with prefixes)
    try
        ct = coeftable(model)
        diagnostics["coef_z_values"] = ct.cols[3]
        diagnostics["coef_p_values"] = ct.cols[4]
        diagnostics["confint_95"] = confint(model, 0.95)
    catch e
        # If inference fails, just skip those fields
        println("Note: Inference stats not available: $e")
    end
    
    # Multicollinearity (NEW - flat with prefixes)
    try
        vif_values = gvif(model)
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

function fit_method1(state::PipelineState)
    results = ModelResult[]
    
    # Features
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation = ["elevation_std", "elevation_std_sq"]
    base = vcat(climate, elevation, state.mesf_columns)
    biotic = ["plants_richness_log", "vertebrates_richness_log"]
    
    # Null model
    formula_null = Term(:fungi_richness) ~ sum(Term.(Symbol.(base)))
    model_null = negbin(formula_null, state.data, LogLink())
    push!(results, ModelResult("method1_null", model_null, extract_diagnostics(model_null, "method1_null", state)))
    
    # Full model
    formula_full = Term(:fungi_richness) ~ sum(Term.(Symbol.(vcat(base, biotic))))
    model_full = negbin(formula_full, state.data, LogLink())
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

function fit_method2(state::PipelineState)
    results = ModelResult[]
    
    # Add categorical features
    df = state.data
    median_bio4 = median(df.bio04_std)
    df.temperate = Int.(df.bio04_std .> median_bio4)
    
    median_plants = median(df.plants_richness)
    df.plant_high = Int.(df.plants_richness .> median_plants)
    
    df.temperate_x_plant_high = df.temperate .* df.plant_high
    df.bio04_std_x_plants_log = df.bio04_std .* df.plants_richness_log
    
    # Features
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation = ["elevation_std", "elevation_std_sq"]
    biodiv = ["plants_richness_log", "vertebrates_richness_log"]
    base = vcat(climate, elevation, biodiv, state.mesf_columns)
    
    # Continuous interaction
    cont_features = vcat(base, ["bio04_std_x_plants_log"])
    formula_cont = Term(:fungi_richness) ~ sum(Term.(Symbol.(cont_features)))
    model_cont = negbin(formula_cont, df, LogLink())
    push!(results, ModelResult("method2_continuous", model_cont, extract_diagnostics(model_cont, "method2_continuous", state)))
    
    # Categorical interaction (keep continuous main effects, add ONLY interaction term)
    cat_features = vcat(
        base,  # Keep all base features including bio04_std and plants_richness_log
        ["temperate_x_plant_high"]  # ONLY the interaction, no main effects
    )
    formula_cat = Term(:fungi_richness) ~ sum(Term.(Symbol.(cat_features)))
    model_cat = negbin(formula_cat, df, LogLink())
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
    
    # Setup
    climate = ["bio01_std", "bio04_std", "bio12_std", "bio15_std"]
    elevation = ["elevation_std", "elevation_std_sq"]
    confounders = vcat(climate, elevation, state.mesf_columns)
    
    # Include interaction term as per DML paper recommendation for 2 treatments
    # The paper notation: A = [a, b], A° = [a×b]
    treatments = Dict(
        "plants" => "plants_richness",
        "vertebrates" => "vertebrates_richness",
        "plants_x_vertebrates" => "plants_x_vertebrates"  # Interaction term
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

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

function run_pipeline(method::String, data_path::String, spatial_step::Int=1000)
    # Load config
    config_path = "src/config/ml/ml_config.yml"
    config = YAML.load_file(config_path)
    
    # Load data
    println("="^60)
    println("Loading data for $method")
    println("="^60)
    
    data, mesf_columns = load_data_with_mesf(data_path, spatial_step, method)
    
    # Create state
    state = PipelineState(
        data,
        mesf_columns,
        config,
        "/scratch/yl998/.cache/mesf",
        "/scratch/yl998/julia"  # Use scratch for large outputs
    )
    
    # Fit models
    all_results = Dict{String, Any}()
    all_models = Dict{String, Any}()
    
    if method == "method1"
        models, lrt = fit_method1(state)
        for m in models
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        all_results["method1_lr_test"] = lrt
        
    elseif method == "method2"
        # First fit method1 for comparisons
        models1, _ = fit_method1(state)
        for m in models1
            all_models[m.name] = m.model
        end
        
        models2 = fit_method2(state)
        for m in models2
            all_results[m.name] = m.diagnostics
            all_models[m.name] = m.model
        end
        
        # LRT comparisons - both models are now properly nested
        if haskey(all_models, "method1_full")
            # Compare continuous interaction model
            lrt_a = lrtest(all_models["method1_full"], all_models["method2_continuous"])
            all_results["lrt_method1_vs_method2a"] = Dict(
                "lr_statistic" => lrt_a.deviance[2],
                "df" => lrt_a.dof[2],
                "p_value" => lrt_a.pval[2],
                "comparison" => "method1_full vs method2_continuous (adds continuous interaction)"
            )
            
            # Compare categorical interaction model  
            lrt_b = lrtest(all_models["method1_full"], all_models["method2_categorical"])
            all_results["lrt_method1_vs_method2b"] = Dict(
                "lr_statistic" => lrt_b.deviance[2],
                "df" => lrt_b.dof[2],
                "p_value" => lrt_b.pval[2],
                "comparison" => "method1_full vs method2_categorical (adds categorical interaction)"
            )
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
    
    if method in ["method1", "method2", "method3", "dml"]
        data_path = length(ARGS) > 1 ? ARGS[2] : error("Data path required")
        spatial_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000
        
        # For method3/dml, load the DML module early
        if method in ["method3", "dml"]
            include(joinpath(@__DIR__, "multivariable_dml_final.jl"))
        end
        
        run_pipeline(method, data_path, spatial_step)
        
    else
        println("Usage: julia $PROGRAM_FILE <method> <data_path> [spatial_step]")
        println("Methods: method1, method2, method3/dml")
    end
else
    println("Usage: julia $PROGRAM_FILE <method> <data_path> [spatial_step]")
end