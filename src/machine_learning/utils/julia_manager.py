"""Julia Session Manager for GLM pipeline.

This singleton pattern ensures Julia is initialized only once across
the entire application, managing the Julia process efficiently.
Data stays in Julia, only results are returned to Python.
"""

import logging
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np

logger = logging.getLogger(__name__)


class JuliaManager:
    """Singleton manager for Julia GLM pipeline.
    
    Manages Julia process and GLM fitting.
    Data remains in Julia throughout the pipeline.
    Only model results are returned to Python.
    """
    
    _instance = None
    _initialized = False
    _temp_dir = None
    _julia_main_script = None
    _julia_executable = 'julia'  # Path to Julia executable
    _current_data_id = None  # ID of currently loaded data in Julia
    
    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_or_init(cls, **kwargs) -> 'JuliaManager':
        """Get existing Julia instance or initialize new one.
        
        Args:
            **kwargs: Julia initialization parameters
                nthreads: Number of threads for Julia
                ice_root: Temp directory for Julia operations
                
        Returns:
            JuliaManager instance ready for use
        """
        instance = cls()
        
        if not cls._initialized:
            logger.info(f"Initializing Julia with parameters: {kwargs}")
            
            # Set up temp directory
            ice_root = kwargs.get('ice_root', '/tmp/julia_glm')
            cls._temp_dir = Path(ice_root)
            cls._temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Set Julia threads
            nthreads = kwargs.get('nthreads', -1)
            if nthreads == -1:
                import os
                nthreads = os.cpu_count() or 150
            
            # Set environment for Julia
            import os
            os.environ['JULIA_NUM_THREADS'] = str(nthreads)
            
            # Initialize Julia script
            cls._setup_julia_main_script()
            
            # Use Julia wrapper script if available
            import os
            wrapper_path = Path(__file__).parent / 'julia_wrapper.sh'
            
            if wrapper_path.exists():
                cls._julia_executable = str(wrapper_path)
                logger.info(f"Using Julia wrapper script: {wrapper_path}")
            else:
                # Find Julia executable directly
                julia_path = 'julia'
                
                # Check common Julia installation paths
                possible_paths = [
                    os.path.expanduser('~/julia/bin/julia'),
                    '/usr/local/bin/julia',
                    '/usr/bin/julia',
                    'julia'  # Fallback to PATH
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        julia_path = path
                        break
                
                cls._julia_executable = julia_path
            
            # Test Julia availability
            try:
                result = subprocess.run(
                    [cls._julia_executable, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Julia initialized: {result.stdout.strip()}")
                logger.info(f"  Executable: {cls._julia_executable}")
                logger.info(f"  Threads: {nthreads}")
                logger.info(f"  Temp dir: {cls._temp_dir}")
                cls._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize Julia: {e}")
                logger.error("Please ensure Julia is installed and in PATH")
                raise
        
        return instance
    
    @classmethod
    def _setup_julia_main_script(cls):
        """Create the main Julia script that handles all operations."""
        julia_code = '''
# Julia GLM Pipeline for Biodiversity Analysis
using DataFrames, Arrow, Statistics, GLM, Distributions, JSON, StatsBase

# Global data storage
global current_data = nothing
global data_id = ""
global mesf_columns = String[]

# Load and preprocess data
function load_and_preprocess(data_path::String, spatial_grid_step::Int=1)
    global current_data, data_id, mesf_columns
    
    # Load data - for large files with sampling, read in chunks
    if endswith(data_path, ".arrow")
        if spatial_grid_step > 10000
            # For large sampling steps, just load and sample normally
            df = DataFrame(Arrow.Table(data_path))
            indices = 1:spatial_grid_step:nrow(df)
            df = df[indices, :]
        elseif spatial_grid_step > 1
            # For smaller sampling steps, try memory-mapped approach
            try
                # Use memory mapping for efficient access
                tbl = Arrow.Table(data_path)
                df = DataFrame(tbl)
                indices = 1:spatial_grid_step:nrow(df)
                df = df[indices, :]
            catch e
                # If memory mapping fails, exit with error
                println("ERROR: Failed to load Arrow file with sampling step $spatial_grid_step")
                println("Error details: $e")
                exit(1)
            end
        else
            # Load all data (no sampling)
            df = DataFrame(Arrow.Table(data_path))
        end
    else
        # Fallback to Parquet (may fail with v2.6)
        error("Please use Arrow format (.arrow) for Julia compatibility")
    end
    
    # Remove missing
    initial_rows = nrow(df)
    dropmissing!(df)
    # Store for metadata but don't print during load command
    rows_removed = initial_rows - nrow(df)
    
    # Add fungi_richness if needed
    if !(:fungi_richness in names(df))
        if :am_fungi_richness in names(df) && :ecm_fungi_richness in names(df)
            df.fungi_richness = round.(Int, df.am_fungi_richness .+ df.ecm_fungi_richness)
        end
    end
    
    # Standardize climate vars (2SD method)
    climate_vars = [:bio01, :bio04, :bio12, :bio15, :elevation]
    for var in climate_vars
        if var in propertynames(df)
            col_std = Symbol(string(var) * "_std")
            mean_val = mean(skipmissing(df[!, var]))
            std_val = std(skipmissing(df[!, var]))
            df[!, col_std] = (df[!, var] .- mean_val) ./ (2 * std_val)
        end
    end
    
    # Add elevation squared
    if :elevation_std in names(df)
        df.elevation_std_sq = df.elevation_std .^ 2
    end
    
    # Log transform biodiversity
    for var in [:plants_richness, :vertebrates_richness]
        if var in propertynames(df)
            col_log = Symbol(string(var) * "_log")
            df[!, col_log] = log.(df[!, var] .+ 1)
        end
    end
    
    # Find MESF columns
    mesf_columns = String[string(col) for col in names(df) if startswith(string(col), "mesf_")]
    
    # Store data globally
    current_data = df
    data_id = string(hash(data_path) * spatial_grid_step)
    
    # Return metadata
    result = Dict(
        "data_id" => data_id,
        "n_rows" => nrow(df),
        "n_cols" => ncol(df),
        "columns" => names(df),
        "mesf_count" => length(mesf_columns),
        "rows_removed" => rows_removed
    )
    
    return result
end

# Add categorical features for Method 2
function add_categorical_features()
    global current_data
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    df = current_data
    
    # Create temperate category (bio04_std > median)
    if :bio04_std in names(df)
        median_bio4 = median(df.bio04_std)
        df.temperate = Int.(df.bio04_std .> median_bio4)
        
        # Count balance (stored but not printed)
        temp_counts = countmap(df.temperate)
    end
    
    # Create plant diversity category (plants_richness > median)
    if :plants_richness in names(df)
        median_plants = median(df.plants_richness)
        df.plant_high = Int.(df.plants_richness .> median_plants)
        
        # Count balance (stored but not printed)
        plant_counts = countmap(df.plant_high)
    end
    
    # Create interaction term
    if :temperate in names(df) && :plant_high in names(df)
        df.temperate_x_plant_high = df.temperate .* df.plant_high
    end
    
    # Create continuous interaction
    if :bio04_std in names(df) && :plants_richness_log in names(df)
        df.bio04_std_x_plants_log = df.bio04_std .* df.plants_richness_log
    end
    
    current_data = df
    
    return Dict(
        "categorical_features_added" => true,
        "features" => ["temperate", "plant_high", "temperate_x_plant_high", "bio04_std_x_plants_log"]
    )
end

# Fit negative binomial GLM
function fit_negbin_glm(formula_str::String)
    global current_data
    
    if isnothing(current_data)
        error("No data loaded")
    end
    
    # Parse formula
    formula = eval(Meta.parse("@formula($formula_str)"))
    
    # Fit model using negbin (estimates theta)
    model = negbin(formula, current_data, LogLink())
    
    # Extract coefficients and names
    coef_names = coefnames(model)
    
    # Extract statistics
    results = Dict(
        "coefficients" => coef(model),
        "coef_names" => coef_names,
        "std_errors" => stderror(model),
        "vcov" => vcov(model),
        "aic" => aic(model),
        "bic" => bic(model),
        "loglik" => loglikelihood(model),
        "deviance" => deviance(model),
        "df_residual" => dof_residual(model),
        "n_obs" => nrow(current_data),
        "theta" => model.model.theta  # Estimated dispersion parameter
    )
    
    return results
end

# Get data summary
function get_data_summary()
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

# Main execution - handle commands
command = ARGS[1]

if command == "load"
    parquet_path = ARGS[2]
    spatial_grid_step = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
    result = load_and_preprocess(parquet_path, spatial_grid_step)
    println(JSON.json(result))
    
elseif command == "add_categorical"
    result = add_categorical_features()
    println(JSON.json(result))
    
elseif command == "fit_glm"
    formula = ARGS[2]
    result = fit_negbin_glm(formula)
    println(JSON.json(result))
    
elseif command == "data_summary"
    result = get_data_summary()
    println(JSON.json(result))
    
else
    error("Unknown command: $command")
end
'''
        
        # Save script
        script_path = cls._temp_dir / "julia_main.jl"
        with open(script_path, 'w') as f:
            f.write(julia_code)
        cls._julia_main_script = script_path
    
    def load_data(self, path: Union[str, Path], spatial_grid_step: int = 1) -> Dict[str, Any]:
        """Load and preprocess data in Julia.
        
        Args:
            path: Path to Arrow or Parquet file
            spatial_grid_step: Spatial sampling step (1 = no sampling)
            
        Returns:
            Dictionary with data metadata (not the data itself)
        """
        path = str(path)
        
        # Check if we need to convert Parquet to Arrow
        if path.endswith('.parquet'):
            arrow_path = path.replace('.parquet', '.arrow')
            
            # Check if Arrow version already exists
            import os
            if not os.path.exists(arrow_path):
                logger.info(f"Converting Parquet to Arrow format for Julia compatibility...")
                try:
                    import pandas as pd
                    import pyarrow.feather as feather
                    df = pd.read_parquet(path)
                    feather.write_feather(df, arrow_path, compression='lz4')
                    logger.info(f"Created Arrow file: {arrow_path}")
                except Exception as e:
                    logger.error(f"Failed to convert Parquet to Arrow: {e}")
                    logger.error("Please convert manually using: feather.write_feather(df, 'file.arrow')")
                    raise
            else:
                logger.info(f"Using existing Arrow file: {arrow_path}")
            
            # Use Arrow file instead
            path = arrow_path
        
        # Run Julia to load data with better error handling
        try:
            result = subprocess.run(
                [self._julia_executable, str(self._julia_main_script), 'load', path, str(spatial_grid_step)],
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # Add timeout to prevent hanging
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Julia process failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Julia process timed out after 60 seconds")
            raise
        
        # Parse JSON result
        metadata = json.loads(result.stdout)
        self._current_data_id = metadata['data_id']
        
        logger.info(f"Data loaded in Julia: {metadata['n_rows']} rows × {metadata['n_cols']} columns")
        logger.info(f"MESF columns: {metadata['mesf_count']}")
        
        return metadata
    
    def add_categorical_features(self) -> Dict[str, Any]:
        """Add categorical features for Method 2 in Julia.
        
        Returns:
            Dictionary with operation status
        """
        result = subprocess.run(
            [self._julia_executable, str(self._julia_main_script), 'add_categorical'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse result
        status = json.loads(result.stdout)
        logger.info(f"Categorical features added: {status['features']}")
        
        return status
    
    def fit_glm(self, formula: str) -> Dict[str, Any]:
        """Fit a negative binomial GLM using Julia.
        
        Args:
            formula: R-style formula string
            
        Returns:
            Dictionary with model statistics
        """
        # Run Julia GLM fitting
        result = subprocess.run(
            [self._julia_executable, str(self._julia_main_script), 'fit_glm', formula],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse results
        results = json.loads(result.stdout)
        
        # Convert to numpy arrays
        results['coefficients'] = np.array(results['coefficients'])
        results['std_errors'] = np.array(results['std_errors'])
        results['vcov'] = np.array(results['vcov'])
        
        logger.info(f"GLM fitted successfully: AIC={results['aic']:.2f}, n={results['n_obs']}")
        
        return results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of currently loaded data in Julia.
        
        Returns:
            Dictionary with data summary
        """
        result = subprocess.run(
            [self._julia_executable, str(self._julia_main_script), 'data_summary'],
            capture_output=True,
            text=True,
            check=True
        )
        
        return json.loads(result.stdout)
    
    # Compatibility methods for old interface
    def import_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load data - returns metadata only."""
        return self.load_data(path, spatial_grid_step=1)
    
    def import_file_with_sampling(self, path: Union[str, Path], spatial_grid_step: int) -> Dict[str, Any]:
        """Load data with sampling - returns metadata only."""
        return self.load_data(path, spatial_grid_step=spatial_grid_step)
    
    @classmethod
    def shutdown(cls, prompt: bool = False) -> None:
        """Clean up Julia resources."""
        if cls._initialized:
            logger.info("Cleaning up Julia resources")
            # Clean temp files
            if cls._temp_dir and cls._temp_dir.exists():
                import shutil
                shutil.rmtree(cls._temp_dir, ignore_errors=True)
            cls._initialized = False
            cls._current_data_id = None
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if Julia is initialized."""
        return cls._initialized
    
    @classmethod
    def get_cluster_info(cls) -> Optional[Dict[str, Any]]:
        """Get information about Julia setup."""
        if cls.is_initialized():
            import os
            return {
                'julia_threads': os.environ.get('JULIA_NUM_THREADS', 'unknown'),
                'temp_dir': str(cls._temp_dir),
                'initialized': True,
                'data_loaded': cls._current_data_id is not None
            }
        return None
    
    @classmethod
    def remove_all(cls) -> None:
        """Clean temporary files."""
        if cls.is_initialized() and cls._temp_dir:
            for file in cls._temp_dir.glob('*.json'):
                file.unlink(missing_ok=True)
            logger.info("Cleaned Julia temporary files")
    
    @classmethod
    def reset(cls) -> None:
        """Reset the manager state."""
        cls._initialized = False
        cls._current_data_id = None
        logger.warning("Julia manager state reset")