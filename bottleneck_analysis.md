# Bottleneck Analysis for 180M Point Processing

## Current Bottlenecks (Remaining Issues)

### 1. ❌ **Bootstrap in Method 3 (DML)**
```python
# Still creates pandas DataFrame copies for each bootstrap iteration!
for b in range(n_bootstrap):
    boot_idx = np.random.choice(n, n, replace=True)
    self.data_processed = original_data.iloc[boot_idx]  # DataFrame copy!
```
**Problem**: Even with joblib parallelization, each bootstrap creates a full copy
**For 180M points**: 100 bootstraps × 30GB = 3TB memory operations!
**Solution Needed**: Sample-based bootstrap or weighted bootstrap

### 2. ❌ **Complete Case Analysis**
```python
data_clean = data.dropna(subset=[...])  # Still in pandas!
```
**Problem**: For 180M points, this is a full pandas scan
**Solution**: Should be done in H2O or during parquet loading

### 3. ❌ **Standardization Still in Pandas** (Methods 1, 2, 3)
```python
for var in climate_vars:
    mean_val = data[var].mean()
    sd_val = data[var].std()
    data[f'{var}_std'] = (data[var] - mean_val) / (2 * sd_val)
```
**Problem**: Single-threaded pandas operations on 180M rows
**Solution**: Should be in H2O (256-thread parallel)

### 4. ⚠️ **GLM Still Gets Pandas Data** (Methods 1, 2)
```python
# In method1_continuous.py
X_null = data[feature_cols_null]  # Pandas DataFrame
y = data['fungi_richness']        # Pandas Series

self.model_null = GLMClass(self.config.config)
self.model_null.fit(X_null, y)    # Converts pd→H2O inside!
```
**Problem**: Still doing pd→H2O conversion internally
**Solution**: Pass H2OFrame directly

## Redundant Pandas Logic Still Present

### 1. **Method 1 & 2: Full Pandas Pipeline**
- `prepare_data()` returns pandas DataFrame
- All feature engineering in pandas
- GLM converts to H2O internally (DUPLICATE MEMORY!)

### 2. **Method 3: Mixed Pipeline**
- Data prep in pandas
- LightGBM uses pandas directly (OK)
- But bootstrap creates many copies

### 3. **H2O GLM Wrapper**
- Still has legacy pandas support
- Warning message but still converts
- Should force H2OFrame-only for large data

## What's Actually Fixed ✅

1. **MESF**: Now cached in parquet (one-time computation)
2. **H2O Direct Loading**: Method1_continuous_h2o.py does it right
3. **Parallel GLM/GBM**: Using all 256 cores
4. **Memory-efficient MESF**: Grid approximation for 180M

## What Still Needs Fixing ❌

### Priority 1: Remove ALL pd→H2O conversions
```python
# BAD (current):
df = pd.read_parquet("data.parquet")
df = prepare_data(df)  # pandas operations
model.fit(df)  # converts to H2O inside!

# GOOD (needed):
h2o_df = h2o.import_file("data.parquet")
h2o_df = prepare_data_h2o(h2o_df)  # H2O operations
model.fit(h2o_df)  # no conversion!
```

### Priority 2: Fix Bootstrap Memory Issue
```python
# Current: Full data copies
for b in range(100):
    boot_data = data.iloc[boot_idx]  # 30GB copy!

# Better: Weight-based bootstrap
for b in range(100):
    weights = np.random.multinomial(n, [1/n]*n)  # Just weights!
    # Pass weights to model, no data copy
```

### Priority 3: Move ALL Operations to H2O
- Dropna → H2O
- Standardization → H2O  
- Log transforms → H2O
- Interaction terms → H2O

## The Real Pipeline Needed for 180M

```python
# 1. One-time MESF
if 'ME1' not in parquet_columns:
    compute_and_add_mesf(parquet_path)  # Grid-based for 180M

# 2. Direct H2O loading
h2o_df = h2o.import_file(parquet_path)  # 30GB

# 3. ALL processing in H2O
h2o_df = h2o_df.dropna()  # Parallel!
h2o_df['bio1_std'] = (h2o_df['bio1'] - h2o_df['bio1'].mean()) / (2 * h2o_df['bio1'].sd())
h2o_df['plants_log'] = h2o_df['plants'].log1p()

# 4. Direct model fitting (no conversion)
model.fit(h2o_df, 'fungi_richness')

# Memory used: 30GB total (not 60GB!)
```

## Summary

**Current State**: We have H2O wrappers but still doing pandas→H2O conversions everywhere!

**Bottlenecks for 180M**:
1. **Memory doubling** from pd→H2O conversion (30GB→60GB)
2. **Bootstrap** creating 100 copies (3TB memory ops)
3. **Single-threaded** pandas operations (dropna, standardization)

**What to do**:
1. Use `method1_continuous_h2o.py` as template - it does it right!
2. Update Methods 1, 2, 3 to work like the H2O version
3. Implement weighted bootstrap to avoid copies
4. Remove ALL internal pd→H2O conversions