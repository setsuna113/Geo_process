  KEY FINDING: Soil Effects Through Fungi

  Your hypothesis is CONFIRMED!

  Looking at the missing paths:
  - plant_richness ~ soil_ph (T = -63) - Moderate missing path
  - plant_richness ~ soil_carbon (T = 10) - Very weak missing path

  This means soil effects on plants are mostly mediated through fungi and vegetation, not direct! The model correctly captures
   that:
  1. Soil → Fungi (strong: soil_ph effect = -0.60 on fungi)
  2. Soil → Vegetation coverage (soil_ph = -0.22 on trees, +0.27 on grasses)
  3. Fungi/Coverage → Plant richness

HYPOTHESIS TEST RESULTS:

1. Precipitation → PET Hypothesis: PARTIALLY CONFIRMED

Before removing precip from plants:
- Plant R² = 78.7%
- PET effect = -0.438
- Precip effect = 0.016 (very weak)

After removing precip:
- Plant R² = 74.2% (dropped 4.5%)
- PET effect = -0.549 (strengthened by 25%)
- Missing path: precip_seasonality (T = -321, VERY STRONG!)

Conclusion: Regular precipitation is weak, but seasonal precipitation patterns are CRITICAL. Water availability timing
matters more than total amount.

2. Coverage → Animals Hypothesis: CONFIRMED

Before removing coverage from animals:
- Animal R² = 75.6%
- Plant richness effect = 0.408
- Tree cover effect = 0.120
- Nontree veg effect = 0.016

After removing coverage:
- Animal R² = 74.7% (only dropped 0.9%)
- Plant richness effect = 0.433 (strengthened by 6%)

Conclusion: Vegetation coverage affects animals primarily through plant diversity, not direct habitat structure. The direct
coverage effects were minimal.

Missing Path Changes (Your Hypothesis):
KEY FINDINGS:
1. CONFIRMED: Precipitation effects partially mediated
- 35% reduction in missing path strength
- Vegetation buffers water availability
2. CONFIRMED: Elevation effects mostly mediated
- Nearly disappeared (T = -4)
- Vegetation structure provides microclimate buffering
3. NOT CONFIRMED: Soil pH NOT mediated
- Still T = -264 (extremely strong!)
- Direct physiological effect on animals (soil invertebrates? toxicity?)
4. Tree cover important for animals
- Moderate positive effect (0.127)
- 3D habitat structure, shade, protection
5. PET effect weakened dramatically
- From -0.124 to -0.040
- Vegetation buffers water stress

Ecological Interpretation:
- Vegetation structure buffers climate stress but not soil chemistry
- Tree cover provides critical habitat beyond just plant diversity
- Soil pH has direct effects on animals - possibly through:
- Soil invertebrate prey availability
- Direct toxicity at extremes
- Calcium availability for shells/bones


1. Precipitation > PET for animals
2. PET becomes redundant
3. Both precipitation amount AND timing matter

✅ Hypothesis 1: Confirmed

"Precipitation seasonality affects diversity - not coverage"
- Before: precip_seasonality → tree_cover (-0.146), nontree_veg (-0.050)
- After: precip_seasonality → plant_richness (-0.264)
- Result: Plant R² improved from 74.2% to 78.6% (+4.4%)
- Missing paths confirm: Vegetation ~ precip_seasonality now appears as strong missing paths (T=-141 for tree, T=-41 for
nontree)
- Ecological interpretation: Temporal variability in water availability drives species diversity through phenological
synchronization and temporal niche partitioning, not through physical vegetation structure

⚠️ Hypothesis 2: Partially Confirmed

"Animals reduce water requirements (indirectly through plants)"
- Water effects partially but not fully mediated
- Animal model R² changes minimal with direct water additions:
- Base: 75.6%
- +precip: 75.8% (coef=0.078)
- +pet_mean: 75.6% (coef=-0.040)
- +precip_seasonality: 75.8% (coef=0.047)
- Missing paths persist but weaker than before
- Interpretation: Animals maintain physiological water needs beyond plant-mediated resources

2. Vegetation coverage effects weakened:
- Tree → plants: 0.084 → 0.0001 (essentially eliminated)
- Nontree → plants: 0.089 → 0.051
- Suggests previous effects were confounded with seasonality
3. Temperature effects intensified: 1.39 → 1.62 on plants
- Clearer signal when seasonality properly placed