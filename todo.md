Multilayer Network:

  # Multiple networks of relationships
  Layer1: Environmental responses (who responds similarly?)
  Layer2: Information flow (who predicts whom?)
  Layer3: Spatial patterns (who clusters together?)
  Layer4: Trophic interactions (who eats whom?)

  # Reveals:
  # - Plants→Fungi strong in Layer2 (information)
  # - Plants←→Fungi weak in Layer1 (different environmental responses)
  # - Vertebrates central in Layer4 (trophic hub)
  # - Contradictions between layers show complex dynamics

  Real Example:

  REGRESSION says:
  "Plants explain 40% of fungi variance"

  MULTILAYER says:
  "Plants and fungi are:
   - Informationally coupled (Layer 2: 0.7 mutual information)
   - Environmentally decoupled (Layer 1: 0.2 correlation)
   - Spatially independent (Layer 3: 0.1 correlation)
   - Functionally interdependent (Layer 4: mutualism network)"

  Why This Matters for Your Question:

  You want to understand functional relationships. These exist in MULTIPLE forms:
  1. Mutualistic (mycorrhizae: plants ↔ fungi)
  2. Trophic (herbivory: plants → vertebrates)
  3. Indirect (vertebrates disperse fungi)
  4. Environmental filtering (all like moisture)


Multilayer networks are a FRAMEWORK that can use many methods, including mutual information and partial correlation. Let me show you the hierarchy:"

The Method Hierarchy:

MULTILAYER NETWORK FRAMEWORK
    ├── Layer Construction Methods (choose per layer):
    │   ├── Mutual Information (nonlinear dependencies)
    │   ├── Partial Correlation (linear, controlled)
    │   ├── Transfer Entropy (directional information)
    │   ├── Convergent Cross Mapping (causal dynamics)
    │   ├── Distance Correlation (nonlinear correlation)
    │
    └── Cross-Layer Analysis Methods:
        ├── Layer Correlation (how similar are layers?)
        ├── Multiplex Centrality (importance across layers)
        ├── Cross-layer Paths (information routes)
        └── Layer-specific Roles (hub in one, peripheral in another)


- Plants create habitats for vertebrates
- Vertebrates disperse plant seeds
- Fungi form mycorrhizal networks with plants
- All respond to AND modify their environment


  Option 1: Hierarchical Model (Most Defensible)

  library(piecewiseSEM)

  # Based on ecological theory: fungi → plants → animals
  model1 <- psem(
    lm(fungi_richness ~ temp + precip + soil_pH + elevation),
    lm(plant_richness ~ temp + precip + soil_pH + elevation + fungi_richness),
    lm(animal_richness ~ temp + precip + elevation + plant_richness)
  )

  # Why this structure?
  # - Fungi establish first (soil decomposers/mycorrhizae)
  # - Plants depend on soil nutrients (affected by fungi)
  # - Animals depend on plant resources


  # - Shannon (needs abundance)
  # - Simpson (needs abundance)
  # - FRic/FEve/FDiv (needs traits)
  # - Rao's Q (needs traits or phylogeny)

  Option 3: What You Actually Should Do First

  # Test for multicollinearity between kingdoms
  cor.test(plant_richness, animal_richness)
  cor.test(plant_richness, fungi_richness)
  cor.test(animal_richness, fungi_richness)

  # If correlations > 0.7, you have a problem
  # SEM assumes your predictors aren't perfectly correlated

  The Reality Check

  What SEM Can Tell You:
  - Direct effects: "After controlling for environment, fungi richness increases plant richness by β = 0.3"
  - Indirect effects: "Temperature affects animals through plants (β = 0.2)"
  - Model fit: "This causal structure explains 60% of variance"

  What SEM CANNOT Tell You:
  - True causality (it's still correlation, just structured)
  - Feedback loops (plants help fungi AND fungi help plants)
  - Mechanisms (WHY fungi increase plant richness)


  # The Only Honest Model
  library(piecewiseSEM)

  # Step 1: Test environmental baseline
  baseline <- psem(
    lm(fungi_richness ~ env_vars),
    lm(plant_richness ~ env_vars),
    lm(animal_richness ~ env_vars)
  )

  # Step 2: Add ONE direction of biological influence
  # Pick based on ecological theory, not data
  biology_model <- psem(
    lm(fungi_richness ~ env_vars),
    lm(plant_richness ~ env_vars + fungi_richness),
    lm(animal_richness ~ env_vars + plant_richness)
  )

  # Step 3: Compare
  anova(baseline, biology_model)

  # If biology_model isn't significantly better, stop here
  # Your "interactions" are just shared environmental response


Variance Partitioning:
1. Variance Partitioning (Legendre 2008, Journal of Plant Ecology):
- "a method of choice for the interpretation of beta diversity"
- Makes NO causal claims
Variance Partitioning explicitly makes no causal claims (Schulz 2025) - just describes variance
# Step 1: Remove environmental signal
  plant_resid <- residuals(lm(plant_div ~ environment))
  animal_resid <- residuals(lm(animal_div ~ environment))
  fungi_resid <- residuals(lm(fungi_div ~ environment))

  # Step 2: Measure coupling
  coupling_strength <- cor(cbind(plant_resid, animal_resid, fungi_resid))

  But Here's The Critical Problem - Direct Quote:

  From Biometrics (2021):
  "The naive estimator is generally overly biased, in the sense that its bias does not tend to zero sufficiently fast to allow consistency at rate n^(-1/2)... This is 
  problematic, in particular, because it renders the construction of valid confidence intervals difficult, if not impossible"

  Why DML's Orthogonalization Exists - The Real Answer:

  From the same paper:
  "The naive ANOVA-based estimator is generally not asymptotically linear when flexible (eg, machine learning based) estimators of the involved regression are used. It will 
  usually be overly biased, resulting in a rate of convergence slower than n^(-1/2)"

  THIS is why DML orthogonalizes! Not to make it "causal" but to fix the bias problem when using ML.

  The Specific Issues With ML Variance Partitioning:

  1. Bias Toward Correlated Variables (BMC Bioinformatics):
  "Variable importance measures show a bias towards correlated predictor variables"
  2. Use of "Impossible Data" (Annual Reviews):
  "The most popular methods... make use of synthetic inputs that combine predictor variables from multiple observations. These inputs can be unlikely, physically impossible, 
  or even logically impossible"
  3. Model-Dependent Results (PMC 2021):
  "important variables for one well-performing model... may be unimportant for another model"

  1. Direct Ecological Applications:

  From bioRxiv (2019) on spotted seatrout recruitment:
  "XGBoost, a machine learning method for regression and classification, was employed because it inherently models variable interactions and seamlessly deals with 
  multi-collinearity, both of which are common features of ecological datasets"

  The paper states:
  "The use of gradient boosting in the fisheries and marine ecology literature has increased... 142 articles (2008-2012)... 316 results (2013-2018)"

  2. Conditional Permutation Importance - Real R Package:

  CRAN Package permimp (actively maintained):
  "The permimp package... Conditional Permutation Importance for random forests... The CPI is available for random forests grown by the randomForest package"

  From BMC Bioinformatics (2020):
  "When features are strongly correlated (common in ecological data), standard permutation importance can be misleading as it exchanges features with values never observed in
   reality"

  3. Model Class Reliance - Published Method:

  From JMLR (2020) and PMC (2021):
  "MCR is the range of variable importance values across all well-performing models... Important variables for one well-performing model may be unimportant for another"

  NHS England actually uses this:
  "MCR can be used to establish the value of using a variable in a model being created for healthcare predictions" (same principles apply to ecology)

  4. Actual Ecological Usage:

  From meta-analysis in Oecologia:
  "The average amount of variance explained by the main factor... mean amount of variance (r²) explained being 2.51–5.42%" across ecological studies

  From biodiversity research (PMC 2021):
  "Analysis of trait variance partitioning across 52 tree species revealed for most traits a dominant influence of phylogeny"

  The Reality Check:

  What People Actually Do:

  1. They DO use ML for variance decomposition - XGBoost and Random Forest are common in ecology papers
  2. They DO report R² and variance explained - Even with ML methods
  3. They DON'T usually use sophisticated corrections - Most papers just report raw importance scores
  4. They're AWARE of the problems - Recent papers acknowledge:
  "Conditional importance measures the loss increase due to information unique to that feature, not encoded in other features"

  Real Tools Available:

  R Packages:
  - permimp - Conditional permutation importance
  - party - Conditional inference forests with corrected importance
  - gradientForest - Has conditional importance options
  - partR2 - Variance partitioning (works with any model)
