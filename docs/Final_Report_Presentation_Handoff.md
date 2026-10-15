# Final Report and Presentation Handoff

## Purpose

Use this context to advise on a concise final-project presentation. The project concerns urban heat and cooling prioritisation in Singapore. Keep recommendations short, clear, and focused on actual work, evidence, and planning relevance. Avoid em dashes.

This is a summary of the report-planning conversation, not an independently verified final results record. Numerical findings listed below should be checked against the latest evaluation outputs before use in slides.

## Required Presentation and Report Coverage

| Requirement | Report coverage |
|---|---|
| Tools and techniques used | 6.2 Tools and Technologies; 4.2 Data Cleaning and Processing; 5.5 Sensor Fusion Approach |
| System design and models | Section 5: architecture, pipeline, model inputs, training targets, sensor fusion, and scoring |
| System performance | 7.1 Evaluation Overview and Framework; 7.2 Results, Findings and Discussion |
| Findings and discussions | 7.2 Results, Findings and Discussion; 7.3 Evaluation Against Project Objectives; 7.4 Changes from the Initial Design |

The structure covers all four requirements. Section 7 must contain actual results and explain what worked, what did not, and what this means for cooling-priority decisions.

## Report Ownership and Structure

The user's partner covers Sections 1 and 2. The user covers Project Scope onwards. The abstract can be completed jointly after the main report.

Front matter: Abstract, Acknowledgement, Table of Contents, List of Figures, List of Tables.

1. Introduction
   - 1.1 Overview
   - 1.2 Problem Statement
   - 1.3 Project Goal and Objectives
2. Project Background
   - 2.1 Background
   - 2.2 Related Work
   - 2.3 Research Gap
   - 2.4 Contributions and Novelty
3. Project Scope
   - 3.1 Study Area and Scope Boundaries
   - 3.2 Key Features and Deliverables
4. Data Collection and Preparation
   - 4.1 Data Sources
   - 4.2 Data Cleaning and Processing
   - 4.3 Dataset Challenges
5. System Design
   - 5.1 System Architecture and Infrastructure Diagram
   - 5.2 Overall Workflow
   - 5.3 Data Processing Pipeline
   - 5.4 Model Workflow and Training Targets
   - 5.5 Sensor Fusion Approach
   - 5.6 Cooling Priority Scoring and Ranking
6. Implementation
   - 6.1 System Implementation Overview
   - 6.2 Tools and Technologies
   - 6.3 Core Components and Sensor Fusion Implementation
   - 6.4 System Demonstration
7. Results and Discussion
   - 7.1 Evaluation Overview and Framework
   - 7.2 Results, Findings and Discussion
   - 7.3 Evaluation Against Project Objectives
   - 7.4 Changes from the Initial Design
8. Challenges and Roadblocks
   - 8.1 Main Challenges and Solutions
   - 8.2 Remaining Limitations
9. Future Work
10. Conclusion
11. AI Tool Declaration
12. References

## Project Goal and Deliverables

Rank Singapore's URA subzones by cooling priority, quantify confidence, and assess potential cooling interventions.

1. **Cooling-priority score:** Combines heat exposure, population sensitivity, and greenery deficit using data-derived weights and uncertainty bands. Supports prioritisation, including consideration of cooling-centre siting and elderly outreach.
2. **Interactive application:** Displays rankings and supports what-if greening scenarios to guide where greening and shading resources could be focused. It does not provide detailed budget estimates or verified intervention outcomes.
3. **Evaluation framework:** Addresses the absence of direct ground truth for cooling priority through external heat comparisons, ranking stability under noise, and assessment of methodological choices.

Target users include urban planners and heat-response agencies. Physical interventions and verification of real-world cooling effects are outside scope.

## Data and Preparation

Primary datasets are publicly accessible. GEE supplies satellite and land-cover products; Singapore-specific data comes through data.gov.sg. Approximately 300 manually labelled reference points were created using a custom validation tool and high-resolution imagery.

| Source | Role |
|---|---|
| Landsat | Surface temperature, heat exposure, and heat-model targets |
| Sentinel-2 | Six spectral bands and three indices: NDVI, NDBI, NDWI |
| Dynamic World | Final training-label source for land-cover classification |
| ESA WorldCover | Initial label source and comparison baseline |
| URA | Subzone boundaries |
| SingStat | Population counts and age profiles |
| MODIS | External surface-temperature reference and heat-noise estimation |
| NEA | Air-temperature cross-check |

Preparation includes Landsat QA masking, Sentinel-2 s2cloudless masking, median compositing, spatial alignment, population cleaning, and standardisation of boundary fields. Satellite seasonal windows are April-May and October-November within 2021-2026, capped at 30 June 2026. Land-cover labels are grouped into Vegetation, Built-up, Bare land, and Water. Dynamic World training labels use their configured full-period window, not necessarily the satellite seasonal restriction.

A recorded land-change comparison used Dynamic World for October 2020-March 2022 versus January 2025-June 2026. It is an approximate 2021 versus 2025-2026 comparison, not an exact annual comparison. Recorded class differences were 12.5% of pixels versus a 7.0% baseline noise estimate; these include classification variability and cannot all be called physical land change.

## System Design and Tools

- **GEE:** Satellite preparation, geospatial processing, and Random Forest training.
- **Google Colab:** Deep-learning training.
- **Google Cloud Storage:** Transfer and storage of training data and model artefacts.
- **Local Python pipeline:** Indicator generation, scoring, evaluation, and application preparation.
- **MLflow:** Training and evaluation records.
- **Streamlit:** Locally run interactive application and label-validation interface.

Distinguish the diagrams:

- Architecture: where components run and how they connect.
- Planning workflow: rank priorities, check confidence, explore heat typologies and greening scenarios.
- Data pipeline: how raw inputs become maps, indicators, rankings, and application outputs.
- Model workflow: each model's inputs, targets, outputs, and downstream dependencies.

## Models and Their Roles

| Model | Inputs and target | Role |
|---|---|---|
| U-Net land-cover classifier | Sentinel-2 features; Dynamic World four-class targets | Selected production land-cover map |
| Random Forest and RF/U-Net hybrid | Land-cover classification alternatives | Model comparisons retained after selecting U-Net |
| CNN heat model | Nine spectral/index channels plus four land-cover channels; resampled Landsat temperature target | Pixel-level heat estimates and local greening scenarios |
| XGBoost | Subzone land-cover fractions, seasonal indices, and demographics; baseline subzone Landsat temperature target | Subzone heat estimates and greening scenarios |
| Regression downscaling | Sentinel-2 indices fitted against Landsat temperature on a 30 m grid | Estimated heat on a 10 m grid with interpolated residual adjustment |
| K-means, with GMM comparison | Environmental and seasonal indicators; no supervised target | Descriptive heat typologies |

Heat typology labels are excluded from XGBoost inputs to avoid indirect target leakage. CNN and XGBoost scenario outputs are compared, not averaged into a single model.

## Section 5.5: Sensor Fusion Approach

The system combines Landsat thermal observations with Sentinel-2 spectral information for heat estimation. Land-cover and demographic data are then integrated at subzone level to support cooling prioritisation and greening scenarios.

| Stage | Inputs | Integration approach | Output |
|---|---|---|---|
| Spatial and temporal alignment | Landsat and Sentinel-2 imagery | Apply cloud masks, use matching seasonal windows, and align grids. Aggregate Sentinel-2 indices to the 30 m Landsat analysis grid for regression training. | Compatible thermal and spectral inputs |
| Thermal and spectral fusion | Landsat temperature; Sentinel-2 NDVI, NDBI, NDWI | Fit regression at 30 m, apply at 10 m, and add interpolated coarse-grid residuals. | Estimated surface temperature on a 10 m grid |
| Land-cover and heat-model integration | Sentinel-2 features, Dynamic World labels, Landsat targets | Train U-Net; combine nine spectral/index channels with four land-cover channels as CNN inputs. | Four-class land-cover map and CNN heat estimates |
| Subzone integration | Baseline Landsat heat, land-cover fractions, spectral indices, demographics | Aggregate by URA subzone. Combine heat, sensitivity, and greenery deficit for scoring; use XGBoost separately for scenarios. | Rankings and indicative scenario estimates |

A 10 m output grid does not establish independently observed 10 m thermal detail. Landsat is not the only freely available surface-temperature source.

MODIS and NEA are validation references, not direct inputs to generating the fused heat map. Keep their detailed discussion in evaluation. Keep technical challenges in Section 8 rather than expanding Section 5.5.

The effort lies in aligning datasets, constructing compatible inputs, validating land-cover labels, and connecting environmental outputs to planning units. Avoid claiming a new fusion algorithm or proven planning improvement without evidence.

## Scoring and Confidence

- Exposure: mean baseline Landsat surface temperature per subzone.
- Sensitivity: normalised population density and elderly proportion, with an equal internal split as a modelling assumption.
- Greenery deficit: one minus the vegetation fraction.
- Main score: normalised indicators combined using PCA-derived weights from standardised indicators; equal weighting is a comparison.
- Resident-focused ranking: excludes subzones with fewer than 500 residents.
- Confidence: input perturbations produce score bands and top-priority membership frequencies. These are conditional on assumed error models, not empirically calibrated coverage guarantees.
- MODIS residual spread informs heat noise; greenery disagreement provides an error proxy. Methodological choices are tested separately.

## Changes from the Initial Design

| Initial plan | Final approach / recorded outcome | Reason or interpretation |
|---|---|---|
| Calibrated confidence bands | Error-informed bands; empirical coverage not tested | No independent cooling-priority reference for testing coverage |
| NEA as the main heat reference | MODIS as main reference and heat-noise source; NEA for context | Recorded coverage: MODIS 329 subzones, NEA 12; air and surface temperature differ |
| RMSE against external references for downscaling | Offset, spread, rank correlation, and coarsen-and-recover skill test | Constant offsets do not affect rankings; coarse references cannot establish fine-scale accuracy |
| Convergence with Cooling Singapore | Supporting comparison with published Singapore-ETH Centre hotspot areas | Different geographical units and inputs |
| WorldCover training labels | Dynamic World labels | Better agreement with manually labelled points |
| RF/U-Net hybrid as final classifier | U-Net selected; RF and hybrid retained for comparison | U-Net performed best in the project evaluation; blending did not improve performance |
| Additional work beyond proposal | Noise-floor comparison, minimum-residents threshold, all-places view, typology stability tests | Better interpretation of decision impact and robustness |

## Finding Requiring Careful Explanation

Recorded XGBoost/CNN comparison: predicted direction of change agreed in 17 of 20 scenarios, but predicted magnitudes differed.

- Direction means whether added greenery predicts cooling or warming.
- Magnitude means the amount of temperature change.
- Agreement in direction is not agreement on cooling amount.
- Disagreement alone does not prove CNN exaggeration; any stronger claim requires separate evaluation evidence.
- Present this primarily as a result in Section 7.2. It is a design deviation only if it changed model use or presentation.

## Existing Illustration Files

- `docs/figures/fusion_diagram.html`: preview of the latest fusion figure.
- `docs/figures/sensor_fusion_report.png`: report image.
- `docs/figures/sensor_fusion_report.svg`: scalable image.
- `docs/figures/draw_sensor_fusion_report.py`: source renderer.

The current illustration contains four integration stages and excludes external validation and uncertainty content.

## Advice Requested from the Next Chat

Help turn this material into a short final presentation covering the four required categories. Recommend slide order, which figures or demonstrations to use, what results need evidence, and concise speaker notes. Preserve the distinction between implemented methods, observed results, and unverified claims. Presentation duration and slide limit have not yet been specified.
