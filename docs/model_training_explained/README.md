# How each model in this project is trained

Plain-English walkthrough of every model in the pipeline: what data it sees,
how that data is filtered/cleaned before training, and — for the two neural
networks — why the architecture and hyperparameters are what they are. This
folder is pure explanation; it doesn't change any code.

## The five models

| # | Model | Type | What it predicts | File |
|---|---|---|---|---|
| 1 | Random Forest | Supervised ML | Land cover (4 classes) per pixel | [02_random_forest_landcover.md](02_random_forest_landcover.md) |
| 2 | U-Net | Supervised Deep Learning | Land cover (4 classes) per pixel | [03_unet_landcover_classifier.md](03_unet_landcover_classifier.md) |
| 3 | CNN heat regressor | Supervised Deep Learning | Land surface temperature (°C) per pixel | [04_cnn_heat_model.md](04_cnn_heat_model.md) |
| 4 | XGBoost | Supervised ML | Land surface temperature (°C) per subzone | [05_xgboost_heat_model.md](05_xgboost_heat_model.md) |
| 5 | K-means / GMM | Unsupervised ML | Hotspot-typology cluster per subzone | [07_hotspot_clustering_kmeans_gmm.md](07_hotspot_clustering_kmeans_gmm.md) |

Models 1 and 2 are also combined post-training into a **hybrid** (not an
"ensemble" — see [06_hybrid_rf_unet_combination.md](06_hybrid_rf_unet_combination.md)
for why that distinction matters here). Models 1, 2, and 3 all share the
same upstream data-preparation pipeline — read
[01_data_preparation_and_filtering.md](01_data_preparation_and_filtering.md)
first, since every other file assumes it.

## Reading order

If you want the full picture start to finish:

1. **01** — the shared data pipeline (season window, cloud filtering, NA handling, non-circularity)
2. **02, 03** — the two land-cover classifiers (same inputs, different model types)
3. **04** — the CNN heat model (reuses U-Net's architecture almost verbatim — read 03 first)
4. **05** — XGBoost (a completely separate, tabular half of the heat model)
5. **06** — how RF + U-Net get combined after training
6. **07** — the one unsupervised model in the project

## The one-sentence version of each model's "why"

- **RF**: fast, cheap, strong baseline; trains server-side on Earth Engine in minutes with no GPU.
- **U-Net**: learns spatial context (a pixel's neighbors, not just its own spectral value) that RF structurally can't see.
- **Hybrid (RF+U-Net)**: RF and U-Net make different mistakes on different pixels; averaging their probabilities is cheaper than picking a "winner" and safer than trusting either alone.
- **CNN heat regressor**: same architecture as U-Net, swapped from classifying land cover to predicting a continuous temperature value — because temperature has real spatial structure too (a hot pixel's neighbors are usually also hot).
- **XGBoost**: a second, independent way to estimate the same thing (subzone-level LST) from tabular features, used as a cross-check against the CNN rather than a replacement for it.
- **K-means/GMM**: finds heat "typologies" (e.g. "hot + dry + built-up" vs "cool + wet + green") without being told the answer in advance — genuinely exploratory, unlike the four supervised models above it.
