"""Heat-typology chapter: unsupervised clustering of subzones by seasonal
heat and surface character. Numbers come from:
  data/processed/hotspot_clusters.csv (scripts/build_hotspot_clusters.py)
  data/processed/diagnostics/typology_*.csv (scripts/evaluate_hotspot_typology.py)
"""

import pandas as pd

from helpers import pct

TITLE = "Heat Typology"
STANDALONE_TITLE = "Heat Typology Evaluation and Ablation Tests"
SUBTITLE = "K-means and Gaussian-mixture clustering of subzones by seasonal heat and surface character"


def _names(profile):
    """Descriptive names assigned by rule, so they stay right if cluster ids change:
    hottest inter-monsoon temperature, highest vegetation, largest seasonal swing, the rest."""
    p = profile.set_index("primary_cluster").copy()
    p["swing"] = p["lst_dry"] - p["lst_wet"]
    names = {p["lst_dry"].idxmax(): "Hot, densely built",
             p["fraction_vegetation"].idxmax(): "Cool and green",
             p["swing"].idxmax(): "Large seasonal swing"}
    for c in p.index:
        names.setdefault(c, "Moderate, mixed urban")
    return names


def build(doc, repo):
    D = repo / "data/processed/diagnostics"
    clusters = pd.read_csv(repo / "data/processed/hotspot_clusters.csv")
    q = pd.read_csv(D / "typology_quality.csv")
    sweep = pd.read_csv(D / "typology_k_sweep.csv")
    stab = pd.read_csv(D / "typology_stability.csv").set_index("test")
    abl = pd.read_csv(D / "typology_feature_ablation.csv")
    coh = pd.read_csv(D / "typology_coherence.csv").set_index("variable")
    prof = pd.read_csv(D / "typology_profile.csv")
    pri = pd.read_csv(D / "typology_priority.csv")
    certainty = float((D / "typology_gmm_certainty.txt").read_text())

    names = _names(prof)
    order = list(prof.sort_values("lst_dry", ascending=False)["primary_cluster"])
    prod, gmm = q.iloc[0], q.iloc[1]
    k, k_gmm = int(prod["k"]), int(gmm["k"])
    km = stab.loc["K-means, other seeds"]
    bs = stab.loc["K-means, bootstrap resamples"]
    gs = stab.loc["GMM, other seeds"]
    km_vs_gmm = stab.loc["K-means vs GMM (production labels)", "mean"]
    p = prof.set_index("primary_cluster")
    hot, green = p["lst_dry"].idxmax(), p["fraction_vegetation"].idxmax()
    swing = (p["lst_dry"] - p["lst_wet"]).idxmax()
    swing_examples = ", ".join(
        clusters[clusters["primary_cluster"] == swing]
        .assign(s=lambda d: d["lst_dry"] - d["lst_wet"]).sort_values("s", ascending=False)["subzone_id"]
        .str.title().head(6))
    res = pri[pri["view"] == "Residents"].set_index("cluster")
    allp = pri[pri["view"] == "All places"].set_index("cluster")

    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    # --- Summary ---------------------------------------------------------------
    doc.section("Summary", numbered=False)
    doc.para(
        f"The heat typology groups the 332 subzones into {k} types by their surface temperature and greenness "
        "in two seasons. It describes subzones; it does not enter the priority score. With no ground truth for "
        "a typology, it is judged on whether its structure is real, repeatable and consistent with data it was "
        "not built from."
    )
    doc.para(
        f"The typology is clearly better than chance: its silhouette ({prod['silhouette']:.2f}) is more than "
        f"twice that of the same features shuffled ({prod['null_silhouette_mean']:.2f}). It is repeatable across "
        f"random seeds (agreement {km['mean']:.2f}) but only moderately stable when the subzones are resampled "
        f"({bs['mean']:.2f}). It agrees strongly with land cover, which was not a clustering input "
        f"(clusters explain {pct(coh.loc['fraction_vegetation', 'eta_squared'], 0)} of the variation in "
        f"vegetation share), and with independent MODIS temperature. The hottest type holds "
        f"{int(res.loc[hot, 'n_in_top20'])} of the residents' top-20 priority subzones; the green type holds none."
    )

    # --- Setup -------------------------------------------------------------------
    doc.section("Setup")
    doc.table(
        ["Item", "Setting"],
        [
            ["Subzones", f"{len(clusters)} URA subzones, every one with complete features."],
            ["Features", "Six subzone means, standardised: Landsat surface temperature, NDVI (greenness) and "
                         "NDBI (built-up index), each for the inter-monsoon months (April, May, October, "
                         "November) and for December to February, 2021 to June 2026."],
            ["Methods", "K-means and a Gaussian mixture model (GMM), each fitted for 2 to 8 clusters. Each "
                        "method's number of clusters is the one with the highest silhouette at 3 or more; the "
                        "method with the higher silhouette becomes the production typology."],
            ["Production typology", f"K-means with {k} clusters (GMM chose {k_gmm})."],
            ["Independent checks", "Land-cover fractions from the production land-cover map and MODIS surface "
                                   "temperature; neither is a clustering input."],
            ["Statistics", "Adjusted Rand index (ARI) for agreement between partitions (1 = identical, "
                           "0 = chance); Kruskal-Wallis tests and eta-squared (share of variance explained by "
                           "the clusters) for the independent checks."],
        ],
        [2300, 7600],
    )

    # --- Evaluation ----------------------------------------------------------------
    doc.section("Evaluation")
    doc.subsection("Internal quality against a null baseline")
    doc.para(
        "The null baseline shuffles each feature independently across subzones, keeping every feature's "
        f"distribution but removing any joint structure, and clusters it with the same method and k "
        f"({int(q.iloc[0]['k'])} clusters, 200 repeats)."
    )
    doc.table(
        ["Typology", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Cluster sizes"],
        [[f"K-means, {k} clusters (production)", f"{prod['silhouette']:.3f}", f"{prod['davies_bouldin']:.2f}",
          f"{prod['calinski_harabasz']:.0f}", f"{int(prod['min_size'])} to {int(prod['max_size'])}"],
         [f"GMM, {k_gmm} clusters", f"{gmm['silhouette']:.3f}", f"{gmm['davies_bouldin']:.2f}",
          f"{gmm['calinski_harabasz']:.0f}", f"{int(gmm['min_size'])} to {int(gmm['max_size'])}"],
         ["Shuffled features (null)", f"{prod['null_silhouette_mean']:.3f} (95th percentile "
          f"{prod['null_silhouette_p95']:.3f})", "", "", ""]],
        [3100, 2000, 1500, 1600, 1700],
    )
    doc.para(
        "Higher silhouette and Calinski-Harabasz, and lower Davies-Bouldin, mean tighter and better-separated "
        f"clusters. A silhouette of {prod['silhouette']:.2f} indicates real but overlapping groups, not sharply "
        "separated ones."
    )
    doc.subsection("Stability")
    doc.table(
        ["Test", "Repeats", "Mean agreement (ARI)", "5th–95th percentile"],
        [["K-means refitted with other random seeds", str(int(km["n"])), f"{km['mean']:.2f}",
          f"{km['p05']:.2f}–{km['p95']:.2f}"],
         ["K-means refitted on bootstrap resamples of subzones", str(int(bs["n"])), f"{bs['mean']:.2f}",
          f"{bs['p05']:.2f}–{bs['p95']:.2f}"],
         ["GMM refitted with other random seeds", str(int(gs["n"])), f"{gs['mean']:.2f}",
          f"{gs['p05']:.2f}–{gs['p95']:.2f}"]],
        [4600, 1100, 2000, 2200],
    )
    doc.para(
        "K-means returns nearly the same partition whatever the seed. Resampling the subzones changes it more: "
        "the main types persist, but subzones near the boundaries between types move, so individual "
        "assignments near a boundary should not be over-read. GMM is less repeatable than K-means."
    )
    doc.subsection("Agreement with independent data")
    cols = ["fraction_vegetation", "fraction_built_up", "fraction_water", "modis_lst"]
    labels = {"fraction_vegetation": "Vegetation share", "fraction_built_up": "Built-up share",
              "fraction_water": "Water share", "modis_lst": "MODIS temperature (°C)"}

    def cell(var, c):
        v = coh.loc[var, f"mean_cluster_{c}"]
        return f"{v:.1f}" if var == "modis_lst" else pct(v, 0)

    doc.table(
        ["Variable (not a clustering input)"] + [names[c] for c in order] + ["Variance explained"],
        [[labels[v]] + [cell(v, c) for c in order] + [pct(coh.loc[v, "eta_squared"], 0)] for v in cols],
        [2500, 1500, 1500, 1500, 1500, 1400],
    )
    least_veg = p["fraction_vegetation"].idxmin()
    doc.para(
        "Every variable differs significantly between types (Kruskal-Wallis p below 0.001). The hottest type is "
        f"also the most built-up and the hottest by MODIS, an independent sensor. It is not the least vegetated: "
        f"the {names[least_veg].lower()} type has slightly less vegetation "
        f"({pct(p.loc[least_veg, 'fraction_vegetation'], 1)} against {pct(p.loc[hot, 'fraction_vegetation'], 1)}), "
        "so the coherence check passes on built-up share and on independent temperature, and only narrowly fails "
        "on vegetation."
    )
    doc.subsection("The types")
    doc.table(
        ["Type", "Subzones", "Temperature, inter-monsoon (°C)", "Temperature, Dec–Feb (°C)", "NDVI", "Built-up share"],
        [[names[c], str(int(p.loc[c, "n_subzones"])), f"{p.loc[c, 'lst_dry']:.1f}", f"{p.loc[c, 'lst_wet']:.1f}",
          f"{p.loc[c, 'ndvi_dry']:.2f}", pct(p.loc[c, "fraction_built_up"], 0)] for c in order],
        [2400, 1100, 1800, 1700, 1200, 1700],
    )
    doc.para(
        f"The large-seasonal-swing type is the city centre ({swing_examples} and others): among the hottest in "
        f"the inter-monsoon months, but the coolest in December to February "
        f"({p.loc[swing, 'lst_dry'] - p.loc[swing, 'lst_wet']:.1f} °C apart, against about "
        f"{(p['lst_dry'] - p['lst_wet']).drop(swing).mean():.1f} °C for the other types). The cause was not "
        "tested."
    )

    # --- Ablations -------------------------------------------------------------------
    doc.section("Ablations and controlled comparisons")
    doc.subsection("K-means against a Gaussian mixture")
    doc.para(
        f"K-means ({k} clusters) has the higher silhouette ({prod['silhouette']:.2f} against "
        f"{gmm['silhouette']:.2f}), lower Davies-Bouldin and better seed stability, so it is the production "
        f"typology. The two partitions agree only moderately (ARI {km_vs_gmm:.2f}), partly because GMM chose "
        f"{k_gmm} clusters: the typology's exact boundaries depend on the method."
    )
    doc.subsection("Number of clusters")
    piv = sweep.pivot(index="k", columns="method", values="silhouette")
    doc.table(
        ["Clusters (k)", "K-means silhouette", "GMM silhouette"],
        [[str(kk) + (" (production)" if kk == k else ""), f"{piv.loc[kk, 'kmeans']:.3f}", f"{piv.loc[kk, 'gmm']:.3f}"]
         for kk in piv.index],
        [3300, 3300, 3300],
    )
    doc.para(
        "The number of clusters was chosen by this sweep on the same data, the standard procedure for "
        "unsupervised clustering, with at least 3 clusters so that a trivial two-way split could not win. "
        "There is no held-out test of the choice."
    )
    doc.subsection("Feature sets")
    doc.para(
        f"K-means with {k} clusters was refitted on subsets of the features and compared with the production "
        "typology. Silhouettes are not comparable across feature sets of different sizes; agreement shows how "
        "much each subset reproduces the production types."
    )
    doc.table(
        ["Features used", "Number", "Silhouette", "Agreement with production (ARI)"],
        [[r.feature_set, str(r.n_features), f"{r.silhouette:.3f}", f"{r.ari_vs_production:.2f}"] for r in abl.itertuples()],
        [3900, 1200, 1800, 3000],
    )
    a = abl.set_index("feature_set")["ari_vs_production"]
    doc.para(
        f"Neither temperature alone (ARI {a['Temperature only']:.2f}) nor the surface indices alone "
        f"({a['Indices only (no temperature)']:.2f}) reproduces the types: they combine heat with greenness and "
        f"built-up character. Dropping one index keeps most of the structure ({a['Without NDBI']:.2f} without "
        f"NDBI, {a['Without NDVI']:.2f} without NDVI), as does keeping only December to February "
        f"({a['December-February only']:.2f}); keeping only the inter-monsoon months changes the types much more "
        f"({a['Inter-monsoon only']:.2f})."
    )

    # --- Relation to the priority ranking ------------------------------------------------
    doc.section("Relation to the priority ranking")
    doc.para(
        "The typology does not enter the priority score, so no choice here can change the ranking and there is "
        "no decision-impact test. The table shows where each view's top-20 subzones fall."
    )
    doc.table(
        ["Type", "Residents view: ranked", "Residents view: in top 20", "All-places view: in top 20",
         "Mean priority score (residents)"],
        [[names[c], str(int(res.loc[c, "n_ranked"])), str(int(res.loc[c, "n_in_top20"])),
          str(int(allp.loc[c, "n_in_top20"])), f"{res.loc[c, 'mean_priority_score']:.2f}"] for c in order],
        [2500, 1700, 1800, 1800, 2100],
    )
    doc.para(
        f"The hottest type supplies most of the top-20 subzones in both views, and the green type none, which is "
        "expected because heat and missing greenery are two of the score's inputs. The typology adds a "
        "description of what kind of place each priority subzone is."
    )

    # --- Exploratory -------------------------------------------------------------------
    doc.section("Exploratory analyses")
    doc.lead("GMM membership certainty. ", (
        f"The GMM gives each subzone a probability of belonging to each type. {pct(certainty, 0)} of subzones "
        "belong to one type with probability 0.9 or more; the rest sit between types, consistent with the "
        "moderate bootstrap stability."))

    # --- Limitations ---------------------------------------------------------------------
    doc.section("Limitations")
    for lead, text in [
        ("No ground truth. ", "There is no reference typology, so the evaluation shows the types are real, "
                              "repeatable and physically consistent, not that they are the right types."),
        ("Daytime only. ", "Landsat passes in the late morning, so the typology says nothing about night-time "
                           "heat; a day-night typology was planned but not built."),
        ("Selection on the same data. ", "The number of clusters and the method were both chosen by silhouette on "
                                         "the data being clustered."),
        ("Boundary subzones. ", "Bootstrap stability is moderate, so subzones near a type boundary can change type."),
        ("Subzone averages. ", "Each subzone is one row of averages, which hides variation within large or mixed "
                               "subzones."),
    ]:
        doc.lead(lead, text)

    doc.section("Sources", numbered=False)
    doc.para(
        "Clustering: scripts/build_hotspot_clusters.py, src/hotspots/cluster.py, src/hotspots/features.py. "
        "Evaluation: scripts/evaluate_hotspot_typology.py, validation/hotspots_validation/typology_evaluation.py, "
        "validation/hotspots_validation/cluster_quality.py (outputs in data/processed/diagnostics/typology_*.csv)."
    )
