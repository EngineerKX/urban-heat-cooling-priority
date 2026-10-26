"""Downscaling skill test (coarsen-and-recover): does the TsHARP-style
regression downscaling (src/downscaling/variants.py::variant_regress10) add
real spatial detail, or only what interpolation would give?

The production layer goes 30 m -> 10 m, and there is no 10 m ground truth to
score it against. The standard workaround is to run the SAME method one step
coarser, where the truth exists: aggregate Landsat LST from 30 m to 90 m,
downscale it back to 30 m, and compare with the real 30 m Landsat LST. This
assumes the method's skill transfers from 90->30 m to 30->10 m (the same
factor-of-3 step); that assumption is a stated limitation, not tested here.

Methods compared (all 90 m -> 30 m):
  nearest            -- each 90 m value copied to its nine 30 m pixels
  bilinear, bicubic  -- interpolation only, no new information (bicubic is
                        the production `lst_bicubic10` method)
  tsharp             -- the production method: LST ~ NDVI + NDBI + NDWI fitted
                        at the coarse scale, applied at the fine scale, plus the
                        bilinear-resampled coarse residual
  tsharp_no_residual -- ablation: the regression prediction alone
  tsharp_ndvi_only   -- ablation: NDVI as the only predictor (classic TsHARP)

The regression is re-implemented here instead of calling variant_regress10,
whose reduceResolution(maxPixels=64) cannot aggregate 10 m to 90 m (81 fine
pixels per coarse pixel). The steps are otherwise the same.
"""

import ee
import numpy as np
import pandas as pd

INDEX_BANDS = ["NDVI", "NDBI", "NDWI"]


def _aggregate(image, crs, scale, max_pixels):
    """Mean-aggregate an image that already has a concrete finer projection."""
    return image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels).reproject(crs=crs, scale=scale)


def tsharp(lst_coarse, indices_coarse, indices_fine, bands, region, crs, coarse_scale, fine_scale,
           sample_n, seed):
    """TsHARP-style downscaling of `lst_coarse` onto the grid of `indices_fine`.
    The regression is fitted once (at the coarse scale) and its coefficients
    are fetched as plain numbers, so the returned images don't re-run the fit.
    Returns (with_residual, without_residual, coefficients dict)."""
    stack = lst_coarse.rename("LST").addBands(indices_coarse.select(bands)).addBands(ee.Image.constant(1).rename("CONST"))
    selectors = ["CONST"] + bands + ["LST"]
    samples = stack.select(selectors).sample(region=region, scale=coarse_scale, numPixels=sample_n, seed=seed,
                                             geometries=False, tileScale=16)
    fit = samples.reduceColumns(reducer=ee.Reducer.linearRegression(numX=len(bands) + 1, numY=1), selectors=selectors)
    terms = ee.Array(fit.get("coefficients")).project([0]).getInfo()

    def _predict(indices):
        out = ee.Image.constant(terms[0])
        for band, coef in zip(bands, terms[1:]):
            out = out.add(indices.select(band).multiply(coef))
        return out

    pred_fine = _predict(indices_fine)
    residual = lst_coarse.subtract(_predict(indices_coarse))
    with_residual = pred_fine.add(residual.resample("bilinear").reproject(crs=crs, scale=fine_scale))
    return with_residual, pred_fine, dict(zip(["intercept"] + bands, terms))


def build_prediction_stack(lst_30m, s2_indices_10m, region, crs, sample_n, seed,
                           fine_scale=30, coarse_scale=90):
    """Truth (Landsat LST at `fine_scale`: the 30 m product itself, or its
    mean-aggregate for a coarser fine scale) plus every method's
    reconstruction from the `coarse_scale` aggregate, as one multi-band
    image. Returns (image, coefficients).

    Why a coarser pair is offered: Landsat 8/9 thermal bands are measured at
    about 100 m and delivered resampled to 30 m, so a 30 m "truth" is itself
    partly interpolated, which favours interpolation methods. A pair above
    100 m (e.g. 360 -> 120 m) scores against detail the sensor really saw."""
    factor = coarse_scale // fine_scale
    truth = lst_30m if fine_scale == 30 else _aggregate(lst_30m, crs, fine_scale, max_pixels=(fine_scale // 30) ** 2 + 1)
    lst_coarse = _aggregate(lst_30m, crs, coarse_scale, max_pixels=(coarse_scale // 30) ** 2 + 1)
    # Fine-scale indices come straight from 10 m. Coarse indices come from 10 m
    # at the 30/90 pair (a 10 -> 30 -> 90 m chain exceeded Earth Engine's
    # interactive memory limit) and from the fine-scale indices otherwise
    # (10 m -> 360 m in one step would be ~1300 pixels per output pixel).
    indices_10m = s2_indices_10m.reproject(crs=crs, scale=10)
    indices_fine = _aggregate(indices_10m, crs, fine_scale, max_pixels=(fine_scale // 10) ** 2 + 1)
    if coarse_scale <= 90:
        indices_coarse = _aggregate(indices_10m, crs, coarse_scale, max_pixels=(coarse_scale // 10) ** 2 + 1)
    else:
        indices_coarse = _aggregate(indices_fine, crs, coarse_scale, max_pixels=factor * factor + 1)

    def _interp(mode):
        img = lst_coarse if mode == "nearest" else lst_coarse.resample(mode)
        return img.reproject(crs=crs, scale=fine_scale)

    common = dict(region=region, crs=crs, coarse_scale=coarse_scale, fine_scale=fine_scale,
                  sample_n=sample_n, seed=seed)
    full, no_res, coef_full = tsharp(lst_coarse, indices_coarse, indices_fine, INDEX_BANDS, **common)
    ndvi, _, coef_ndvi = tsharp(lst_coarse, indices_coarse, indices_fine, ["NDVI"], **common)

    stack = (
        truth.rename("truth")
        .addBands(_interp("nearest").rename("nearest"))
        .addBands(_interp("bilinear").rename("bilinear"))
        .addBands(_interp("bicubic").rename("bicubic"))
        .addBands(full.rename("tsharp"))
        .addBands(no_res.rename("tsharp_no_residual"))
        .addBands(ndvi.rename("tsharp_ndvi_only"))
    )
    return stack, {"tsharp": coef_full, "tsharp_ndvi_only": coef_ndvi}


METHODS = ["nearest", "bilinear", "bicubic", "tsharp", "tsharp_no_residual", "tsharp_ndvi_only"]


def sample_pixels(stack, region, scale, n_pixels, seed, batch_size=4000) -> pd.DataFrame:
    """Random 30 m pixels inside `region` with every band valid, plus x/y in
    the stack's metres (for the spatial block bootstrap). Fetched in batches
    with different seeds because a single getInfo() is capped at 5000 rows;
    a pixel drawn by two batches is kept once."""
    with_xy = stack.addBands(ee.Image.pixelCoordinates(stack.select("truth").projection()))
    frames = []
    for i in range((n_pixels + batch_size - 1) // batch_size):
        fc = with_xy.sample(region=region, scale=scale, numPixels=batch_size, seed=seed + i,
                            geometries=False, tileScale=16)
        frames.append(pd.DataFrame([r["properties"] for r in fc.getInfo()["features"]]))
        print(f"  batch {i + 1}: {len(frames[-1])} pixels")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    # pixelCoordinates on a scaled projection counts PIXELS of that projection; convert to metres.
    df[["x", "y"]] = df[["x", "y"]] * scale
    return df


def skill_table(df: pd.DataFrame, methods=METHODS, hot_q=0.9) -> pd.DataFrame:
    """Per method: bias, RMSE, MAE, R² against the real 30 m LST; the
    variance ratio (pred sd / truth sd -- below 1 means the method smooths);
    and the mean error in the hottest and coolest deciles of the TRUTH
    (negative in the hottest decile = the method under-reads peak heat)."""
    truth = df["truth"]
    hot, cool = truth >= truth.quantile(hot_q), truth <= truth.quantile(1 - hot_q)
    ss_tot = ((truth - truth.mean()) ** 2).sum()
    rows = []
    for m in methods:
        err = df[m] - truth
        rows.append({
            "method": m,
            "bias_c": err.mean(),
            "rmse_c": float(np.sqrt((err ** 2).mean())),
            "mae_c": err.abs().mean(),
            "r2": 1 - (err ** 2).sum() / ss_tot,
            "sd_ratio": df[m].std() / truth.std(),
            "hottest10_bias_c": err[hot].mean(),
            "coolest10_bias_c": err[cool].mean(),
            "n_pixels": len(df),
        })
    return pd.DataFrame(rows)


def block_bootstrap_rmse_diff(df: pd.DataFrame, method_a: str, method_b: str, block_m: float,
                              n_boot: int, seed: int) -> dict:
    """RMSE(a) - RMSE(b) with a 95% interval from a spatial block bootstrap:
    pixels are grouped into block_m x block_m squares and whole blocks are
    resampled, because neighbouring pixels' errors are correlated and a plain
    pixel bootstrap would give falsely narrow intervals."""
    blocks = (df["x"] // block_m).astype(int).astype(str) + "_" + (df["y"] // block_m).astype(int).astype(str)
    sq_a, sq_b = (df[method_a] - df["truth"]) ** 2, (df[method_b] - df["truth"]) ** 2
    per_block = pd.DataFrame({"a": sq_a, "b": sq_b, "blk": blocks.values}).groupby("blk").agg(["sum", "count"])
    sa, sb, cnt = per_block[("a", "sum")].values, per_block[("b", "sum")].values, per_block[("a", "count")].values
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cnt), len(cnt))
        n = cnt[idx].sum()
        diffs.append(np.sqrt(sa[idx].sum() / n) - np.sqrt(sb[idx].sum() / n))
    point = np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())
    return {"comparison": f"{method_a} minus {method_b}", "rmse_diff_c": float(point),
            "ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5)),
            "n_blocks": int(len(cnt)), "block_m": block_m}


# --- Local implementation (numpy) ---------------------------------------------
# The Earth Engine path above exceeds the interactive memory limit when the
# coarse scale is well above 90 m (10 m Sentinel-2 aggregated to 360 m). The
# functions below download the 30 m Landsat LST and 30 m Sentinel-2 indices
# once and do every aggregation, interpolation and regression locally, so any
# factor-of-3 scale pair can be tested with one implementation. Interpolation
# uses scipy (order 1 = bilinear, order 3 = cubic spline) rather than Earth
# Engine's kernels; the 90 -> 30 m pair is run both ways as a cross-check.

def download_grids(lst_30m, s2_indices_10m, crs, bounds_xy, scale=30, tile=512):
    """Fetch LST and NDVI/NDBI/NDWI at `scale` on a grid whose origin is
    snapped to a multiple of 360 m (so 30/90/120/360 m blocks nest exactly).
    `bounds_xy` = (xmin, ymin, xmax, ymax) in `crs` metres. Returns
    (dict of 2-D float arrays, x0, y0)."""
    snap = 360
    xmin, ymin, xmax, ymax = bounds_xy
    x0, y1 = np.floor(xmin / snap) * snap, np.ceil(ymax / snap) * snap
    width = int(np.ceil((xmax - x0) / snap) * snap / scale)
    height = int(np.ceil((y1 - ymin) / snap) * snap / scale)
    indices = _aggregate(s2_indices_10m.reproject(crs=crs, scale=10), crs, scale, max_pixels=(scale // 10) ** 2 + 1)
    image = lst_30m.rename("LST").addBands(indices.select(INDEX_BANDS)).unmask(-9999)
    bands = ["LST"] + INDEX_BANDS
    out = {b: np.full((height, width), np.nan, dtype=np.float64) for b in bands}
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            h, w = min(tile, height - r0), min(tile, width - c0)
            arr = ee.data.computePixels({
                "expression": image, "fileFormat": "NUMPY_NDARRAY",
                "grid": {"dimensions": {"width": w, "height": h}, "crsCode": crs,
                         "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x0 + c0 * scale,
                                             "shearY": 0, "scaleY": -scale, "translateY": y1 - r0 * scale}},
            })
            for b in bands:
                block = arr[b].astype(np.float64)
                block[block <= -9998] = np.nan
                out[b][r0:r0 + h, c0:c0 + w] = block
            print(f"  tile row {r0}, col {c0}: {h}x{w}")
    return out, x0, y1


def block_mean(arr, k, min_valid=0.5):
    """k x k block mean ignoring NaN; NaN where under `min_valid` of the block is valid."""
    if k == 1:
        return arr.copy()
    h, w = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
    blocks = arr[:h, :w].reshape(h // k, k, w // k, k)
    valid = np.isfinite(blocks).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(blocks, axis=(1, 3))
    mean[valid < min_valid] = np.nan
    return mean


def upsample(arr, k, order):
    """Coarse -> fine by factor k. order 0 copies each coarse value into its
    block; order 1/3 interpolate between coarse pixel centres. NaNs are
    filled from the nearest valid pixel before interpolating (and the fine
    result is masked separately by the evaluation)."""
    from scipy import ndimage

    if order == 0:
        return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)
    filled = arr
    if np.isnan(arr).any():
        idx = ndimage.distance_transform_edt(np.isnan(arr), return_distances=False, return_indices=True)
        filled = arr[tuple(idx)]
    return ndimage.zoom(filled, k, order=order, mode="nearest", grid_mode=True)


def local_skill_frame(grids, land_fine, fine, coarse, sample_n, seed, base=30):
    """Every method's fine-scale reconstruction from the coarse aggregate,
    as a per-pixel DataFrame (truth, methods, x, y in metres from the grid
    origin) over valid land pixels. Returns (frame, coefficients)."""
    kf, k = fine // base, coarse // fine
    lst_f = block_mean(grids["LST"], kf)
    idx_f = {b: block_mean(grids[b], kf) for b in INDEX_BANDS}
    lst_c = block_mean(lst_f, k)
    idx_c = {b: block_mean(idx_f[b], k) for b in INDEX_BANDS}
    shape = (lst_c.shape[0] * k, lst_c.shape[1] * k)
    lst_f, land_fine = lst_f[:shape[0], :shape[1]], land_fine[:shape[0], :shape[1]]
    idx_f = {b: a[:shape[0], :shape[1]] for b, a in idx_f.items()}

    rng = np.random.default_rng(seed)

    def fit(bands):
        X = np.column_stack([idx_c[b].ravel() for b in bands])
        y = lst_c.ravel()
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        pick = rng.choice(np.flatnonzero(ok), size=min(sample_n, ok.sum()), replace=False)
        A = np.column_stack([np.ones(len(pick)), X[pick]])
        return np.linalg.lstsq(A, y[pick], rcond=None)[0]

    def predict(coef, bands, idx):
        return coef[0] + sum(c * idx[b] for c, b in zip(coef[1:], bands))

    preds, coefs = {}, {}
    for name, bands in (("tsharp", INDEX_BANDS), ("tsharp_ndvi_only", ["NDVI"])):
        coef = fit(bands)
        coefs[name] = dict(zip(["intercept"] + bands, coef))
        raw = predict(coef, bands, idx_f)
        residual = lst_c - predict(coef, bands, idx_c)
        preds[name] = raw + upsample(residual, k, order=1)
        if name == "tsharp":
            preds["tsharp_no_residual"] = raw
    preds["nearest"] = upsample(lst_c, k, order=0)
    preds["bilinear"] = upsample(lst_c, k, order=1)
    preds["bicubic"] = upsample(lst_c, k, order=3)

    rows, cols = np.indices(shape)
    frame = pd.DataFrame({"truth": lst_f.ravel(), **{m: preds[m].ravel() for m in METHODS},
                          "x": (cols.ravel() + 0.5) * fine, "y": (rows.ravel() + 0.5) * fine,
                          "land": land_fine.ravel()})
    frame = frame[frame["land"] & np.isfinite(frame[["truth"] + METHODS]).all(axis=1)]
    return frame.drop(columns="land").reset_index(drop=True), coefs
