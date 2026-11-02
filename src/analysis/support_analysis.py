"""State-space coverage / dataset-support analysis.

For each district at its most-recent year, compute the local density of the
offline dataset around its embedding -- specifically the inverse of the mean
distance to its k=10 nearest neighbours in the offline-dataset embedding
manifold. Districts with low density are out-of-support for the learned
policy: CQL/IQL recommendations there should be treated with caution.

The per-county aggregate density is mapped onto the CA county polygons.

Outputs:
  results/analysis/dataset_support.json
  figures/support_map.pdf
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "config"))

log = logging.getLogger(__name__)

DATASET_DIR = REPO / "results" / "offline_dataset"
ANALYSIS_DIR = REPO / "results" / "analysis"
FIG_DIR = REPO / "figures"
COUNTY_GEOJSON = (
    Path("c:/Users/visha/Downloads/r_and_d/SFUSD_DATA_ANALYSIS")
    / "dashboard" / "public" / "data" / "ca" / "geo" / "county_polygons.geojson"
)

K_NEIGHBORS = 10
LOW_SUPPORT_PCTL = 25  # bottom-quartile densities flagged as low-support


def _load_env():
    for cand in (
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ):
        if cand.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(cand, override=False)
            except ImportError:
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


def _load_master_panel_county() -> pd.DataFrame:
    _load_env()
    import r2_client as r2  # type: ignore

    df = r2.download("processed/joined/master_panel.parquet")
    keep = ["cds", "year_num", "county_name", "county_code"]
    return df[keep].copy()


def _load_district_index() -> dict[int, str]:
    """Invert the metadata's district_cds_to_index map."""
    meta_path = (
        REPO.parent / "edu-gnn" / "results" / "embeddings"
        / "embedding_metadata.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inv = {int(v): str(k) for k, v in meta["district_cds_to_index"].items()}
    return inv


# --------------------------------------------------------------------------- #
def analyze(fast: bool = False) -> dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(DATASET_DIR / "raw_transitions.npz")
    obs = raw["observations"].astype(np.float32)
    district_indices = raw["district_indices"]
    years = raw["years"]

    seen: dict[int, int] = {}
    for i, (d, y) in enumerate(zip(district_indices, years)):
        if d not in seen or y > years[seen[d]]:
            seen[d] = int(i)
    rows = sorted(seen.values())
    obs_recent = obs[rows]
    d_recent = district_indices[rows]
    n = len(rows)
    log.info("support analysis on %d unique-district recent states", n)

    # k-NN on the FULL embedding manifold (queries -> all training transitions)
    from sklearn.neighbors import NearestNeighbors  # noqa: WPS433

    k_eff = min(K_NEIGHBORS + 1, len(obs))  # +1 because self-row will be included
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(obs)
    dists, _ = nn.kneighbors(obs_recent, n_neighbors=k_eff)
    # Drop the self-distance (will be 0 if obs_recent is a subset of obs)
    if dists.shape[1] > 1:
        dists = dists[:, 1:]
    mean_dists = dists.mean(axis=1)
    density = 1.0 / (mean_dists + 1e-9)

    # Map district index -> CDS -> county
    idx_to_cds = _load_district_index()
    panel = _load_master_panel_county()
    cds_to_county = (
        panel.dropna(subset=["county_name"])
        .drop_duplicates(subset=["cds"])
        .set_index("cds")[["county_name", "county_code"]]
        .to_dict("index")
    )

    per_district = []
    for i, d in enumerate(d_recent):
        cds = idx_to_cds.get(int(d))
        county_info = cds_to_county.get(cds, {})
        per_district.append({
            "district_idx": int(d),
            "cds": cds,
            "county_name": county_info.get("county_name"),
            "county_code": county_info.get("county_code"),
            "mean_dist_to_k_neighbors": float(mean_dists[i]),
            "density": float(density[i]),
        })

    densities = np.array([p["density"] for p in per_district])
    low_thresh = float(np.percentile(densities, LOW_SUPPORT_PCTL))
    n_low = int((densities <= low_thresh).sum())
    for p in per_district:
        p["low_support"] = bool(p["density"] <= low_thresh)

    # County aggregate
    county_density: dict[str, list[float]] = {}
    for p in per_district:
        cn = (p.get("county_name") or "").replace(" County", "")
        if not cn:
            continue
        county_density.setdefault(cn, []).append(p["density"])
    county_summary = {
        name: {
            "mean_density": float(np.mean(vals)),
            "median_density": float(np.median(vals)),
            "n_districts": int(len(vals)),
        }
        for name, vals in county_density.items()
    }

    payload = {
        "k_neighbors": K_NEIGHBORS,
        "n_districts_evaluated": int(n),
        "n_low_support_districts": n_low,
        "low_support_threshold_density": low_thresh,
        "density_distribution": {
            "mean": float(np.mean(densities)),
            "median": float(np.median(densities)),
            "p10": float(np.percentile(densities, 10)),
            "p25": float(np.percentile(densities, 25)),
            "p75": float(np.percentile(densities, 75)),
            "p90": float(np.percentile(densities, 90)),
        },
        "county_summary": county_summary,
        "per_district": per_district,
        "fast": bool(fast),
    }
    (ANALYSIS_DIR / "dataset_support.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Map figure (geopandas + matplotlib)
    try:
        _draw_support_map(county_summary)
    except Exception as e:  # noqa: BLE001
        log.warning("support map figure failed: %s", e)

    log.info("support: %d/%d districts low (threshold density=%.4f)",
             n_low, n, low_thresh)
    return payload


def _draw_support_map(county_summary: dict[str, dict[str, float]]):
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not COUNTY_GEOJSON.exists():
        log.warning("county geojson missing at %s; skipping map", COUNTY_GEOJSON)
        return
    gdf = gpd.read_file(COUNTY_GEOJSON)
    gdf["mean_density"] = gdf["county_name"].map(
        lambda nm: county_summary.get(nm, {}).get("mean_density")
    )

    fig, ax = plt.subplots(figsize=(7, 8))
    gdf.plot(
        column="mean_density",
        ax=ax,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Mean offline-dataset density", "shrink": 0.6},
        missing_kwds={"color": "#dddddd", "edgecolor": "white", "hatch": "//"},
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_axis_off()
    ax.set_title("CA-county offline-RL dataset support\n"
                 "(grey = no covered districts)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "support_map.pdf")
    plt.close(fig)


def run(fast: bool = False) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    return analyze(fast=fast)


if __name__ == "__main__":
    run(fast=False)
