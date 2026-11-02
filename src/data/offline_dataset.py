"""Offline RL dataset constructor over the master panel.

Builds (state, action, reward, next_state) transitions from the historical
district panel. Each district contributes (T-1) transitions, one per
year_T -> year_T+1 pairing where both the district embedding and the policy
proxies are available.

States:    Layer-4 GNN district embedding (128-dim).
Actions:   5-dim reconstructed YoY changes in policy proxies, z-scored.
Rewards:   weighted sum of CAASPP outcome z-deltas, weights derived from the
           anchor effects in edu-causal-rl/results/interventional/summary.json.
Augment:   Gaussian noise N(0, 0.01) added to state embeddings for coverage.

Outputs (results/offline_dataset/):
  dataset.h5            d3rlpy MDPDataset dump
  behavioral_dist.json  per-dim histograms of the empirical action distribution
  stats.json            dataset shape + per-dim action/reward summary statistics
  raw_transitions.npz   tensors (S, A, R, S_next, terminals, district_idx, year)
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Repo paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "config"))

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration constants (no magic numbers in the body)
# --------------------------------------------------------------------------- #
MASTER_PANEL_KEY = "processed/joined/master_panel.parquet"
EMBEDDINGS_KEY = "embeddings/edu-gnn/district_embeddings.pt"
EMBEDDINGS_LOCAL_FALLBACK = (
    REPO.parent / "edu-gnn" / "results" / "embeddings" / "district_embeddings.pt"
)
METADATA_LOCAL_FALLBACK = (
    REPO.parent / "edu-gnn" / "results" / "embeddings" / "embedding_metadata.json"
)
ANCHOR_PATH = (
    REPO.parent / "edu-causal-rl" / "results" / "interventional" / "summary.json"
)

# 5 policy proxies in the action vector (ordered)
ACTION_PROXIES = [
    "per_pupil_expenditure",   # bea_gdp_total / enrollment
    "teacher_student_ratio",   # enrollment / experienced_teachers_count
    "ell_intensity",           # pct_ell
    "intervention_budget",     # census_saipe_child_poverty_rate (inverted-sign proxy)
    "infrastructure",          # kd_econ_housing_broadband_mean
]

# 5 CAASPP-anchored outcomes for the reward
OUTCOMES = [
    "caaspp_math_met_pct",
    "caaspp_ela_met_pct",
    "chronic_absenteeism_rate",
    "suspension_rate_pct",
    "graduation_rate_pct",
]
# Sign convention: positive = good. Higher absent/suspension/etc. is bad.
OUTCOME_SIGN = {
    "caaspp_math_met_pct": +1.0,
    "caaspp_ela_met_pct": +1.0,
    "graduation_rate_pct": +1.0,
    "chronic_absenteeism_rate": -1.0,
    "suspension_rate_pct": -1.0,
}

STATE_DIM = 128
ACTION_DIM = 5
STATE_NOISE_STD = 0.01

OUT_DIR = REPO / "results" / "offline_dataset"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_env():
    """Try the same .env paths preflight does, so credentials are present."""
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


def _load_master_panel() -> pd.DataFrame:
    _load_env()
    import r2_client as r2  # type: ignore

    df = r2.download(MASTER_PANEL_KEY)
    log.info("master panel loaded: %s rows x %s cols", *df.shape)
    return df


def _load_embeddings() -> tuple[torch.Tensor, dict[str, int], dict[str, int]]:
    """Load (tensor[N,T,D], district_cds_to_index, year_to_index) from R2 or local."""
    _load_env()
    try:
        import r2_client as r2  # type: ignore

        client = r2._client()  # noqa: SLF001
        obj = client.get_object(Bucket=r2.bucket_name(), Key=EMBEDDINGS_KEY)
        tensor = torch.load(io.BytesIO(obj["Body"].read()),
                            map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        log.warning("R2 embedding load failed (%s); trying local fallback", e)
        tensor = torch.load(EMBEDDINGS_LOCAL_FALLBACK,
                            map_location="cpu", weights_only=False)
    if not METADATA_LOCAL_FALLBACK.exists():
        raise FileNotFoundError(
            f"embedding metadata not found at {METADATA_LOCAL_FALLBACK}; "
            "cannot map embedding indices to district CDS codes")
    meta = json.loads(METADATA_LOCAL_FALLBACK.read_text(encoding="utf-8"))
    dist_map = {str(k): int(v) for k, v in meta["district_cds_to_index"].items()}
    year_map = {str(k): int(v) for k, v in meta["year_to_index"].items()}
    log.info("embeddings: %s, districts=%d, years=%d",
             tuple(tensor.shape), len(dist_map), len(year_map))
    return tensor, dist_map, year_map


def _load_anchor_effects() -> dict[str, float]:
    """Return {outcome: |anchor|} normalised to sum to 1; falls back to uniform."""
    if not ANCHOR_PATH.exists():
        log.warning("anchor effects not found at %s; using uniform weights",
                    ANCHOR_PATH)
        return {o: 1.0 / len(OUTCOMES) for o in OUTCOMES}
    s = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    weights = {}
    for o in OUTCOMES:
        info = s.get("outcomes", {}).get(o, {})
        weights[o] = abs(float(info.get("anchor", 0.0)))
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _compute_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 policy-proxy columns to df. Missing inputs -> NaN."""
    df = df.copy()
    # 1. per-pupil expenditure proxy: GDP per enrolled student
    df["per_pupil_expenditure"] = df["bea_gdp_total"] / df["enrollment"].replace(0, np.nan)
    # 2. teacher-student ratio: students per experienced-teacher
    #    pct_experienced_teachers is per 100 students -> teachers / 100 students
    teacher_share = df["pct_experienced_teachers"].replace(0, np.nan)
    df["teacher_student_ratio"] = 1.0 / teacher_share  # higher = more students/teacher
    # 3. ELL intensity
    df["ell_intensity"] = df["pct_ell"]
    # 4. intervention budget proxy: inverse child poverty (higher = more capacity)
    df["intervention_budget"] = -df["census_saipe_child_poverty_rate"]
    # 5. infrastructure proxy
    if "kd_econ_housing_broadband_mean" in df.columns:
        df["infrastructure"] = df["kd_econ_housing_broadband_mean"]
    elif "kd_econ_housing_income_level_mean" in df.columns:
        df["infrastructure"] = df["kd_econ_housing_income_level_mean"]
    else:
        df["infrastructure"] = np.nan
    return df


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Column-wise z-score with NaN-safe stats."""
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (arr - mu) / sd


def _histogram(vec: np.ndarray, bins: int = 20) -> dict:
    """JSON-serialisable histogram."""
    vec = vec[~np.isnan(vec)]
    if len(vec) == 0:
        return {"bin_edges": [], "counts": [], "mean": None, "std": None,
                "min": None, "max": None}
    counts, edges = np.histogram(vec, bins=bins)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
    }


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build(fast: bool = False) -> dict[str, Any]:
    """Construct the offline RL dataset and persist it.

    Returns a stats dict (also dumped to stats.json).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_master_panel()
    embeds, district_idx, year_idx = _load_embeddings()
    weights = _load_anchor_effects()

    df = _compute_proxies(df)

    # Sort -> deterministic iteration
    df = df.sort_values(["cds", "year_num"]).reset_index(drop=True)

    if fast:
        # Subsample districts to ~30% for the smoke run
        unique = sorted(df["cds"].dropna().unique().tolist())
        rng = np.random.default_rng(42)
        keep_n = max(8, int(len(unique) * 0.30))
        keep = set(rng.choice(unique, size=keep_n, replace=False).tolist())
        df = df[df["cds"].isin(keep)].reset_index(drop=True)
        log.info("--fast: subsampled to %d districts (from %d)", keep_n, len(unique))

    # Z-score the action proxies + outcomes across the full retained panel
    action_raw = df[ACTION_PROXIES].to_numpy(dtype=np.float64)
    action_z = _zscore(action_raw)
    df_action_z = pd.DataFrame(action_z, columns=ACTION_PROXIES, index=df.index)

    outcome_raw = df[OUTCOMES].to_numpy(dtype=np.float64)
    outcome_z = _zscore(outcome_raw)
    df_out_z = pd.DataFrame(outcome_z, columns=OUTCOMES, index=df.index)

    # Build transitions: (state_T, action_T, reward_T, state_{T+1})
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[float] = []
    next_obs_list: list[np.ndarray] = []
    terminal_list: list[float] = []
    timeout_list: list[float] = []
    district_idx_list: list[int] = []
    year_list: list[int] = []

    rng = np.random.default_rng(7)
    n_districts_used = 0
    n_skip_no_emb = 0
    n_skip_nan = 0

    for cds, group in df.groupby("cds", sort=False):
        if cds not in district_idx:
            n_skip_no_emb += 1
            continue
        d_idx = district_idx[cds]
        group = group.sort_values("year_num").reset_index(drop=True)
        used_any = False
        for i in range(len(group) - 1):
            y_t = int(group.loc[i, "year_num"])
            y_tp1 = int(group.loc[i + 1, "year_num"])
            y_key_t, y_key_tp1 = str(y_t), str(y_tp1)
            if y_key_t not in year_idx or y_key_tp1 not in year_idx:
                continue
            # Action: YoY change in z-scored proxies (T+1 - T)
            a_t = df_action_z.loc[group.index[i + 1], ACTION_PROXIES].to_numpy(
                dtype=np.float64
            ) - df_action_z.loc[group.index[i], ACTION_PROXIES].to_numpy(
                dtype=np.float64
            )
            # Reward: weighted YoY change in outcomes (sign-corrected)
            outc_t = df_out_z.loc[group.index[i], OUTCOMES].to_numpy(dtype=np.float64)
            outc_tp1 = df_out_z.loc[group.index[i + 1], OUTCOMES].to_numpy(dtype=np.float64)
            d_outcome = outc_tp1 - outc_t
            signed = np.array(
                [OUTCOME_SIGN[o] for o in OUTCOMES], dtype=np.float64
            ) * d_outcome
            w_vec = np.array([weights[o] for o in OUTCOMES], dtype=np.float64)
            mask = ~np.isnan(signed)
            if mask.sum() == 0:
                n_skip_nan += 1
                continue
            reward = float(np.sum(np.where(mask, signed * w_vec, 0.0)) /
                           max(float(w_vec[mask].sum()), 1e-9))
            # Replace any remaining NaN in action with 0 (panel gap mid-row)
            a_t = np.where(np.isnan(a_t), 0.0, a_t).astype(np.float32)

            s_t = embeds[d_idx, year_idx[y_key_t]].numpy().astype(np.float32)
            s_tp1 = embeds[d_idx, year_idx[y_key_tp1]].numpy().astype(np.float32)
            # Augment state with N(0, 0.01) for coverage
            s_t = s_t + rng.normal(0.0, STATE_NOISE_STD, size=s_t.shape).astype(np.float32)
            s_tp1_aug = s_tp1 + rng.normal(0.0, STATE_NOISE_STD,
                                           size=s_tp1.shape).astype(np.float32)

            obs_list.append(s_t)
            act_list.append(a_t)
            rew_list.append(reward)
            next_obs_list.append(s_tp1_aug)
            is_last = (i == len(group) - 2)
            terminal_list.append(1.0 if is_last else 0.0)
            timeout_list.append(0.0)
            district_idx_list.append(int(d_idx))
            year_list.append(y_t)
            used_any = True
        if used_any:
            n_districts_used += 1

    if not obs_list:
        raise RuntimeError(
            "no transitions built -- check master panel, embedding metadata, "
            "and overlap between district CDS codes")

    obs = np.stack(obs_list).astype(np.float32)
    actions = np.stack(act_list).astype(np.float32)
    rewards = np.array(rew_list, dtype=np.float32)
    next_obs = np.stack(next_obs_list).astype(np.float32)
    terminals = np.array(terminal_list, dtype=np.float32)
    timeouts = np.array(timeout_list, dtype=np.float32)
    district_indices = np.array(district_idx_list, dtype=np.int64)
    years = np.array(year_list, dtype=np.int64)

    log.info(
        "built %d transitions across %d districts; "
        "skipped %d districts missing embedding, %d transitions with all-NaN outcomes",
        len(obs), n_districts_used, n_skip_no_emb, n_skip_nan,
    )

    # --------------------------------------------------------------------- #
    # Persist to disk
    # --------------------------------------------------------------------- #
    # d3rlpy MDPDataset wants observations + next-via-terminal flag. Construct
    # the flattened observation array (each transition is its own 1-step episode
    # in our framing because the embedding-augmentation breaks within-episode
    # determinism if we re-use observations).
    from d3rlpy.dataset import MDPDataset  # noqa: WPS433

    # For MDPDataset, we treat each district as one trajectory: obs[t] is
    # rebuilt by stacking obs_t for the first transition of each district,
    # then next_obs for subsequent ones. terminal flag marks the last row.
    # The simplest faithful representation is to stack (obs, next_obs) into a
    # per-episode time series.
    dataset = MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    h5_path = OUT_DIR / "dataset.h5"
    # h5py requires the file handle to be readable too (it uses a memory-mapped
    # view internally). "w+b" gives us read+write on a freshly-truncated file.
    with open(h5_path, "w+b") as fh:
        dataset.dump(fh)

    # Save raw tensors for downstream modules that don't want d3rlpy
    np.savez_compressed(
        OUT_DIR / "raw_transitions.npz",
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=terminals,
        timeouts=timeouts,
        district_indices=district_indices,
        years=years,
    )

    # Behavioural distribution: per-dim histograms of the empirical actions
    behavioural = {
        "action_proxies": ACTION_PROXIES,
        "histograms": {
            ACTION_PROXIES[i]: _histogram(actions[:, i]) for i in range(ACTION_DIM)
        },
    }
    (OUT_DIR / "behavioral_dist.json").write_text(
        json.dumps(behavioural, indent=2), encoding="utf-8"
    )

    stats = {
        "n_transitions": int(len(obs)),
        "n_districts": int(n_districts_used),
        "n_years": int(len(np.unique(years))),
        "state_dim": int(obs.shape[1]),
        "action_dim": int(actions.shape[1]),
        "n_skip_no_embedding": int(n_skip_no_emb),
        "n_skip_all_nan_outcome": int(n_skip_nan),
        "outcome_weights": weights,
        "action_proxies": ACTION_PROXIES,
        "reward_summary": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        },
        "fast_mode": bool(fast),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    log.info("offline dataset persisted to %s", OUT_DIR)
    return stats


def run(fast: bool = False) -> dict[str, Any]:
    """Public entry point used by orchestrator + tests."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    return build(fast=fast)


if __name__ == "__main__":
    run(fast=False)
