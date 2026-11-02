"""Three-way policy comparison: Layer-6 SAC vs offline CQL vs offline IQL.

For each district at its most-recent year:
  1. Load the district embedding (Layer-4).
  2. Query each agent for a deterministic recommended action.
  3. Compute pairwise cosine similarity and sign-agreement statistics.
  4. Aggregate by KMeans archetype (k=5) over embeddings.

If the SAC checkpoint at edu-rl-agent/results/checkpoints/best.pt is missing,
fall back to the empirical (behavioural) action distribution mean as the
baseline policy.

Outputs:
  results/analysis/three_way_agreement.json
  figures/policy_agreement.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "config"))

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
DATASET_DIR = REPO / "results" / "offline_dataset"
CQL_POLICY = REPO / "results" / "cql" / "policy.d3"
IQL_POLICY = REPO / "results" / "iql" / "policy.d3"
SAC_CHECKPOINT = (
    REPO.parent / "edu-rl-agent" / "results" / "checkpoints" / "best.pt"
)
ANALYSIS_DIR = REPO / "results" / "analysis"
FIG_DIR = REPO / "figures"

K_ARCHETYPES = 5
STATE_DIM = 128
ACTION_DIM = 5


# --------------------------------------------------------------------------- #
class _SACActor(nn.Module):
    """Architecture-matched stand-in for the Layer-6 SAC actor.

    Mirrors edu-rl-agent/src/agent/sac_lagrangian.py::Actor closely enough to
    load its state_dict without depending on edu-rl-agent at import time.
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256, action_low: float = -3.0,
                 action_high: float = 3.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head_mean = nn.Linear(hidden_dim, action_dim)
        self.head_log_std = nn.Linear(hidden_dim, action_dim)
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = self.body(x)
        mean = self.head_mean(h)
        squashed = torch.tanh(mean)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        return (squashed * scale + offset).cpu().numpy()


def _load_sac() -> Any:
    """Return an object with a .predict(obs)->np.ndarray method, or None."""
    if not SAC_CHECKPOINT.exists():
        return None
    ck = torch.load(SAC_CHECKPOINT, map_location="cpu", weights_only=False)
    if "actor" not in ck:
        return None
    actor = _SACActor()
    try:
        actor.load_state_dict(ck["actor"], strict=False)
    except Exception as e:  # noqa: BLE001
        log.warning("SAC actor load failed (%s); falling back to baseline", e)
        return None
    actor.eval()
    return actor


def _load_cql() -> Any:
    if not CQL_POLICY.exists():
        return None
    from d3rlpy.algos import CQLConfig  # noqa: WPS433

    algo = CQLConfig().create(device="cpu:0")
    algo.load_model(str(CQL_POLICY))
    return algo


def _load_iql() -> Any:
    if not IQL_POLICY.exists():
        return None
    from d3rlpy.algos import IQLConfig  # noqa: WPS433

    algo = IQLConfig().create(device="cpu:0")
    algo.load_model(str(IQL_POLICY))
    return algo


# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity in [-1, 1]."""
    na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return ((a * b).sum(axis=1, keepdims=True) / (na * nb)).squeeze(1)


def _sign_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row fraction of dims where sign(a)==sign(b)."""
    return (np.sign(a) == np.sign(b)).mean(axis=1)


# --------------------------------------------------------------------------- #
def compare(fast: bool = False) -> dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(DATASET_DIR / "raw_transitions.npz")
    obs = raw["observations"]
    actions = raw["actions"]
    district_indices = raw["district_indices"]
    years = raw["years"]

    # Pick the most-recent transition for each district
    keep_rows: list[int] = []
    seen: dict[int, int] = {}  # district_idx -> row with max year so far
    for i, (d, y) in enumerate(zip(district_indices, years)):
        if d not in seen or y > years[seen[d]]:
            seen[d] = int(i)
    keep_rows = sorted(seen.values())
    obs_recent = obs[keep_rows].astype(np.float32)
    actions_recent = actions[keep_rows].astype(np.float32)
    n_districts = len(keep_rows)
    log.info("evaluating policies on %d unique-district most-recent states",
             n_districts)

    sac = _load_sac()
    cql = _load_cql()
    iql = _load_iql()
    sac_kind = "neural" if sac is not None else "behavioural_mean_baseline"
    if sac is None:
        # Baseline: historical behavioural mean over the full panel
        mean_action = actions.mean(axis=0, dtype=np.float32)
        log.info("SAC unavailable; using behavioural-mean baseline %s",
                 mean_action.tolist())

        class _Baseline:
            def predict(self_, X):  # noqa: D401, N803
                return np.broadcast_to(mean_action, (len(X), ACTION_DIM)).copy()

        sac = _Baseline()
    if cql is None:
        raise FileNotFoundError(
            f"CQL policy missing at {CQL_POLICY} -- train cql_agent first")
    if iql is None:
        raise FileNotFoundError(
            f"IQL policy missing at {IQL_POLICY} -- train iql_agent first")

    a_sac = np.asarray(sac.predict(obs_recent), dtype=np.float32)
    a_cql = np.asarray(cql.predict(obs_recent), dtype=np.float32)
    a_iql = np.asarray(iql.predict(obs_recent), dtype=np.float32)

    # Pairwise cosine matrix per district -> aggregate to a 3x3 mean matrix
    pairs = {
        ("SAC", "CQL"): _cosine(a_sac, a_cql),
        ("SAC", "IQL"): _cosine(a_sac, a_iql),
        ("CQL", "IQL"): _cosine(a_cql, a_iql),
    }
    labels = ["SAC", "CQL", "IQL"]
    matrix = np.ones((3, 3), dtype=np.float64)
    for (a, b), vec in pairs.items():
        i, j = labels.index(a), labels.index(b)
        m = float(np.mean(vec))
        matrix[i, j] = m
        matrix[j, i] = m

    # Sign-agreement (per-dim, then headline = mean over districts)
    sign_pairs = {
        "SAC_vs_CQL": _sign_agreement(a_sac, a_cql),
        "SAC_vs_IQL": _sign_agreement(a_sac, a_iql),
        "CQL_vs_IQL": _sign_agreement(a_cql, a_iql),
    }
    # Headline: districts where all 3 agents agree on direction of EVERY dim
    sign_sac = np.sign(a_sac)
    sign_cql = np.sign(a_cql)
    sign_iql = np.sign(a_iql)
    all_agree_per_dim = (sign_sac == sign_cql) & (sign_cql == sign_iql)
    all_three_dir_full = all_agree_per_dim.all(axis=1).mean()

    # Cosine-based three-way agreement: mean of the 3 pairwise cosines
    three_way_cos = float(np.mean(
        [pairs[("SAC", "CQL")], pairs[("SAC", "IQL")], pairs[("CQL", "IQL")]]
    ))

    # KMeans archetype agreement
    from sklearn.cluster import KMeans  # noqa: WPS433

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=min(K_ARCHETYPES, max(2, n_districts // 4)),
                    n_init=10, random_state=42).fit(obs_recent)
    clusters = km.labels_.astype(int)
    by_arch: dict[int, dict[str, float]] = {}
    for c in sorted(np.unique(clusters)):
        mask = clusters == c
        by_arch[int(c)] = {
            "n_districts": int(mask.sum()),
            "mean_cos_SAC_CQL": float(np.mean(pairs[("SAC", "CQL")][mask])),
            "mean_cos_SAC_IQL": float(np.mean(pairs[("SAC", "IQL")][mask])),
            "mean_cos_CQL_IQL": float(np.mean(pairs[("CQL", "IQL")][mask])),
            "all_three_sign_agree_frac":
                float(all_agree_per_dim[mask].all(axis=1).mean()),
        }

    # Persist the JSON
    payload = {
        "n_districts_evaluated": int(n_districts),
        "sac_source": sac_kind,
        "pairwise_mean_cosine": matrix.tolist(),
        "pairwise_labels": labels,
        "pairwise_sign_agreement_fractions": {
            k: float(np.mean(v)) for k, v in sign_pairs.items()
        },
        "three_way_agreement_fraction": three_way_cos,
        "all_three_agree_on_direction_fraction": float(all_three_dir_full),
        "archetype_breakdown": by_arch,
        "fast": bool(fast),
    }
    (ANALYSIS_DIR / "three_way_agreement.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Heatmap figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color="black" if abs(matrix[i, j]) < 0.7 else "white",
                        fontsize=10)
        ax.set_title("Pairwise mean cosine similarity\n"
                     f"(N={n_districts}, sac={sac_kind})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "policy_agreement.pdf")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("heatmap figure failed: %s", e)

    log.info("three-way cos=%.3f, all-dims-agree=%.3f",
             three_way_cos, float(all_three_dir_full))
    return payload


def run(fast: bool = False) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    return compare(fast=fast)


if __name__ == "__main__":
    run(fast=False)
