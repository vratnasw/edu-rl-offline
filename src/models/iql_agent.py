"""Implicit Q-Learning agent (expectile=0.7).

Reads results/offline_dataset/dataset.h5 + raw_transitions.npz produced by
src/data/offline_dataset.py and trains a d3rlpy IQL policy on it.

Mirrors src/models/cql_agent.py for direct algorithmic comparison.

Outputs:
  results/iql/policy.d3        d3rlpy save
  results/iql/results.json     {final_loss, bc_error, training_steps, ...}
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "config"))

log = logging.getLogger(__name__)

DATASET_DIR = REPO / "results" / "offline_dataset"
OUT_DIR = REPO / "results" / "iql"

IQL_CONFIG_FULL = {
    "expectile": 0.7,
    "batch_size": 256,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
}
IQL_CONFIG_FAST = {
    "expectile": 0.7,
    "batch_size": 64,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
}
N_STEPS_FULL = 500_000
N_STEPS_FAST = 5_000


def _load_dataset() -> tuple[Any, np.ndarray, np.ndarray]:
    from d3rlpy.dataset import MDPDataset  # noqa: WPS433

    raw = np.load(DATASET_DIR / "raw_transitions.npz")
    obs = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    rewards = raw["rewards"].astype(np.float32)
    terminals = raw["terminals"].astype(np.float32)
    timeouts = raw["timeouts"].astype(np.float32)
    dataset = MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    return dataset, obs, actions


def _compute_bc_error(algo: Any, obs: np.ndarray, actions: np.ndarray) -> float:
    pred = np.asarray(algo.predict(obs.astype(np.float32)))
    return float(np.mean(np.linalg.norm(pred - actions, axis=1)))


def train(fast: bool = False) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    from d3rlpy.algos import IQLConfig  # noqa: WPS433
    from d3rlpy.logging.file_adapter import FileAdapterFactory  # noqa: WPS433

    n_steps = N_STEPS_FAST if fast else N_STEPS_FULL
    iql_cfg = IQL_CONFIG_FAST if fast else IQL_CONFIG_FULL
    log.info("training IQL for %d steps (fast=%s)", n_steps, fast)
    dataset, obs, actions = _load_dataset()

    cfg = IQLConfig(**iql_cfg)
    algo = cfg.create(device="cpu:0")

    t0 = time.time()
    n_steps_per_epoch = max(500, min(n_steps // 2, 10_000))
    history = algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name="iql",
        with_timestamp=False,
        show_progress=False,
        save_interval=10_000_000,
        logger_adapter=FileAdapterFactory(root_dir=str(OUT_DIR / "d3rlpy_logs")),
    )
    elapsed = time.time() - t0

    final_metrics = history[-1][1] if history else {}
    final_loss = float(
        final_metrics.get("loss")
        or final_metrics.get("critic_loss")
        or final_metrics.get("actor_loss")
        or 0.0
    )

    bc_error = _compute_bc_error(algo, obs, actions)
    save_path = OUT_DIR / "policy.d3"
    algo.save(str(save_path))

    results = {
        "algo": "IQL",
        "training_steps": int(n_steps),
        "elapsed_s": round(elapsed, 2),
        "final_loss": final_loss,
        "final_metrics": {k: float(v) for k, v in final_metrics.items()
                          if isinstance(v, (int, float))},
        "bc_error": bc_error,
        "config": {**iql_cfg, "fast": fast},
        "n_train_transitions": int(len(obs)),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    log.info("IQL trained in %.1fs, bc_error=%.4f, final_loss=%.4f",
             elapsed, bc_error, final_loss)
    return results


def run(fast: bool = False) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    return train(fast=fast)


if __name__ == "__main__":
    run(fast=False)
