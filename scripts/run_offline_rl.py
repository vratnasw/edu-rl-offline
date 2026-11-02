"""Phase B orchestrator for edu-rl-offline.

Runs the full pipeline end-to-end:

  1. Build the offline dataset (src.data.offline_dataset)
  2. Train CQL (src.models.cql_agent)
  3. Train IQL (src.models.iql_agent)
  4. Three-way policy comparison (src.analysis.policy_comparison)
  5. Dataset-support analysis (src.analysis.support_analysis)
  6. Emit results/offline_paper_summary.json

With --fast: 5000-step CQL+IQL, 30% district subsample (fast-budget mode).
With --full or no flag: 500_000 steps + full panel.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")


def _cap_torch_threads():
    """Phase B is CPU-only and runs alongside the AECF national pull -- cap
    threads so we don't starve that job."""
    try:
        import torch  # noqa: WPS433

        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="5000-step training, 30%% district subsample, <15min")
    ap.add_argument("--full", action="store_true",
                    help="500k-step training, full panel (no-op if --fast set)")
    args = ap.parse_args()
    if args.full and args.fast:
        print("both --fast and --full set; --fast wins", file=sys.stderr)
    fast = bool(args.fast)

    _setup_logging()
    _cap_torch_threads()
    t0 = time.time()
    log.info("=== edu-rl-offline orchestrator (fast=%s) ===", fast)

    # 1. Build offline dataset
    from src.data.offline_dataset import run as run_dataset  # noqa: WPS433

    log.info("[1/5] building offline dataset...")
    ds_stats = run_dataset(fast=fast)

    # 2. CQL
    from src.models.cql_agent import run as run_cql  # noqa: WPS433

    log.info("[2/5] training CQL...")
    cql_results = run_cql(fast=fast)

    # 3. IQL
    from src.models.iql_agent import run as run_iql  # noqa: WPS433

    log.info("[3/5] training IQL...")
    iql_results = run_iql(fast=fast)

    # 4. Policy comparison
    from src.analysis.policy_comparison import run as run_compare  # noqa: WPS433

    log.info("[4/5] three-way policy comparison...")
    compare_results = run_compare(fast=fast)

    # 5. Support analysis
    from src.analysis.support_analysis import run as run_support  # noqa: WPS433

    log.info("[5/5] dataset support analysis...")
    support_results = run_support(fast=fast)

    # Emit paper summary
    summary = {
        "dataset_size": int(ds_stats["n_transitions"]),
        "n_districts": int(ds_stats["n_districts"]),
        "n_years": int(ds_stats["n_years"]),
        "cql_bc_error": float(cql_results["bc_error"]),
        "iql_bc_error": float(iql_results["bc_error"]),
        "three_way_agreement_fraction":
            float(compare_results["three_way_agreement_fraction"]),
        "all_three_agree_on_direction_fraction":
            float(compare_results["all_three_agree_on_direction_fraction"]),
        "low_support_districts_n":
            int(support_results["n_low_support_districts"]),
        "fast_mode": fast,
        "elapsed_s": round(time.time() - t0, 2),
    }
    out_path = REPO / "results" / "offline_paper_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("=== summary written to %s in %.1fs ===", out_path,
             summary["elapsed_s"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
