"""Smoke test: imports all module stubs and verifies they raise
NotImplementedError at run() time. Phase A scaffold guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without an installed package
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_imports_and_not_implemented():
    import pytest
    from src.data import offline_dataset as _data_offline_dataset
    from src.models import cql_agent as _models_cql_agent
    from src.models import iql_agent as _models_iql_agent
    from src.analysis import policy_comparison as _analysis_policy_comparison
    from src.analysis import support_analysis as _analysis_support_analysis
    mods = [_data_offline_dataset, _models_cql_agent, _models_iql_agent, _analysis_policy_comparison, _analysis_support_analysis]
    assert len(mods) > 0, "no modules collected"
    for m in mods:
        with pytest.raises(NotImplementedError):
            m.run()


def test_config_loads():
    sys.path.insert(0, str(REPO / "src"))
    from utils.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg
