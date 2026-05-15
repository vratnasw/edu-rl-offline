# edu-rl-offline

**Status:** SCAFFOLD (Phase A). Module files raise `NotImplementedError`.
Phase B implementation pending.

**Target journal:** ICML or Educational Researcher

## Research question

How well can policies learned offline from observational district panels (no simulator, no world model) match or exceed model-based RL trained against the Layer 5 world model?

## Description

Offline RL with CQL + IQL -- model-free policy learning directly from historical district panels.

## Architecture overview

Two offline-RL algorithms (Conservative Q-Learning with conservative_weight=5.0, and Implicit Q-Learning with expectile=0.7) trained for 500k steps on the master panel.

## Layout

```
config/                # config.yaml + canonical r2_client.py
src/                   # module stubs (raise NotImplementedError)
scripts/preflight.py   # infra check (R2 + checkpoints + GPU)
scripts/run_*.py       # orchestrator stub
tests/test_smoke.py    # imports + NotImplementedError assertions
notebooks/             # exploration stub
figures/, results/     # output dirs
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in R2 credentials
python scripts/preflight.py
python scripts/run_*.py --fast
pytest tests/
```

## Phase B will produce

- CQL + IQL checkpoints under results/checkpoints/
- Policy-comparison report vs. model-based baseline
- Support / coverage analysis (state-action density)

## Operational note

The AECF national pull is running in background on this dev machine; do
not attempt parallel heavy compute (training, large embedding jobs) that
might conflict with it.
