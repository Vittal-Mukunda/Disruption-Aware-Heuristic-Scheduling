"""Unit tests for the offline-FQI baseline (baselines/offline_fqi.py).

Acceptance: the transition logger produces correctly shaped transitions under
the configured behaviour policy; fitted Q-iteration trains and yields a usable
model; the policy returns valid rules and obeys the FEFO action mask; save/load
round-trips; and the trained baseline plugs into the evaluation harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.offline_fqi import (
    MASKED_RULE,
    PCT_PERISHABLE_IDX,
    OfflineFQIPolicy,
    _mask_fefo,
    _q_values,
    load_offline_fqi,
    log_transitions,
    save_offline_fqi,
    subset_transitions,
    train_fqi,
)
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from seed import shift_corpora
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.state_extractor import N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
N_SMOKE_SHIFTS = 3
_TINY_XGB = {"max_depth": 3, "n_estimators": 30, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def cfg():
    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    # The regime GMM belongs to a trained Stage-3 run that need not exist during
    # unit tests; the feature-parity path is exercised by the e9 driver.
    cfg.baselines.offline_fqi.use_regime_features = False
    return cfg


@pytest.fixture(scope="module")
def pool(cfg):
    return resolve_pool(cfg)


@pytest.fixture(scope="module")
def transitions(cfg, pool):
    seeds = shift_corpora(cfg)["train"][:N_SMOKE_SHIFTS]
    return log_transitions(seeds, cfg, pool)


@pytest.fixture(scope="module")
def fqi_model(cfg, transitions, pool):
    return train_fqi(
        transitions, gamma=0.95, n_iterations=3, xgb_params=_TINY_XGB,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        fefo_idx=pool.index(MASKED_RULE) if MASKED_RULE in pool else None,
        model_seed=int(cfg.seeds.model),
    )


def _policy(cfg, model, pool) -> OfflineFQIPolicy:
    return OfflineFQIPolicy(
        model=model,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        arms=pool,
    )


def test_log_transitions_shapes(transitions, pool):
    n = N_SMOKE_SHIFTS * 32
    assert transitions["states"].shape == (n, N_FEATURES)
    assert transitions["next_states"].shape == (n, N_FEATURES)
    assert transitions["actions"].shape == (n,)
    assert int(transitions["n_actions"][0]) == len(pool)
    # exactly one terminal transition per shift
    assert int(transitions["dones"].sum()) == N_SMOKE_SHIFTS
    assert np.isfinite(transitions["rewards"]).all()
    # The reward is NOT sign-constrained, and the earlier assertion that it was
    # non-positive was simply wrong. Phi carries a holding term w_hold * |queue|,
    # so draining the queue lowers the potential and yields a small POSITIVE
    # reward — which is the intended incentive. What must hold is that cost
    # accrues over a shift, so the total is negative and the positive excursions
    # are bounded by the holding term.
    assert transitions["rewards"].sum() < 0.0
    pos = transitions["rewards"][transitions["rewards"] > 0]
    if pos.size:
        assert pos.max() < abs(transitions["rewards"].min())
    assert set(np.unique(transitions["actions"])).issubset(set(range(len(pool))))


def test_behaviour_policy_is_not_a_function_of_the_interval_index(transitions):
    """Reviewer 1 (6.b): coverage must not be degenerate given the state.

    Under the submitted `a = interval_idx % |H|` scheme, every shift logs the
    identical action sequence, so conditional on the interval index — an
    observed feature — only one action is ever seen. Distinct sequences across
    shifts is the observable signature that this has been fixed.
    """
    per_shift = [
        transitions["actions"][transitions["shift_id"] == s]
        for s in np.unique(transitions["shift_id"])
    ]
    assert any(
        not np.array_equal(per_shift[0], other) for other in per_shift[1:]
    ), "every shift logged the same action sequence — coverage is still degenerate"


def test_subset_transitions(transitions):
    sub = subset_transitions(transitions, np.array([0, 2]))
    assert sub["states"].shape[0] == 2 * 32
    assert set(np.unique(sub["shift_id"])) == {0, 2}


def test_train_fqi_returns_usable_model(transitions, fqi_model, pool):
    q = _q_values(fqi_model, transitions["states"][:5], len(pool))
    assert q.shape == (5, len(pool))
    assert np.isfinite(q).all()


def test_policy_returns_valid_heuristic(cfg, transitions, fqi_model, pool):
    policy = _policy(cfg, fqi_model, pool)
    for state in transitions["states"][:20]:
        assert policy(state) in pool
    policy.reset()  # no-op, must not raise


def test_policy_rejects_wrong_shape(cfg, fqi_model, pool):
    policy = _policy(cfg, fqi_model, pool)
    with pytest.raises(ValueError):
        policy(np.zeros(N_FEATURES + 1, dtype=np.float64))


@pytest.mark.skipif(
    MASKED_RULE not in resolve_pool(with_default_scales(OmegaConf.load(CONFIG_PATH))),
    reason="FEFO is not in the deployed pool; the mask is a no-op by design.",
)
def test_fefo_mask_excludes_fefo_below_threshold(pool):
    fefo_idx = pool.index(MASKED_RULE)
    q = np.full((1, len(pool)), -10.0)
    q[0, fefo_idx] = 100.0
    non_perishable = np.zeros((1, N_FEATURES))  # pct_perishable = 0 < threshold
    masked = _mask_fefo(q, non_perishable, threshold=0.05, fefo_idx=fefo_idx)
    assert masked[0, fefo_idx] == -np.inf
    assert int(np.argmax(masked[0])) != fefo_idx
    # With enough perishables in the queue, FEFO stays available.
    perishable = np.zeros((1, N_FEATURES))
    perishable[0, PCT_PERISHABLE_IDX] = 0.5
    kept = _mask_fefo(q, perishable, threshold=0.05, fefo_idx=fefo_idx)
    assert kept[0, fefo_idx] == 100.0


def test_mask_is_a_noop_without_fefo(pool):
    q = np.zeros((1, len(pool)))
    assert np.array_equal(
        _mask_fefo(q, np.zeros((1, N_FEATURES)), threshold=0.05, fefo_idx=None), q
    )


def test_pct_perishable_index_is_resolved_by_name():
    """The submitted module hardcoded 4; revision inserted three expiry features."""
    from simulation.state_extractor import FEATURE_NAMES

    assert FEATURE_NAMES[PCT_PERISHABLE_IDX] == "pct_perishable"


def test_save_load_roundtrip(cfg, fqi_model, transitions, pool, tmp_path):
    threshold = float(cfg.heuristics.fefo_mask_threshold)
    save_offline_fqi(
        tmp_path, fqi_model,
        {"fefo_threshold": threshold, "arms": list(pool),
         "use_regime_features": False},
    )
    assert (tmp_path / "qmodel.joblib").exists()
    loaded = load_offline_fqi(tmp_path, cfg=cfg)
    assert loaded.fefo_threshold == threshold
    assert loaded.arms == pool
    ref = _policy(cfg, fqi_model, pool)
    for state in transitions["states"][:10]:
        assert loaded(state) == ref(state)


def _usable_fqi(run_dir: Path) -> bool:
    """True only for a model trained by the CURRENT pipeline.

    A pre-revision qmodel expects 25 features + 4 action columns = 29 inputs;
    the current state is 26 + 8 = 34, so loading one raises a shape mismatch.
    `arms` is written only by the revised trainer, which makes it the marker.
    """
    import joblib

    if not (run_dir / "qmodel.joblib").exists():
        return False
    meta = run_dir / "meta.joblib"
    if not meta.exists():
        return False
    try:
        return bool(joblib.load(meta).get("arms"))
    except Exception:
        return False


@pytest.mark.skipif(
    not _usable_fqi(REPO_ROOT / "runs" / "offline_fqi"),
    reason="no current-revision FQI model at runs/offline_fqi/ (make stage4-fqi)",
)
def test_offline_fqi_via_evaluate_harness(cfg, tmp_path):
    """The trained baseline plugs into the evaluation harness for one shift."""
    policy = load_offline_fqi(REPO_ROOT / "runs" / "offline_fqi", cfg=cfg)
    seeds = canonical_test_seeds(cfg)[:1]
    df = evaluate_policy(
        "offline_fqi", policy, seeds, cfg, results_dir=tmp_path, save=True
    )
    assert len(df) == 1
    assert 0.0 <= df.iloc[0]["service_failure_rate"] <= 1.0
    assert (tmp_path / "offline_fqi.parquet").exists()


# --- Transition-cache staleness (revision) -----------------------------------
#
# `data/offline_fqi_transitions.npz` caches the logged transition corpus. It
# used to carry no record of which pool, feature set or objective produced it and
# was loaded blindly whenever the file existed; its own docstring said "delete
# the cache after changing it, the pool, or the objective", which makes
# correctness depend on someone remembering.
#
# The committed pre-revision cache holds 25-dimensional states and pre-screening
# action indices. Loading it against the current 26-feature observation and the
# six-rule pool crashed with a bare KeyError — and the crash was the lucky
# outcome. Action indices are POSITIONAL, so had the widths happened to agree,
# fitted Q-iteration would have trained on actions meaning different rules and
# the baseline would have been quietly wrong.


def test_cache_stamp_changes_with_pool_features_policy_and_objective():
    from omegaconf import OmegaConf, open_dict

    from experiments.e9_offline_fqi import _cache_stamp

    cfg = OmegaConf.load(REPO_ROOT / "config.yaml")
    base = list(_cache_stamp(cfg))

    def perturbed(mutate):
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(c):
            mutate(c)
        return list(_cache_stamp(c))

    # Reordering the pool must change the stamp: action indices are positional,
    # so the same rules in a different order are a different action set.
    def reorder(c):
        c.heuristics.pool = list(reversed(list(c.heuristics.pool)))

    variants = {
        "pool order": reorder,
        "pool membership": lambda c: c.heuristics.pool.pop(),
        "behaviour policy": lambda c: c.labeling.__setitem__(
            "observed_policy", "round_robin"),
        "objective weight": lambda c: c.objective.__setitem__("w_spoil", 99.0),
        "priority weighting": lambda c: c.objective.__setitem__(
            "use_priority_weights", False),
    }
    for name, mutate in variants.items():
        assert perturbed(mutate) != base, (
            f"cache stamp is blind to a change in {name}; a stale transition log "
            f"would be reused and the offline-RL baseline would be wrong"
        )


def test_pre_revision_cache_is_rejected(tmp_path, monkeypatch):
    """An unstamped cache must be discarded, not loaded."""
    import numpy as np
    from omegaconf import OmegaConf

    import experiments.e9_offline_fqi as e9

    cache = tmp_path / "transitions.npz"
    # Exactly the committed pre-revision schema: no stamp, 25-D states.
    np.savez(
        cache,
        states=np.zeros((10, 25)), actions=np.zeros(10, dtype=np.int64),
        rewards=np.zeros(10), next_states=np.zeros((10, 25)),
        dones=np.zeros(10, dtype=bool), shift_id=np.zeros(10, dtype=np.int64),
    )
    monkeypatch.setattr(e9, "TRANSITIONS_CACHE", cache)

    built = {}

    def _fake_log(*_a, **_k):
        built["rebuilt"] = True
        return {"states": np.zeros((1, 26)), "actions": np.zeros(1, dtype=np.int64)}

    monkeypatch.setattr(e9, "log_transitions", _fake_log)
    monkeypatch.setattr(e9, "_train_seeds", lambda cfg: [0])

    cfg = OmegaConf.load(REPO_ROOT / "config.yaml")
    e9._load_or_build_transitions(cfg)
    assert built.get("rebuilt"), "the unstamped pre-revision cache was reused"
