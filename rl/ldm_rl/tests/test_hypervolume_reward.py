"""Coverage for the 2D hypervolume-improvement reward."""

import pytest

from ldm_rl.env import REWARD_POLICIES, EnvConfig, LDMEnv
from ldm_rl.episodes import EpisodeSpec


def test_hypervolume_is_a_policy():
    assert "hypervolume" in REWARD_POLICIES
    cfg = EnvConfig(iterations=1, reward="hypervolume", reward_ref_point=(0.0, 5.0))
    assert cfg.reward == "hypervolume"


def test_hypervolume_requires_fixed_ref_point():
    # The moving per-round nadir is disabled (PR #2): a fixed ref is mandatory.
    with pytest.raises(ValueError, match="reward_ref_point"):
        EnvConfig(iterations=1, reward="hypervolume")


def test_hv2d_single_point():
    assert LDMEnv._hypervolume_2d([(2.0, 2.0)], (0.0, 0.0)) == pytest.approx(4.0)


def test_hv2d_two_point_front_with_overlap():
    # rects [0,2]x[0,1] and [0,1]x[0,2] -> union area 3
    assert LDMEnv._hypervolume_2d([(2.0, 1.0), (1.0, 2.0)], (0.0, 0.0)) == pytest.approx(3.0)


def test_hv2d_dominated_point_ignored():
    # (1,1) is dominated by (2,2); HV == HV of (2,2) alone
    assert LDMEnv._hypervolume_2d([(2.0, 2.0), (1.0, 1.0)], (0.0, 0.0)) == pytest.approx(4.0)


def test_hv2d_points_below_ref_excluded():
    assert LDMEnv._hypervolume_2d([(-1.0, 5.0), (5.0, -1.0)], (0.0, 0.0)) == pytest.approx(0.0)


def test_hv_improvement_is_nonnegative_and_monotone():
    # front growing from {(1,1)} to {(1,1),(2,2)} increases HV
    ref = (0.0, 0.0)
    before = LDMEnv._hypervolume_2d([(1.0, 1.0)], ref)
    after = LDMEnv._hypervolume_2d([(1.0, 1.0), (2.0, 2.0)], ref)
    assert after > before
    assert max(0.0, after - before) == pytest.approx(after - before)


def test_ref_point_threads_through_episode_spec_json():
    spec = EpisodeSpec(
        task="small_molecule",
        mode="real",
        reward="hypervolume",
        reward_ref_point=(0.0, 5.0),
        real={"gp_history_file": "/tmp/gp.jsonl"},
    )
    assert tuple(spec.to_env_config().reward_ref_point) == (0.0, 5.0)
    # JSON round-trip preserves the values (tuple -> list is fine downstream)
    rt = EpisodeSpec.from_json(spec.to_json())
    assert tuple(rt.reward_ref_point) == (0.0, 5.0)
