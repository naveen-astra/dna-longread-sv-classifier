"""Config validation guards.

These tests protect the architecture's correctness rules at the configuration level,
so a well-meaning edit to config.yaml cannot silently disable them.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from svml.utils.config import ConfigError, load_config

CONFIG_PATH = "config/config.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


def _reload(tmp_path, raw, root):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path, root=root)


def test_shipped_config_is_valid(cfg):
    assert cfg.digest
    assert cfg.seed == 42
    assert cfg.labelled_samples()


def test_splits_are_disjoint_in_shipped_config(cfg):
    mapping = cfg.split_map()
    assert mapping["chr1"] == "train"
    assert mapping["chr20"] == "test"
    assert len(mapping) == len(set(mapping))


def test_pilot_regions_reach_every_split(cfg):
    """A chromosome-grouped split needs labelled regions on train, val and test chroms."""
    split_map = cfg.split_map()
    covered = {split_map[chrom] for chrom, _, _ in cfg.active_windows("labelled")}
    assert covered == {"train", "val", "test"}


def test_single_chromosome_pilot_is_rejected(cfg, tmp_path):
    """A chr19-only pilot leaves train and test empty; the config must refuse it."""
    raw = copy.deepcopy(cfg.raw)
    raw["regions"]["tier1"] = {"labelled": ["chr19"], "generalisation": ["chr19"],
                               "unlabelled": ["chr19"]}
    with pytest.raises(ConfigError, match="chromosome-grouped"):
        _reload(tmp_path, raw, cfg.root)


def test_generalisation_sample_uses_only_test_windows(cfg):
    split_map = cfg.split_map()
    assert {split_map[c] for c, _, _ in cfg.active_windows("generalisation")} == {"test"}


def test_label_and_eval_matchers_must_agree(cfg, tmp_path):
    raw = copy.deepcopy(cfg.raw)
    raw["evaluation"]["truvari"] = dict(raw["evaluation"]["truvari"])
    raw["evaluation"]["truvari"]["refdist"] = 9999

    with pytest.raises(ConfigError, match="must be identical"):
        _reload(tmp_path, raw, cfg.root)


def test_confident_region_gate_cannot_be_disabled(cfg, tmp_path):
    """Finding H4: disabling the gate poisons the negative class."""
    raw = copy.deepcopy(cfg.raw)
    raw["labels"]["require_inside_confident_regions"] = False

    with pytest.raises(ConfigError, match="require_inside_confident_regions"):
        _reload(tmp_path, raw, cfg.root)


def test_stage_c_cannot_be_marked_benchmarked(cfg, tmp_path):
    """Finding B1: no GIAB truth set labels DUP/INV/TRA."""
    raw = copy.deepcopy(cfg.raw)
    raw["ml"]["stages"]["C"]["benchmarked"] = True

    with pytest.raises(ConfigError, match="pseudo-labels"):
        _reload(tmp_path, raw, cfg.root)


def test_stage_b_must_exclude_the_leaky_type_feature(cfg, tmp_path):
    """Finding H1."""
    raw = copy.deepcopy(cfg.raw)
    raw["ml"]["stages"]["B"]["exclude_features"] = []

    with pytest.raises(ConfigError, match="identity function"):
        _reload(tmp_path, raw, cfg.root)


@pytest.mark.parametrize("dropped", ["union", "intersection"])
def test_baselines_must_include_union_and_intersection(cfg, tmp_path, dropped):
    """Finding H2: without these the ML layer is not compared to what it must beat."""
    raw = copy.deepcopy(cfg.raw)
    raw["evaluation"]["baselines"] = [b for b in raw["evaluation"]["baselines"] if b != dropped]

    with pytest.raises(ConfigError, match=dropped):
        _reload(tmp_path, raw, cfg.root)


def test_sample_manifest_requires_truth_for_labelled_roles(cfg, tmp_path):
    labelled = cfg.samples[cfg.samples["role"] == "labelled"]
    assert (labelled["truth"] != "none").all()
    assert (labelled["truth"] != "").all()


def test_every_declared_truth_exists_in_references(cfg):
    known = set(cfg.references["truth"]) | {"none", ""}
    assert set(cfg.samples["truth"]) <= known


def test_parse_region_and_windows():
    from svml.utils.config import in_windows, parse_region

    assert parse_region("chr16:1-15000000") == ("chr16", 0, 15000000)
    assert parse_region("chr19")[0] == "chr19"
    windows = [parse_region("chr16:1000001-2000000")]
    assert in_windows("chr16", 1500000, windows)
    assert not in_windows("chr16", 1005000, windows, margin=20000)   # too close to edge
    assert not in_windows("chr17", 1500000, windows)
    with pytest.raises(ConfigError):
        parse_region("chr1:500-100")
