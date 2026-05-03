"""Tests de la config ML persistante."""
import json

import pytest

from stocks.ml import config as cfg_mod


def test_default_config_uses_5_50_95():
    cfg = cfg_mod.MLConfig()
    assert cfg.quantiles == (0.05, 0.5, 0.95)
    assert cfg.mape_threshold == 0.45
    assert cfg.coverage_target == 0.80
    assert cfg.coverage_tolerance == 0.10
    assert cfg.tuned_params == cfg_mod.DEFAULT_HGB_PARAMS


def test_quantile_fractions_property():
    cfg = cfg_mod.MLConfig(quantiles=(0.1, 0.5, 0.9))
    assert cfg.quantile_fractions == (0.1, 0.9)


def test_quantiles_must_have_3_values():
    with pytest.raises(ValueError, match="3 valeurs"):
        cfg_mod.MLConfig(quantiles=(0.1, 0.9))


def test_median_quantile_must_be_05():
    with pytest.raises(ValueError, match="median"):
        cfg_mod.MLConfig(quantiles=(0.1, 0.4, 0.9))


def test_quantiles_are_sorted():
    cfg = cfg_mod.MLConfig(quantiles=(0.95, 0.05, 0.5))
    assert cfg.quantiles == (0.05, 0.5, 0.95)


def test_load_config_returns_defaults_if_missing(tmp_path):
    cfg = cfg_mod.load_config(tmp_path / "absent.json")
    assert cfg.quantiles == (0.05, 0.5, 0.95)


def test_save_then_load_roundtrip(tmp_path):
    target = tmp_path / "config.json"
    original = cfg_mod.MLConfig(
        quantiles=(0.1, 0.5, 0.9),
        mape_threshold=0.30,
        tuned_params={"max_iter": 500, "max_depth": 4},
        tuned_at="2026-05-03T10:00:00+00:00",
        tuning_score=0.123,
    )
    cfg_mod.save_config(original, target)
    assert target.exists()

    loaded = cfg_mod.load_config(target)
    assert loaded.quantiles == (0.1, 0.5, 0.9)
    assert loaded.mape_threshold == 0.30
    assert loaded.tuned_params["max_iter"] == 500
    assert loaded.tuned_params["max_depth"] == 4
    assert loaded.tuned_at == "2026-05-03T10:00:00+00:00"
    assert loaded.tuning_score == pytest.approx(0.123)


def test_load_config_handles_corrupt_json(tmp_path, caplog):
    target = tmp_path / "broken.json"
    target.write_text("{not valid json")
    with caplog.at_level("WARNING"):
        cfg = cfg_mod.load_config(target)
    assert cfg.quantiles == (0.05, 0.5, 0.95)


def test_load_config_merges_partial_tuned_params(tmp_path):
    target = tmp_path / "partial.json"
    target.write_text(json.dumps({
        "quantiles": [0.05, 0.5, 0.95],
        "tuned_params": {"max_iter": 500},
    }))
    cfg = cfg_mod.load_config(target)
    # Les autres params HGB doivent etre les defauts.
    assert cfg.tuned_params["max_iter"] == 500
    assert cfg.tuned_params["max_depth"] == cfg_mod.DEFAULT_HGB_PARAMS["max_depth"]


def test_as_dict_serializable():
    cfg = cfg_mod.MLConfig()
    out = cfg.as_dict()
    assert isinstance(out, dict)
    assert isinstance(out["quantiles"], list)
    json.dumps(out)  # ne doit pas lever
