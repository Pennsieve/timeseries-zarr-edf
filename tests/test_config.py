"""Tests for the run configuration."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from timeseries_zarr import MAX_LEVELS, WriteOpts

from timeseries_zarr_edf.config import (
    DEFAULT_SWEEP_BYTES,
    MAX_ZSTD_LEVEL,
    Config,
    load_config,
)

ARGV = ["in.edf", "out_bundle"]
ENV = {"ZARR_WRITER_TIMEZONE": "UTC"}


def test_two_positionals_populate_paths() -> None:
    cfg = load_config(ENV, ["in.edf", "out_bundle"])
    assert isinstance(cfg, Config)
    assert cfg.edf_path == Path("in.edf")
    assert cfg.output_path == Path("out_bundle")


def test_unset_settings_use_writeopts_defaults() -> None:
    cfg = load_config(ENV, ARGV)
    assert cfg.opts == WriteOpts()


def test_env_settings_override_defaults() -> None:
    cfg = load_config(
        {
            **ENV,
            "ZARR_WRITER_ZSTD_LEVEL": "9",
            "ZARR_WRITER_MAX_LEVELS": "4",
            "ZARR_WRITER_MIN_BINS": "512",
            "ZARR_WRITER_INNER_LEN": "1024",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "2048",
            "ZARR_WRITER_VAR": "1",
        },
        ARGV,
    )
    assert cfg.opts == WriteOpts(
        zstd_level=9,
        max_levels=4,
        min_bins=512,
        inner_len=1024,
        target_shard_bytes=2048,
        write_var=True,
    )


@pytest.mark.parametrize(
    ("key", "value", "valid"),
    [
        ("ZARR_WRITER_ZSTD_LEVEL", -1, f"0 to {MAX_ZSTD_LEVEL}"),
        (
            "ZARR_WRITER_ZSTD_LEVEL",
            MAX_ZSTD_LEVEL + 1,
            f"0 to {MAX_ZSTD_LEVEL}",
        ),
        ("ZARR_WRITER_MAX_LEVELS", -1, f"0 to {MAX_LEVELS}"),
        ("ZARR_WRITER_MAX_LEVELS", MAX_LEVELS + 1, f"0 to {MAX_LEVELS}"),
        ("ZARR_WRITER_MIN_BINS", 0, "1 or more"),
        ("ZARR_WRITER_INNER_LEN", 0, "1 or more"),
        ("ZARR_WRITER_INNER_LEN", -1, "1 or more"),
        ("ZARR_WRITER_TARGET_SHARD_BYTES", 0, "1 or more"),
    ],
)
def test_a_setting_outside_its_range_raises(
    key: str, value: int, valid: str
) -> None:
    with pytest.raises(
        ValueError, match=f"^{key} must be {valid}, got {value}$"
    ):
        load_config({**ENV, key: str(value)}, ARGV)


@pytest.mark.parametrize(
    ("key", "value", "field"),
    [
        ("ZARR_WRITER_ZSTD_LEVEL", 0, "zstd_level"),
        ("ZARR_WRITER_ZSTD_LEVEL", MAX_ZSTD_LEVEL, "zstd_level"),
        ("ZARR_WRITER_MAX_LEVELS", 0, "max_levels"),
        ("ZARR_WRITER_MAX_LEVELS", MAX_LEVELS, "max_levels"),
        ("ZARR_WRITER_MIN_BINS", 1, "min_bins"),
        ("ZARR_WRITER_INNER_LEN", 1, "inner_len"),
        ("ZARR_WRITER_TARGET_SHARD_BYTES", 1, "target_shard_bytes"),
    ],
)
def test_accepts_a_setting_at_an_end_of_its_range(
    key: str, value: int, field: str
) -> None:
    cfg = load_config({**ENV, key: str(value)}, ARGV)
    assert getattr(cfg.opts, field) == value


@pytest.mark.parametrize(
    "argv", [[], ["in.edf"], ["in.edf", "out_bundle", "extra"]]
)
def test_a_wrong_number_of_positionals_raises(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="two arguments"):
        load_config(ENV, argv)


@pytest.mark.parametrize("value", ["high", "", "1.5"])
def test_non_integer_setting_raises(value: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"^ZARR_WRITER_INNER_LEN must be an integer, got {value!r}$",
    ):
        load_config({**ENV, "ZARR_WRITER_INNER_LEN": value}, ARGV)


def test_var_is_off_when_the_variable_is_unset() -> None:
    assert load_config(ENV, ARGV).opts.write_var is False


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
def test_var_follows_a_zero_or_one_flag(value: str, expected: bool) -> None:
    cfg = load_config({**ENV, "ZARR_WRITER_VAR": value}, ARGV)
    assert cfg.opts.write_var is expected


@pytest.mark.parametrize("value", ["", "true", "yes", "2", " 1", "01"])
def test_var_rejects_any_value_but_zero_or_one(value: str) -> None:
    with pytest.raises(ValueError, match="ZARR_WRITER_VAR must be 0 or 1"):
        load_config({**ENV, "ZARR_WRITER_VAR": value}, ARGV)


def test_sweep_bytes_defaults_to_the_default_sweep_budget() -> None:
    assert load_config(ENV, ARGV).sweep_bytes == DEFAULT_SWEEP_BYTES


def test_sweep_bytes_follows_its_variable() -> None:
    cfg = load_config({**ENV, "ZARR_WRITER_SWEEP_BYTES": "4096"}, ARGV)
    assert cfg.sweep_bytes == 4096


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "lots"])
def test_sweep_bytes_rejects_a_value_that_is_not_a_positive_integer(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="ZARR_WRITER_SWEEP_BYTES"):
        load_config({**ENV, "ZARR_WRITER_SWEEP_BYTES": value}, ARGV)


def test_raw_shard_bytes_is_the_whole_chunks_that_fit_in_the_target() -> None:
    cfg = load_config(
        {
            **ENV,
            "ZARR_WRITER_INNER_LEN": "100",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "3000",
        },
        ARGV,
    )
    assert cfg.raw_shard_bytes == 7 * 100 * 4


def test_raw_shard_bytes_is_one_chunk_when_a_chunk_exceeds_the_target() -> None:
    cfg = load_config(
        {
            **ENV,
            "ZARR_WRITER_INNER_LEN": "16777216",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "4096",
        },
        ARGV,
    )
    assert cfg.raw_shard_bytes == 16777216 * 4


def test_sweep_channels_is_the_raw_shards_that_fit_in_the_sweep_budget() -> (
    None
):
    cfg = load_config(
        {
            **ENV,
            "ZARR_WRITER_SWEEP_BYTES": "10000",
            "ZARR_WRITER_INNER_LEN": "100",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "3000",
        },
        ARGV,
    )
    assert cfg.sweep_channels == 3


def test_sweep_channels_counts_a_chunk_larger_than_the_target_shard() -> None:
    cfg = load_config({**ENV, "ZARR_WRITER_INNER_LEN": "16777216"}, ARGV)
    assert cfg.sweep_channels == DEFAULT_SWEEP_BYTES // (16777216 * 4)


def test_sweep_channels_is_one_when_the_budget_is_below_one_shard() -> None:
    cfg = load_config({**ENV, "ZARR_WRITER_SWEEP_BYTES": "100"}, ARGV)
    assert cfg.sweep_channels == 1


def test_the_default_sweep_holds_128_channels_of_default_shards() -> None:
    assert load_config(ENV, ARGV).sweep_channels == 128


def test_an_unset_timezone_raises_naming_the_variable() -> None:
    with pytest.raises(
        ValueError,
        match="^ZARR_WRITER_TIMEZONE must be set to an IANA zone name such as "
        "UTC or America/New_York$",
    ):
        load_config({}, ARGV)


def test_timezone_follows_a_valid_zone_name() -> None:
    env = {"ZARR_WRITER_TIMEZONE": "America/New_York"}
    assert load_config(env, ARGV).timezone == ZoneInfo("America/New_York")


@pytest.mark.parametrize("value", ["Mars/Olympus", "", "../etc/passwd"])
def test_an_invalid_zone_raises_naming_the_variable(value: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"^ZARR_WRITER_TIMEZONE must be an IANA zone name, got {value!r}$",
    ):
        load_config({"ZARR_WRITER_TIMEZONE": value}, ARGV)


def test_subject_fields_default_to_code_and_sex() -> None:
    cfg = load_config(ENV, ARGV)
    assert cfg.subject_fields == frozenset({"code", "sex"})


def test_an_empty_subject_fields_value_chooses_no_field() -> None:
    cfg = load_config({**ENV, "ZARR_WRITER_SUBJECT_FIELDS": ""}, ARGV)
    assert cfg.subject_fields == frozenset()


def test_subject_fields_take_a_list_with_spaces() -> None:
    env = {**ENV, "ZARR_WRITER_SUBJECT_FIELDS": " code , birthdate,name "}
    cfg = load_config(env, ARGV)
    assert cfg.subject_fields == frozenset({"code", "birthdate", "name"})


def test_an_unknown_subject_field_raises_naming_it_and_the_valid_names() -> (
    None
):
    env = {**ENV, "ZARR_WRITER_SUBJECT_FIELDS": "code,address"}
    with pytest.raises(ValueError) as error:
        load_config(env, ARGV)
    message = str(error.value)
    assert "ZARR_WRITER_SUBJECT_FIELDS" in message
    assert "'address'" in message
    assert "code, sex, birthdate, name" in message
