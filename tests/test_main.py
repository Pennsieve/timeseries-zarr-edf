"""Tests for the command line."""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from timeseries_zarr import open_bundle

from tests.conftest import SignalSpec, build_edf
from timeseries_zarr_edf.main import main, run

START = datetime(2021, 3, 4, 5, 6, 7)
PATIENT = "P-17 F 02-MAR-1980 Jane_Doe"

SIGNALS: list[SignalSpec] = [
    ("A", "uV", -100.0, 100.0, -100, 100, 4),
    ("B", "mV", 0.0, 10.0, 0, 1000, 2),
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove writer settings the host environment may carry."""
    for key in (
        "ZARR_WRITER_TIMEZONE",
        "ZARR_WRITER_SUBJECT_FIELDS",
        "ZARR_WRITER_SWEEP_BYTES",
        "ZARR_WRITER_VAR",
        "ZARR_WRITER_INNER_LEN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ZARR_WRITER_TIMEZONE", "UTC")


def digital(n: int) -> list[list[npt.NDArray[np.int16]]]:
    """Return n records for SIGNALS whose samples count up."""
    return [
        [
            np.arange(4, dtype=np.int16) + 4 * k,
            np.arange(2, dtype=np.int16) + 2 * k,
        ]
        for k in range(n)
    ]


def make(tmp_path: Path, n: int = 6, **kwargs: object) -> Path:
    """Write a two-signal EDF+C file of n records."""
    options: dict[str, object] = {
        "signals": SIGNALS,
        "records": digital(n),
        "start": START,
        "patient_field": PATIENT,
    }
    options.update(kwargs)
    return build_edf(tmp_path / "x.edf", **options)  # type: ignore[arg-type]


def recording_json(out: Path) -> dict[str, dict[str, object]]:
    """Return the parsed recording.json of a bundle."""
    return json.loads((out / "recording.json").read_text())  # type: ignore[no-any-return]


def test_main_converts_a_file_to_a_bundle_that_opens(tmp_path: Path) -> None:
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    bundle = open_bundle(out)
    assert sorted(bundle.continuous) == [0, 1]
    assert bundle.events == {}
    a, b = bundle.continuous[0], bundle.continuous[1]
    assert (a.name, a.id, a.unit, a.rate_hz) == ("A", "0", "uV", 4.0)
    assert (b.name, b.id, b.unit, b.rate_hz) == ("B", "1", "uV", 2.0)
    assert a.offset_us == b.offset_us == 0


def test_main_writes_the_physical_values_as_float32(tmp_path: Path) -> None:
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    bundle = open_bundle(out)
    raw_a = np.asarray(bundle.continuous[0].raw[:])
    raw_b = np.asarray(bundle.continuous[1].raw[:])
    assert raw_a.dtype == np.float32
    np.testing.assert_array_equal(raw_a, np.arange(24, dtype=np.float32))
    # 0.01 millivolts per step, stored as microvolts.
    expected_b = (np.arange(12, dtype=np.float64) * 0.01 * 1000).astype(
        np.float32
    )
    np.testing.assert_allclose(raw_b, expected_b, rtol=1e-6)


def test_main_writes_the_header_start_as_the_reference_time(
    tmp_path: Path,
) -> None:
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    recording = recording_json(out)
    assert recording["reference_time"] == "2021-03-04T05:06:07.000000+00:00"
    assert recording["reference_time"] == recording["session"]["start_time"]
    assert recording["source"]["timezone"] == "UTC"


def test_main_places_the_start_in_the_zone_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_TIMEZONE", "America/New_York")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    recording = recording_json(out)
    assert recording["reference_time"] == "2021-03-04T05:06:07.000000-05:00"
    assert recording["reference_time"] == recording["session"]["start_time"]
    assert open_bundle(out).continuous[0].offset_us == 0


def test_main_stores_a_fractional_first_stamp_as_the_channel_offset(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, n=4, record_onsets=[0.37, 1.37, 2.37, 3.37])
    out = tmp_path / "x.zarr"
    assert main([str(path), str(out)]) == 0
    bundle = open_bundle(out)
    assert bundle.continuous[0].offset_us == 370_000
    assert bundle.continuous[1].offset_us == 370_000
    recording = recording_json(out)
    assert recording["reference_time"] == "2021-03-04T05:06:07.000000+00:00"
    assert recording["reference_time"] == recording["session"]["start_time"]
    raw = np.asarray(bundle.continuous[0].raw[:])
    np.testing.assert_array_equal(raw, np.arange(16, dtype=np.float32))


def test_main_stores_a_fractional_first_stamp_of_an_edf_d_file(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [("A", "uV", -100.0, 100.0, -100, 100, 4)]
    records = [[np.arange(4, dtype=np.int16) + 4 * k] for k in range(4)]
    path = make(
        tmp_path,
        signals=signals,
        records=records,
        flavor="EDF+D",
        record_onsets=[0.37, 1.37, 5.37, 6.37],
    )
    out = tmp_path / "x.zarr"
    assert main([str(path), str(out)]) == 0
    channel = open_bundle(out).continuous[0]
    raw = np.asarray(channel.raw[:])
    assert channel.offset_us == 370_000
    assert raw.shape == (28,)
    assert not np.isnan(raw[:8]).any()
    assert np.isnan(raw[8:20]).all()


def test_main_writes_nan_in_the_gap_of_an_edf_d_file(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [("A", "uV", -100.0, 100.0, -100, 100, 4)]
    records = [[np.arange(4, dtype=np.int16) + 4 * k] for k in range(4)]
    path = make(
        tmp_path,
        signals=signals,
        records=records,
        flavor="EDF+D",
        record_onsets=[0.0, 1.0, 5.0, 6.0],
    )
    out = tmp_path / "x.zarr"
    assert main([str(path), str(out)]) == 0
    raw = np.asarray(open_bundle(out).continuous[0].raw[:])
    assert raw.shape == (28,)
    assert np.isnan(raw[8:20]).all()
    np.testing.assert_array_equal(raw[:8], np.arange(8))
    np.testing.assert_array_equal(raw[20:], np.arange(8, 16))


def test_main_writes_the_annotations_as_one_event_channel(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, annotations=[(2.0, 1.5, "wake"), (4.0, None, "rem")])
    out = tmp_path / "x.zarr"
    assert main([str(path), str(out)]) == 0
    bundle = open_bundle(out)
    assert sorted(bundle.events) == [2]
    channel = bundle.events[2]
    assert (channel.id, channel.name) == ("annotations", "annotations")
    table = channel.read()
    assert table.text == ["wake", "rem"]
    assert table.time_us.tolist() == [2_000_000, 4_000_000]
    assert table.duration_us is not None
    assert table.duration_us.tolist() == [1_500_000, 0]
    assert table.label_name == ["wake", "rem"]


def test_main_writes_the_code_and_sex_of_the_subject_by_default(
    tmp_path: Path,
) -> None:
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    assert recording_json(out)["subject"] == {"id": "P-17", "sex": "F"}


def test_main_follows_the_subject_fields_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_SUBJECT_FIELDS", "name,birthdate")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    assert recording_json(out)["subject"] == {
        "name": "Jane_Doe",
        "date_of_birth": "1980-03-02",
    }


def test_main_omits_the_subject_when_no_field_is_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_SUBJECT_FIELDS", "")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    assert "subject" not in recording_json(out)


def test_main_keeps_identity_out_of_every_node_but_recording_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_SUBJECT_FIELDS", "code,sex,birthdate,name")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 0
    for path in out.rglob("zarr.json"):
        text = path.read_text()
        for secret in ("P-17", "Jane_Doe", "1980"):
            assert secret not in text


def _bundle_channels(out: Path) -> dict[str, str]:
    """Return the id of every channel group in a bundle by group path."""
    root = json.loads((out / "zarr.json").read_text())
    return {
        path: node["attributes"]["tsz"]["id"]
        for path, node in root["consolidated_metadata"]["metadata"].items()
        if "/" not in path and "tsz" in node.get("attributes", {})
    }


def test_main_writes_the_same_channels_when_the_sweep_splits_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = tmp_path / "whole.zarr"
    path = make(tmp_path)
    assert main([str(path), str(whole)]) == 0
    monkeypatch.setenv("ZARR_WRITER_SWEEP_BYTES", "1")
    split = tmp_path / "split.zarr"
    assert main([str(path), str(split)]) == 0
    assert (
        _bundle_channels(split)
        == _bundle_channels(whole)
        == {
            "0": "0",
            "1": "1",
        }
    )
    whole_bundle, split_bundle = open_bundle(whole), open_bundle(split)
    for key in (0, 1):
        np.testing.assert_array_equal(
            np.asarray(split_bundle.continuous[key].raw[:]),
            np.asarray(whole_bundle.continuous[key].raw[:]),
        )


def test_main_replaces_a_bundle_from_an_earlier_run(tmp_path: Path) -> None:
    path = make(tmp_path)
    out = tmp_path / "x.zarr"
    assert main([str(path), str(out)]) == 0
    (out / "9").mkdir()
    assert main([str(path), str(out)]) == 0
    assert (out / "0" / "raw" / "zarr.json").exists()
    assert not (out / "9").exists()


def test_main_returns_1_for_an_output_directory_holding_other_files(
    tmp_path: Path,
) -> None:
    out = tmp_path / "x.zarr"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    assert main([str(make(tmp_path)), str(out)]) == 1
    assert (out / "notes.txt").read_text() == "keep"
    assert not (out / "zarr.json").exists()


@pytest.mark.parametrize("argv", [[], ["only.edf"], ["a.edf", "b", "c"]])
def test_main_returns_2_on_a_bad_invocation(argv: list[str]) -> None:
    assert main(argv) == 2


def test_main_returns_2_for_a_setting_outside_its_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_INNER_LEN", "0")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 2
    assert not out.exists()


def test_main_returns_2_for_an_unknown_zone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZARR_WRITER_TIMEZONE", "Mars/Olympus")
    assert main([str(make(tmp_path)), str(tmp_path / "x.zarr")]) == 2


def test_main_returns_2_when_the_zone_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZARR_WRITER_TIMEZONE")
    out = tmp_path / "x.zarr"
    assert main([str(make(tmp_path)), str(out)]) == 2
    assert not out.exists()


def test_main_returns_1_when_the_input_cannot_be_read(tmp_path: Path) -> None:
    out = tmp_path / "out.zarr"
    assert main([str(tmp_path / "missing.edf"), str(out)]) == 1
    assert not out.exists()


def test_main_logs_no_start_date_for_a_malformed_header(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = make(tmp_path)
    data = bytearray(path.read_bytes())
    data[168:176] = b"31.04.23"
    path.write_bytes(bytes(data))
    with caplog.at_level("DEBUG"):
        assert main([str(path), str(tmp_path / "out.zarr")]) == 1
    assert caplog.records
    for record in caplog.records:
        text = record.getMessage()
        if record.exc_info:
            text += "".join(traceback.format_exception(*record.exc_info))
        assert "31.04.23" not in text
        assert "2023" not in text


def test_main_returns_1_for_a_file_that_is_not_an_edf(tmp_path: Path) -> None:
    path = tmp_path / "bad.edf"
    path.write_bytes(b"not an edf file" * 40)
    out = tmp_path / "out.zarr"
    assert main([str(path), str(out)]) == 1
    assert not out.exists()


def test_run_exits_with_the_main_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "x.zarr"
    monkeypatch.setattr(
        sys, "argv", ["timeseries-zarr-edf", str(make(tmp_path)), str(out)]
    )
    with pytest.raises(SystemExit) as exit_info:
        run()
    assert exit_info.value.code == 0
    assert (out / "zarr.json").exists()
