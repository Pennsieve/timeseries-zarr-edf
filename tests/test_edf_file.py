"""Tests for the EDF record reader."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import (
    SignalSpec,
    annotation_samples,
    build_edf,
    encode_tal,
)
from timeseries_zarr_edf.edf_file import (
    ONSET_TOLERANCE_S,
    EdfAnnotation,
    EdfFile,
    Segment,
    Tal,
    parse_tals,
)
from timeseries_zarr_edf.edf_header import ANNOTATION_LABEL, EdfFormatError

START = datetime(2021, 3, 4, 5, 6, 7)

SIGNALS: list[SignalSpec] = [
    ("A", "uV", -100.0, 100.0, -100, 100, 4),
    ("B", "uV", 0.0, 10.0, 0, 1000, 2),
]


def records_for(n: int) -> list[list[npt.NDArray[np.int16]]]:
    """Return n records whose samples count up across the whole file."""
    out = []
    for k in range(n):
        a = np.arange(4, dtype=np.int16) + 4 * k
        b = np.arange(2, dtype=np.int16) + 100 * k
        out.append([a, b])
    return out


def make(tmp_path: Path, n: int = 4, **kwargs: object) -> Path:
    """Write a two-signal file of n records."""
    options: dict[str, object] = {
        "signals": SIGNALS,
        "records": records_for(n),
        "start": START,
    }
    options.update(kwargs)
    return build_edf(tmp_path / "x.edf", **options)  # type: ignore[arg-type]


def test_open_maps_the_declared_records(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        assert edf.n_records == 4
        assert edf.num_samples(0) == 16
        assert edf.num_samples(1) == 8
        assert edf.header.flavor == "EDF+C"


def test_open_derives_the_count_from_the_file_size(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path, n_records_field=-1)) as edf:
        assert edf.n_records == 4
        assert edf.header.n_records == -1


def test_open_rejects_a_truncated_file(tmp_path: Path) -> None:
    with pytest.raises(EdfFormatError, match="truncated"):
        EdfFile.open(make(tmp_path, n_records_field=9))


def test_open_warns_about_trailing_bytes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = make(tmp_path, n_records_field=3)
    with caplog.at_level(logging.WARNING), EdfFile.open(path) as edf:
        assert edf.n_records == 3
    assert "follow the last complete record" in caplog.text


def test_open_accepts_a_file_with_no_records(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path, n=0)) as edf:
        assert edf.n_records == 0
        assert edf.segments() == []
        assert edf.read_records(0, 0, 0).size == 0


def test_read_records_spans_a_record_boundary(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        np.testing.assert_array_equal(
            edf.read_records(0, 1, 3), np.arange(4, 12)
        )
        np.testing.assert_array_equal(
            edf.read_records(1, 0, 4), [0, 1, 100, 101, 200, 201, 300, 301]
        )
        assert edf.read_records(0, 2, 2).size == 0


def test_read_records_rejects_a_bad_range_or_signal(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        with pytest.raises(ValueError, match="record range"):
            edf.read_records(0, 2, 9)
        with pytest.raises(ValueError, match="record range"):
            edf.read_records(0, 3, 2)
        with pytest.raises(ValueError, match="signal index"):
            edf.read_records(7, 0, 1)


def test_read_window_holds_one_row_per_record(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        window = edf.read_window(1, 3)
        assert window.shape == (2, edf.header.record_samples)
        np.testing.assert_array_equal(
            window[:, 0:4], [[4, 5, 6, 7], [8, 9, 10, 11]]
        )
        np.testing.assert_array_equal(window[:, 4:6], [[100, 101], [200, 201]])
        assert edf.read_window(2, 2).shape == (0, edf.header.record_samples)


def test_signal_physical_of_a_window_equals_read_physical(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        window = edf.read_window(0, 4)
        for signal_index in (0, 1):
            np.testing.assert_array_equal(
                edf.signal_physical(window, signal_index),
                edf.read_physical(signal_index, 0, 4),
            )
            np.testing.assert_array_equal(
                edf.signal_digital(window, signal_index),
                edf.read_records(signal_index, 0, 4),
            )
        with pytest.raises(ValueError, match="signal index"):
            edf.signal_physical(window, 7)


def test_a_read_raises_when_the_file_shrinks_after_open(
    tmp_path: Path,
) -> None:
    path = make(tmp_path)
    with EdfFile.open(path) as edf:
        with path.open("r+b") as handle:
            handle.truncate(edf.header.header_bytes + edf.header.record_bytes)
        with pytest.raises(EdfFormatError, match="ends inside a data record"):
            edf.read_window(0, 4)


def test_read_physical_applies_gain_and_offset_in_float32(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        values = edf.read_physical(1, 0, 2)
        assert values.dtype == np.float32
        np.testing.assert_allclose(
            values, np.array([0, 1, 100, 101]) * 0.01, rtol=1e-6
        )


def test_read_physical_handles_an_inverted_range(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [("A", "uV", 100.0, -100.0, -10, 10, 3)]
    records = [[np.array([-10, 0, 10], dtype=np.int16)]]
    with EdfFile.open(make(tmp_path, signals=signals, records=records)) as edf:
        np.testing.assert_allclose(edf.read_physical(0, 0, 1), [100, 0, -100])


def test_read_physical_handles_a_zero_based_digital_range(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [("A", "%", 0.0, 100.0, 0, 127, 3)]
    records = [[np.array([0, 127, 64], dtype=np.int16)]]
    with EdfFile.open(make(tmp_path, signals=signals, records=records)) as edf:
        np.testing.assert_allclose(
            edf.read_physical(0, 0, 1), [0, 100, 64 * 100 / 127], rtol=1e-6
        )


def test_read_physical_handles_an_offset_digital_range(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [("A", "%", 0.0, 100.0, 327, 1200, 2)]
    records = [[np.array([327, 1200], dtype=np.int16)]]
    with EdfFile.open(make(tmp_path, signals=signals, records=records)) as edf:
        np.testing.assert_allclose(
            edf.read_physical(0, 0, 1), [0, 100], atol=1e-4
        )


def test_record_onsets_are_multiples_of_the_duration_for_edf_c(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, record_duration=0.5)
    with EdfFile.open(path) as edf:
        np.testing.assert_allclose(edf.record_onsets(), [0, 0.5, 1.0, 1.5])
        assert edf.record_onsets() is edf.record_onsets()


def test_record_onsets_of_edf_c_start_at_the_first_stamp(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, record_onsets=[0.25, 9.0, 9.0, 9.0])
    with EdfFile.open(path) as edf:
        np.testing.assert_allclose(
            edf.record_onsets(), [0.25, 1.25, 2.25, 3.25]
        )
        assert edf.segments() == [Segment(0, 4, 0.25)]
        assert edf.first_onset_s() == 0.25


def test_record_onsets_of_edf_c_without_a_time_keeping_list_raise(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [
        (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, 8)
    ]
    records = [[annotation_samples(b"", 8)]]
    path = make(tmp_path, signals=signals, records=records)
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="no time-keeping"),
    ):
        edf.record_onsets()


def test_record_onsets_of_an_edf_c_header_without_annotations_start_at_0(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, flavor="EDF")
    data = bytearray(path.read_bytes())
    data[192:196] = b"EDF+"
    data[196] = ord("C")
    path.write_bytes(bytes(data))
    with EdfFile.open(path) as edf:
        np.testing.assert_allclose(edf.record_onsets(), [0, 1, 2, 3])


def test_first_onset_is_0_for_plain_edf_and_an_empty_file(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path, flavor="EDF")) as edf:
        assert edf.first_onset_s() == 0.0
    with EdfFile.open(make(tmp_path, n=0)) as edf:
        assert edf.first_onset_s() == 0.0


def test_record_onsets_read_the_time_keeping_lists_for_edf_d(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, flavor="EDF+D", record_onsets=[0.0, 1.0, 5.0, 6.0])
    with EdfFile.open(path) as edf:
        np.testing.assert_allclose(edf.record_onsets(), [0, 1, 5, 6])


def test_record_onsets_reject_edf_d_without_an_annotation_signal(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, flavor="EDF")
    data = bytearray(path.read_bytes())
    data[192:196] = b"EDF+"
    data[196] = ord("D")
    path.write_bytes(bytes(data))
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="no annotation signal"),
    ):
        edf.record_onsets()


def test_record_onsets_reject_a_record_without_a_time_keeping_list(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [
        (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, 8)
    ]
    records = [
        [annotation_samples(encode_tal(0.0, None, [""]), 8)],
        [annotation_samples(b"", 8)],
    ]
    path = make(tmp_path, signals=signals, records=records, flavor="EDF+D")
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="no time-keeping"),
    ):
        edf.record_onsets()


def test_segments_is_one_segment_for_edf_c(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        assert edf.segments() == [Segment(0, 4, 0.0)]


def test_segments_split_at_a_gap(tmp_path: Path) -> None:
    path = make(tmp_path, flavor="EDF+D", record_onsets=[0.0, 1.0, 5.0, 6.0])
    with EdfFile.open(path) as edf:
        assert edf.segments() == [Segment(0, 2, 0.0), Segment(2, 2, 5.0)]


def test_segments_tolerate_float_noise_in_a_step(tmp_path: Path) -> None:
    jitter = ONSET_TOLERANCE_S / 10
    path = make(
        tmp_path, flavor="EDF+D", record_onsets=[0.0, 1.0 + jitter, 2.0, 3.0]
    )
    with EdfFile.open(path) as edf:
        assert edf.segments() == [Segment(0, 4, 0.0)]


@pytest.mark.parametrize("second", [0.5, 0.0, -1.0])
def test_segments_reject_a_step_shorter_than_a_record(
    tmp_path: Path, second: float
) -> None:
    path = make(tmp_path, n=3, flavor="EDF+D", record_onsets=[0.0, second, 2.0])
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="before the previous record ends"),
    ):
        edf.segments()


def test_segments_reject_a_negative_first_onset(tmp_path: Path) -> None:
    path = make(tmp_path, n=2, flavor="EDF+D", record_onsets=[-1.0, 0.0])
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="negative"),
    ):
        edf.segments()


def test_parse_tals_reads_several_lists_in_one_buffer() -> None:
    raw = encode_tal(0.0, None, [""]) + encode_tal(1.5, 2.0, ["Alpha"])
    raw += encode_tal(-3.0, None, ["Beta"]) + b"\x00\x00"
    assert parse_tals(raw, where="x") == [
        Tal(0.0, None, ("",)),
        Tal(1.5, 2.0, ("Alpha",)),
        Tal(-3.0, None, ("Beta",)),
    ]


def test_parse_tals_reads_several_texts_in_one_list() -> None:
    assert parse_tals(encode_tal(2.0, None, ["a", "b"]), where="x") == [
        Tal(2.0, None, ("a", "b"))
    ]


def test_parse_tals_decodes_utf8() -> None:
    tals = parse_tals(encode_tal(2.0, None, ["caf\u00e9"]), where="x")
    assert tals[0].texts == ("caf\u00e9",)


def test_parse_tals_of_padding_is_empty() -> None:
    assert parse_tals(b"\x00" * 10, where="x") == []
    assert parse_tals(b"", where="x") == []


def test_parse_tals_rejects_a_missing_text_separator() -> None:
    with pytest.raises(EdfFormatError, match="text separator"):
        parse_tals(b"+1.0\x00", where="x")


def test_parse_tals_rejects_a_non_numeric_onset() -> None:
    with pytest.raises(EdfFormatError, match="onset"):
        parse_tals(b"+abc\x14t\x14\x00", where="x")
    with pytest.raises(EdfFormatError, match="onset"):
        parse_tals(b"1.0\x14t\x14\x00", where="x")


def test_parse_tals_rejects_a_non_numeric_duration() -> None:
    with pytest.raises(EdfFormatError, match="duration"):
        parse_tals(b"+1.0\x15long\x14t\x14\x00", where="x")


def test_parse_tals_rejects_an_oversized_list() -> None:
    with pytest.raises(EdfFormatError, match="exceeds"):
        parse_tals(b"+1\x14" + b"a" * 70000 + b"\x14\x00", where="x")


def test_annotations_skip_the_time_keeping_list_and_empty_texts(
    tmp_path: Path,
) -> None:
    path = make(
        tmp_path,
        annotations=[
            (2.5, 1.0, "second"),
            (0.25, None, "first"),
            (2.75, None, ""),
        ],
    )
    with EdfFile.open(path) as edf:
        assert edf.annotations() == [
            EdfAnnotation(0.25, None, "first"),
            EdfAnnotation(2.5, 1.0, "second"),
        ]


def test_annotations_keep_file_order_not_onset_order(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [
        (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, 16)
    ]
    records = [
        [
            annotation_samples(
                encode_tal(0.0, None, [""])
                + encode_tal(0.9, None, ["late"])
                + encode_tal(0.1, None, ["early"]),
                16,
            )
        ]
    ]
    path = make(tmp_path, signals=signals, records=records)
    with EdfFile.open(path) as edf:
        assert [a.text for a in edf.annotations()] == ["late", "early"]


def test_annotations_read_a_second_annotation_signal(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [
        (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, 16)
    ]
    records = [
        [annotation_samples(encode_tal(0.5, None, ["extra"]), 16)],
        [annotation_samples(encode_tal(1.5, 0.25, ["more"]), 16)],
    ]
    path = make(
        tmp_path,
        signals=signals,
        records=records,
        annotations=[(1.2, None, "main")],
    )
    with EdfFile.open(path) as edf:
        assert edf.header.annotation_indices == (0, 1)
        assert edf.annotations() == [
            EdfAnnotation(0.5, None, "extra"),
            EdfAnnotation(1.5, 0.25, "more"),
            EdfAnnotation(1.2, None, "main"),
        ]


def test_annotations_are_empty_for_plain_edf(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path, flavor="EDF")) as edf:
        assert edf.annotations() == []


def test_close_is_idempotent_and_a_read_after_close_raises(
    tmp_path: Path,
) -> None:
    edf = EdfFile.open(make(tmp_path))
    edf.close()
    edf.close()
    with pytest.raises(ValueError, match="closed"):
        edf.read_records(0, 0, 1)
    with pytest.raises(ValueError, match="closed"):
        edf.read_physical(0, 0, 1)


def test_context_manager_closes_the_file(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        pass
    with pytest.raises(ValueError, match="closed"):
        edf.read_records(0, 0, 1)
