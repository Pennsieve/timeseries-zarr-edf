"""Tests for the EDF channel sources and recording metadata."""

import itertools
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pytest
import timeseries_zarr
from timeseries_zarr import (
    ContinuousChannelMetadata,
    EventChannelSource,
    MultiChannelSource,
)

from tests.conftest import SignalSpec, build_edf
from timeseries_zarr_edf import edf_reader
from timeseries_zarr_edf.edf_events import EdfAnnotationSource
from timeseries_zarr_edf.edf_file import ONSET_TOLERANCE_S, EdfFile
from timeseries_zarr_edf.edf_header import EdfFormatError
from timeseries_zarr_edf.edf_reader import (
    EdfChannel,
    EdfRecordSource,
    build_sources_from_edf,
    channel_layout,
    recording_meta,
    start_us,
)

START = datetime(2021, 3, 4, 5, 6, 7)
UTC_ZONE = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
ALL_FIELDS = frozenset({"code", "sex", "birthdate", "name"})

SIGNALS: list[SignalSpec] = [
    ("A", "uV", -100.0, 100.0, -100, 100, 4),
    ("B", "mV", 0.0, 10.0, 0, 1000, 2),
]


def records_for(n: int) -> list[list[npt.NDArray[np.int16]]]:
    """Return n records whose samples count up across the whole file."""
    return [
        [
            np.arange(4, dtype=np.int16) + 4 * k,
            np.arange(2, dtype=np.int16) + 100 * k,
        ]
        for k in range(n)
    ]


def make(tmp_path: Path, n: int = 5, **kwargs: object) -> Path:
    """Write a two-signal file of n records."""
    options: dict[str, object] = {
        "signals": SIGNALS,
        "records": records_for(n),
        "start": START,
    }
    options.update(kwargs)
    return build_edf(tmp_path / "x.edf", **options)  # type: ignore[arg-type]


def make_gapped(tmp_path: Path) -> Path:
    """Write an EDF+D file whose records start at 0, 1, 5, and 6 seconds."""
    return make(
        tmp_path, n=4, flavor="EDF+D", record_onsets=[0.0, 1.0, 5.0, 6.0]
    )


def build(
    edf: EdfFile,
    *,
    timezone: ZoneInfo = UTC_ZONE,
    subject_fields: frozenset[str] = frozenset(),
    max_sweep_channels: int | None = None,
) -> edf_reader.EdfSources:
    """Build sources with test defaults."""
    return build_sources_from_edf(
        edf,
        timezone=timezone,
        subject_fields=subject_fields,
        max_sweep_channels=max_sweep_channels,
    )


def channel_list(sources: edf_reader.EdfSources) -> list[EdfChannel]:
    """Return every continuous channel in channel order."""
    return [c for source in sources.continuous for c in source.channels()]


def collect(source: EdfRecordSource) -> list[npt.NDArray[np.float32]]:
    """Return each channel's samples concatenated across the windows."""
    columns: list[list[npt.NDArray[np.float32]]] = [
        [] for _ in source.channels()
    ]
    for window in source.iter_windows():
        for column, block in zip(columns, window, strict=True):
            column.append(block)
    return [np.concatenate(column) for column in columns]


def test_channels_are_the_data_signals_in_storage_order(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        channels = channel_list(build(edf))
    assert [(c.id, c.name, c.signal_index) for c in channels] == [
        ("0", "A", 0),
        ("1", "B", 1),
    ]


def test_an_empty_label_names_the_channel_by_its_id(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [("", "uV", -1.0, 1.0, -1, 1, 2)]
    records = [[np.zeros(2, dtype=np.int16)]]
    path = make(tmp_path, signals=signals, records=records)
    with EdfFile.open(path) as edf:
        (channel,) = channel_list(build(edf))
    assert channel.name == channel.id == "0"


def test_units_are_folded(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [
        ("A", "\u00b5V", -1.0, 1.0, -1, 1, 2),
        ("B", "%", -1.0, 1.0, -1, 1, 2),
        ("C", "", -1.0, 1.0, -1, 1, 2),
    ]
    records = [[np.zeros(2, dtype=np.int16) for _ in signals]]
    path = make(tmp_path, signals=signals, records=records)
    with EdfFile.open(path) as edf:
        channels = channel_list(build(edf))
    assert [c.unit for c in channels] == ["uV", "%", ""]


def test_rate_start_and_length_follow_the_header(tmp_path: Path) -> None:
    path = make(tmp_path, record_duration=0.5)
    with EdfFile.open(path) as edf:
        a, b = channel_list(build(edf, timezone=NEW_YORK))
    expected = int(
        datetime(2021, 3, 4, 5, 6, 7, tzinfo=NEW_YORK).timestamp() * 1_000_000
    )
    assert (a.rate_hz(), b.rate_hz()) == (8.0, 4.0)
    assert a.start_us() == b.start_us() == expected
    assert (a.num_samples(), b.num_samples()) == (20, 10)


def test_start_us_places_the_wall_clock_in_the_zone(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        header = edf.header
    utc = int(datetime(2021, 3, 4, 5, 6, 7, tzinfo=UTC).timestamp()) * 10**6
    assert start_us(header, UTC_ZONE) == utc
    assert start_us(header, NEW_YORK) == utc + 5 * 3600 * 10**6


def test_an_ambiguous_wall_clock_takes_the_first_occurrence(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, start=datetime(2021, 11, 7, 1, 30, 0))
    with EdfFile.open(path) as edf:
        sources = build(edf, timezone=NEW_YORK)
    started = sources.recording.session_start_time
    assert started is not None
    assert started.utcoffset() == timedelta(hours=-4)


def test_provenance_holds_format_index_ranges_transducer_and_prefilter(
    tmp_path: Path,
) -> None:
    path = make(tmp_path)
    data = bytearray(path.read_bytes())
    # Fields are field-major over three signals, the last being annotations.
    transducers = 256 + 3 * 16
    prefilters = transducers + 3 * (80 + 8 * 5)
    data[transducers : transducers + 3] = b"Ag "
    data[prefilters + 80 : prefilters + 86] = b"HP:0.1"
    path.write_bytes(bytes(data))
    with EdfFile.open(path) as edf:
        a, b = channel_list(build(edf))
    assert a.provenance() == {
        "format": "EDF+C",
        "signal_index": 0,
        "physical_min": -100.0,
        "physical_max": 100.0,
        "digital_min": -100,
        "digital_max": 100,
        "transducer": "Ag",
    }
    assert b.provenance() == {
        "format": "EDF+C",
        "signal_index": 1,
        "physical_min": 0.0,
        "physical_max": 10.0,
        "digital_min": 0,
        "digital_max": 1000,
        "prefilter": "HP:0.1",
    }


def test_provenance_returns_a_copy(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        (channel, _) = channel_list(build(edf))
    channel.provenance()["format"] = "changed"
    assert channel.provenance()["format"] == "EDF+C"


def test_sources_satisfy_the_writer_protocols(tmp_path: Path) -> None:
    path = make(tmp_path, annotations=[(1.0, None, "x")])
    with EdfFile.open(path) as edf:
        sources = build(edf)
    for source in sources.continuous:
        assert isinstance(source, MultiChannelSource)
        for channel in source.channels():
            assert isinstance(channel, ContinuousChannelMetadata)
    for event in sources.events:
        assert isinstance(event, EventChannelSource)


def test_windows_tile_the_records_for_two_rates(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        source = EdfRecordSource(
            edf,
            [0, 1],
            build(edf).continuous[0].channels(),
            records_per_window=2,
        )
        windows = list(source.iter_windows())
        assert [[len(b) for b in w] for w in windows] == [
            [8, 4],
            [8, 4],
            [4, 2],
        ]
        a, b = collect(source)
        np.testing.assert_array_equal(a, edf.read_physical(0, 0, 5))
        np.testing.assert_array_equal(b, edf.read_physical(1, 0, 5))
        totals = [c.num_samples() for c in source.channels()]
    assert (len(a), len(b)) == tuple(totals) == (20, 10)


def test_a_window_holds_at_most_the_records_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        # One record holds 4 + 2 + 8 samples, so 28 bytes.
        monkeypatch.setattr(edf_reader, "WINDOW_BYTES", 2 * 28 + 1)
        (source,) = build(edf).continuous
        sizes = [len(window[0]) for window in source.iter_windows()]
    assert sizes == [8, 8, 4]


def test_a_record_larger_than_the_window_still_reads_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        monkeypatch.setattr(edf_reader, "WINDOW_BYTES", 1)
        (source,) = build(edf).continuous
        sizes = [len(window[0]) for window in source.iter_windows()]
    assert sizes == [4] * 5


def test_a_gap_is_nan_and_samples_land_at_the_onset(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make_gapped(tmp_path)) as edf:
        (source,) = build(edf).continuous
        a, b = collect(source)
        expected_a = np.concatenate(
            [
                edf.read_physical(0, 0, 2),
                np.full(12, np.nan, dtype=np.float32),
                edf.read_physical(0, 2, 4),
            ]
        )
        expected_b = np.concatenate(
            [
                edf.read_physical(1, 0, 2),
                np.full(6, np.nan, dtype=np.float32),
                edf.read_physical(1, 2, 4),
            ]
        )
    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)
    assert [c.num_samples() for c in source.channels()] == [len(a), len(b)]
    assert (len(a), len(b)) == (28, 14)


def test_a_gap_is_cut_into_windows_of_the_records_per_window(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make_gapped(tmp_path)) as edf:
        channels = build(edf).continuous[0].channels()
        source = EdfRecordSource(edf, [0, 1], channels, records_per_window=1)
        windows = list(source.iter_windows())
    assert [[len(b) for b in w] for w in windows] == [[4, 2]] * 7
    assert [bool(np.isnan(w[0]).all()) for w in windows] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_gap_blocks_scale_with_the_samples_per_record_of_each_channel(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [
        ("A", "uV", -1.0, 1.0, -1, 1, 1),
        ("B", "uV", -1.0, 1.0, -1, 1, 4),
    ]
    records = [
        [np.zeros(1, dtype=np.int16), np.zeros(4, dtype=np.int16)]
        for _ in range(2)
    ]
    path = make(
        tmp_path,
        signals=signals,
        records=records,
        flavor="EDF+D",
        record_onsets=[0.0, 3.0],
    )
    with EdfFile.open(path) as edf:
        channels = build(edf).continuous[0].channels()
        source = EdfRecordSource(edf, [0, 1], channels, records_per_window=1)
        sizes = [[len(b) for b in w] for w in source.iter_windows()]
        layouts = [channel_layout(edf, s) for s in (0, 1)]
    assert sizes == [[1, 4], [1, 4], [1, 4], [1, 4]]
    assert layouts[0][1] == 4
    assert layouts[1][1] == 16


def test_channel_layout_of_a_continuous_file_is_one_segment(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        assert channel_layout(edf, 0) == ([(0, 0, 5)], 20)
        assert channel_layout(edf, 1) == ([(0, 0, 5)], 10)


def test_channel_layout_of_several_segments_gives_the_gaps(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make_gapped(tmp_path)) as edf:
        assert channel_layout(edf, 0) == ([(0, 0, 2), (12, 2, 2)], 28)
        assert channel_layout(edf, 1) == ([(0, 0, 2), (6, 2, 2)], 14)


def test_channel_layout_of_an_empty_file_is_empty(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path, n=0)) as edf:
        assert channel_layout(edf, 0) == ([], 0)


def make_single_signal_gapped(
    tmp_path: Path, samples_per_record: int, onsets: list[float]
) -> Path:
    """Write an EDF+D file of one signal with one record at each onset."""
    signals: list[SignalSpec] = [
        ("A", "uV", -1.0, 1.0, -1, 1, samples_per_record)
    ]
    records = [[np.zeros(samples_per_record, dtype=np.int16)] for _ in onsets]
    return make(
        tmp_path,
        signals=signals,
        records=records,
        flavor="EDF+D",
        record_onsets=onsets,
    )


def test_channel_layout_rejects_a_segment_start_off_the_sample_grid(
    tmp_path: Path,
) -> None:
    path = make_single_signal_gapped(tmp_path, 3, [0.0, 2.1])
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="sample period"),
    ):
        channel_layout(edf, 0)


def test_channel_layout_accepts_an_onset_within_the_tolerance(
    tmp_path: Path,
) -> None:
    onset = 2.3 + ONSET_TOLERANCE_S / 10
    path = make_single_signal_gapped(tmp_path, 10, [0.0, onset])
    with EdfFile.open(path) as edf:
        assert channel_layout(edf, 0) == ([(0, 0, 1), (13, 1, 1)], 33)


def test_build_sources_rejects_a_segment_start_off_the_sample_grid(
    tmp_path: Path,
) -> None:
    path = make_single_signal_gapped(tmp_path, 3, [0.0, 2.1])
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="sample period"),
    ):
        build(edf)


def test_grouping_into_sweeps_keeps_channel_order_and_values(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [
        (name, "uV", -100.0, 100.0, -100, 100, 4) for name in "ABCDE"
    ]
    records = [
        [np.arange(4, dtype=np.int16) + 10 * s + k for s in range(5)]
        for k in range(3)
    ]
    path = make(tmp_path, signals=signals, records=records)
    with EdfFile.open(path) as edf:
        whole = build(edf)
        split = build(edf, max_sweep_channels=2)
        assert len(whole.continuous) == 1
        assert [len(s.channels()) for s in split.continuous] == [2, 2, 1]
        assert [c.id for c in channel_list(split)] == list("01234")
        assert channel_list(split) == channel_list(whole)
        expected = collect(whole.continuous[0])
        got = [
            column for source in split.continuous for column in collect(source)
        ]
    for want, have in zip(expected, got, strict=True):
        np.testing.assert_array_equal(want, have)


@pytest.mark.parametrize("value", [0, -1])
def test_max_sweep_channels_below_one_raises(
    tmp_path: Path, value: int
) -> None:
    with (
        EdfFile.open(make(tmp_path)) as edf,
        pytest.raises(ValueError, match="max_sweep_channels"),
    ):
        build(edf, max_sweep_channels=value)


def test_a_file_without_data_signals_raises(tmp_path: Path) -> None:
    path = make(tmp_path, signals=[], records=[[], []])
    with EdfFile.open(path) as edf, pytest.raises(EdfFormatError, match="data"):
        build(edf)


def test_a_file_without_annotations_has_no_event_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO), EdfFile.open(make(tmp_path)) as edf:
        sources = build(edf)
    assert sources.events == []
    assert "no annotations" in caplog.text


def test_annotations_give_one_event_source_at_the_header_start(
    tmp_path: Path,
) -> None:
    path = make(tmp_path, annotations=[(2.0, 1.0, "sleep")])
    with EdfFile.open(path) as edf:
        sources = build(edf)
        (event,) = sources.events
        header_start = start_us(edf.header, UTC_ZONE)
    assert isinstance(event, EdfAnnotationSource)
    assert event.read_text(0, 1) == ["sleep"]
    assert event.read_events(0, 1).tolist() == [header_start + 2_000_000]


def test_recording_meta_defaults_to_the_utc_header_start(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=frozenset()
        )
    assert meta.session_start_time == datetime(2021, 3, 4, 5, 6, 7, tzinfo=UTC)
    assert meta.reference_time is None
    assert meta.source["timezone"] == "UTC"


def test_recording_meta_places_the_start_in_a_named_zone(
    tmp_path: Path,
) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        meta = recording_meta(
            edf.header, timezone=NEW_YORK, subject_fields=frozenset()
        )
    started = meta.session_start_time
    assert started is not None
    assert started.utcoffset() == timedelta(hours=-5)
    assert started.replace(tzinfo=None) == START
    assert meta.source["timezone"] == "America/New_York"


@pytest.mark.parametrize(
    "fields",
    [
        frozenset(combo)
        for size in range(5)
        for combo in itertools.combinations(sorted(ALL_FIELDS), size)
    ],
    ids=lambda fields: "+".join(sorted(fields)) or "none",
)
def test_subject_follows_the_chosen_fields(
    tmp_path: Path, fields: frozenset[str]
) -> None:
    path = make(tmp_path, patient_field="P-17 F 02-MAR-1980 Jane_Doe")
    with EdfFile.open(path) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=fields
        )
    expected = {
        "code": ("id", "P-17"),
        "sex": ("sex", "F"),
        "birthdate": ("date_of_birth", "1980-03-02"),
        "name": ("name", "Jane_Doe"),
    }
    assert meta.subject == {
        key: value for f, (key, value) in expected.items() if f in fields
    }


def test_subject_omits_unknown_sub_fields(tmp_path: Path) -> None:
    with EdfFile.open(make(tmp_path)) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=ALL_FIELDS
        )
    assert meta.subject == {}


def test_a_plain_edf_gives_an_empty_subject_and_session(tmp_path: Path) -> None:
    path = make(tmp_path, flavor="EDF", patient_field="P-17 F 02-MAR-1980 Jo")
    with EdfFile.open(path) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=ALL_FIELDS
        )
    assert meta.subject == {}
    assert meta.session == {}
    assert "equipment" not in meta.source
    assert meta.source["format"] == "EDF"


def test_session_id_is_the_administration_code_and_never_the_technician(
    tmp_path: Path,
) -> None:
    path = make(
        tmp_path, recording_field="Startdate 04-MAR-2021 ADM-9 TECH-1 EQ-3"
    )
    with EdfFile.open(path) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=ALL_FIELDS
        )
    assert meta.session == {"id": "ADM-9"}
    assert "TECH-1" not in repr(meta)


def test_source_names_converter_writer_format_timezone_and_equipment(
    tmp_path: Path,
) -> None:
    path = make(
        tmp_path, recording_field="Startdate 04-MAR-2021 ADM-9 TECH-1 EQ-3"
    )
    with EdfFile.open(path) as edf:
        meta = recording_meta(
            edf.header, timezone=UTC_ZONE, subject_fields=frozenset()
        )
    source = meta.source
    assert str(source["converter"]).startswith("timeseries-zarr-edf ")
    assert source["writer"] == f"timeseries-zarr {timeseries_zarr.__version__}"
    assert source["format"] == "EDF+C"
    assert source["timezone"] == "UTC"
    assert source["equipment"] == "EQ-3"


def test_a_fractional_first_stamp_moves_the_channel_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = make(
        tmp_path,
        n=4,
        record_onsets=[0.37, 1.37, 2.37, 3.37],
        annotations=[(0.5, None, "x")],
    )
    with caplog.at_level(logging.INFO), EdfFile.open(path) as edf:
        sources = build(edf)
        header_start = start_us(edf.header, UTC_ZONE)
        channels = channel_list(sources)
        (source,) = sources.continuous
        a, _ = collect(source)
        assert channel_layout(edf, 0) == ([(0, 0, 4)], 16)
        (event,) = sources.events
    assert [c.start_us() for c in channels] == [header_start + 370_000] * 2
    assert [c.num_samples() for c in channels] == [16, 8]
    assert not np.isnan(a).any()
    assert event.read_events(0, 1).tolist() == [header_start + 500_000]
    assert "starts 370000 us after the header start" in caplog.text


def test_a_zero_first_stamp_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO), EdfFile.open(make(tmp_path)) as edf:
        build(edf)
    assert "after the header start" not in caplog.text


def test_a_first_stamp_of_one_second_or_more_warns_and_still_applies(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = make(tmp_path, n=2, record_onsets=[3.0, 4.0])
    with caplog.at_level(logging.WARNING), EdfFile.open(path) as edf:
        (channel, _) = channel_list(build(edf))
        header_start = start_us(edf.header, UTC_ZONE)
    assert channel.start_us() == header_start + 3_000_000
    assert "more than the fraction of a second" in caplog.text


def test_channel_layout_grids_edf_d_segments_from_the_first_record(
    tmp_path: Path,
) -> None:
    path = make(
        tmp_path,
        n=4,
        flavor="EDF+D",
        record_onsets=[0.37, 1.37, 5.37, 6.37],
    )
    with EdfFile.open(path) as edf:
        assert channel_layout(edf, 0) == ([(0, 0, 2), (12, 2, 2)], 28)
        assert channel_layout(edf, 1) == ([(0, 0, 2), (6, 2, 2)], 14)


def test_channel_layout_rejects_a_later_segment_off_the_grid_of_the_first(
    tmp_path: Path,
) -> None:
    path = make_single_signal_gapped(tmp_path, 3, [0.37, 2.47])
    with (
        EdfFile.open(path) as edf,
        pytest.raises(EdfFormatError, match="sample period"),
    ):
        channel_layout(edf, 0)
