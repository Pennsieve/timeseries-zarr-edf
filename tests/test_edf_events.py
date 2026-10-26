"""Tests for the annotation event channel."""

import numpy as np
import pytest
from timeseries_zarr import EventChannelSource

from timeseries_zarr_edf import edf_events
from timeseries_zarr_edf.edf_events import (
    ANNOTATIONS_CHANNEL_ID,
    EdfAnnotationSource,
)
from timeseries_zarr_edf.edf_file import EdfAnnotation

START_US = 1_600_000_000_000_000


def source(
    annotations: list[EdfAnnotation], flavor: str = "EDF+C"
) -> EdfAnnotationSource:
    """Return a source over the annotations."""
    return EdfAnnotationSource(annotations, start_us=START_US, flavor=flavor)


def test_columns_is_labels_and_text_without_durations() -> None:
    events = source(
        [EdfAnnotation(1.0, None, "a"), EdfAnnotation(2.0, None, "")]
    )
    assert events.columns() == {"labels", "text"}


def test_columns_is_text_alone_without_durations_or_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(edf_events, "MAX_LABELS", 1)
    events = source(
        [EdfAnnotation(1.0, None, "a"), EdfAnnotation(2.0, None, "b")]
    )
    assert events.columns() == {"text"}


def test_columns_adds_durations_and_labels_when_a_duration_is_present() -> None:
    events = source(
        [EdfAnnotation(1.0, 2.0, "a"), EdfAnnotation(2.0, None, "b")]
    )
    assert events.columns() == {"durations", "labels", "text"}


def test_columns_is_empty_without_events() -> None:
    events = source([])
    assert events.num_events() == 0
    assert events.columns() == frozenset()
    assert events.max_duration_us() is None


def test_id_name_and_unit_name_the_annotations_channel() -> None:
    events = source([EdfAnnotation(1.0, None, "a")])
    assert events.id == ANNOTATIONS_CHANNEL_ID == "annotations"
    assert events.name == events.id
    assert events.unit is None


def test_provenance_holds_the_flavor_and_the_signal_label() -> None:
    events = source([EdfAnnotation(1.0, None, "a")], flavor="EDF+D")
    assert events.provenance() == {
        "format": "EDF+D",
        "signal": "EDF Annotations",
    }


def test_times_are_the_start_plus_the_onset_in_whole_microseconds() -> None:
    events = source(
        [
            EdfAnnotation(1.5, None, "a"),
            EdfAnnotation(-2.0, None, "b"),
            EdfAnnotation(0.0000004, None, "c"),
        ]
    )
    times = events.read_events(0, 3)
    assert times.dtype == np.int64
    assert times.tolist() == [
        START_US - 2_000_000,
        START_US,
        START_US + 1_500_000,
    ]


def test_events_are_sorted_by_time_and_stable_for_ties() -> None:
    events = source(
        [
            EdfAnnotation(5.0, None, "late"),
            EdfAnnotation(1.0, None, "first"),
            EdfAnnotation(1.0, None, "second"),
        ]
    )
    assert events.read_text(0, 3) == ["first", "second", "late"]


def test_label_names_are_the_sorted_distinct_texts_and_labels_index_them() -> (
    None
):
    events = source(
        [
            EdfAnnotation(1.0, None, "b"),
            EdfAnnotation(2.0, None, "a"),
            EdfAnnotation(3.0, None, "b"),
        ]
    )
    assert events.label_names() == ["a", "b"]
    assert events.num_labels() == 2
    labels = events.read_labels(0, 3)
    assert labels.dtype == np.uint16
    assert labels.tolist() == [1, 0, 1]


def test_labels_are_dropped_when_the_distinct_texts_exceed_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(edf_events, "MAX_LABELS", 2)
    at_limit = source(
        [EdfAnnotation(float(i), None, str(i % 2)) for i in range(4)]
    )
    assert "labels" in at_limit.columns()
    over = source([EdfAnnotation(float(i), None, str(i)) for i in range(3)])
    assert "labels" not in over.columns()
    assert over.label_names() is None
    assert over.num_labels() == 0
    with pytest.raises(ValueError, match="labels"):
        over.read_labels(0, 3)


def test_max_duration_is_the_longest_and_a_missing_duration_is_zero() -> None:
    events = source(
        [
            EdfAnnotation(1.0, 2.5, "a"),
            EdfAnnotation(2.0, None, "b"),
            EdfAnnotation(3.0, 0.5, "c"),
        ]
    )
    assert events.max_duration_us() == 2_500_000
    durations = events.read_durations(0, 3)
    assert durations.dtype == np.int64
    assert durations.tolist() == [2_500_000, 0, 500_000]


def test_a_negative_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration"):
        source([EdfAnnotation(1.0, -1.0, "a")])


def test_text_width_is_the_longest_utf8_text() -> None:
    events = source(
        [
            EdfAnnotation(1.0, None, "ab"),
            EdfAnnotation(2.0, None, "\u00e9\u00e9\u00e9"),
        ]
    )
    assert events.text_width() == 6


def test_nul_is_removed_from_text() -> None:
    events = source([EdfAnnotation(1.0, None, "a\x00b")])
    assert events.read_text(0, 1) == ["ab"]


def test_the_unoffered_readers_raise() -> None:
    events = source([EdfAnnotation(1.0, None, "a")])
    with pytest.raises(ValueError, match="duration"):
        events.read_durations(0, 1)
    with pytest.raises(ValueError, match="channels"):
        events.read_channel_ids(0, 1)
    with pytest.raises(ValueError, match="waveform_mean"):
        events.waveform_mean()
    with pytest.raises(ValueError, match="waveform_mean"):
        events.waveform_period_us()


def test_the_source_satisfies_the_event_channel_protocol() -> None:
    assert isinstance(
        source([EdfAnnotation(1.0, None, "a")]), EventChannelSource
    )
