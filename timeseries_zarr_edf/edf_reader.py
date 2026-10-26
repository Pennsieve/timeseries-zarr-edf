"""Continuous channel sources and recording metadata read from an EDF file."""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from importlib.metadata import version
from typing import Final, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import timeseries_zarr
from timeseries_zarr import RecordingMeta

from timeseries_zarr_edf.edf_events import (
    MICROSECONDS_PER_SECOND,
    EdfAnnotationSource,
)
from timeseries_zarr_edf.edf_file import ONSET_TOLERANCE_S, EdfFile
from timeseries_zarr_edf.edf_header import (
    EdfFormatError,
    EdfHeader,
    fold_unit,
)

logger = logging.getLogger(__name__)

WINDOW_BYTES: Final = 8 * 2**20
"""Most bytes of data records one window reads."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICROSECOND: Final = timedelta(microseconds=1)
_SEX_VALUES: Final = ("M", "F")


@dataclass(frozen=True, slots=True)
class EdfChannel:
    """A continuous channel of one data signal.

    Attributes:
        id: Storage index of the signal as a decimal string.
        name: Signal label, or the id when the label is empty.
        unit: Physical dimension with micro signs folded to "u".
        signal_index: Storage index of the signal in the file.
    """

    id: str
    name: str
    unit: str
    signal_index: int
    _rate_hz: float
    _start_us: int
    _num_samples: int
    _provenance: dict[str, object]

    def rate_hz(self) -> float:
        """Return the sample rate in hertz."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the microseconds since the Unix epoch of sample 0."""
        return self._start_us

    def num_samples(self) -> int:
        """Return the samples of the channel, gaps included."""
        return self._num_samples

    def provenance(self) -> dict[str, object]:
        """Return the signal index, ranges, and EDF flavor."""
        return dict(self._provenance)


class EdfRecordSource:
    """A group of channels read together from the data records of one file.

    Each window holds the same records for every channel, and a gap between
    segments of an EDF+D file is a run of NaN windows before the segment.
    """

    def __init__(
        self,
        edf: EdfFile,
        signal_indices: Sequence[int],
        channels: Sequence[EdfChannel],
        *,
        records_per_window: int,
    ) -> None:
        """Bind the channels of the signals at signal_indices.

        channels holds one channel per signal index, in the same order.
        Raises ValueError if the two differ in length or records_per_window
        is below 1.
        """
        if len(signal_indices) != len(channels):
            raise ValueError(
                f"{len(signal_indices)} signal indices but "
                f"{len(channels)} channels"
            )
        if records_per_window < 1:
            raise ValueError(
                f"records_per_window must be 1 or more, got {records_per_window}"
            )
        self._edf = edf
        self._signal_indices = list(signal_indices)
        self._channels = list(channels)
        self._records_per_window = records_per_window

    def channels(self) -> Sequence[EdfChannel]:
        """Return the channels in channel order."""
        return list(self._channels)

    def iter_windows(self) -> Iterator[list[npt.NDArray[np.float32]]]:
        """Yield one block per channel for each window of the file.

        Raises:
            EdfFormatError: A segment onset is not a multiple of the sample
                period of a channel, or a segment starts before the previous
                one ends at the rate of a channel.
        """
        edf = self._edf
        header = edf.header
        layouts = [channel_layout(edf, s)[0] for s in self._signal_indices]
        per_record = [
            header.signals[s].samples_per_record for s in self._signal_indices
        ]
        for segment_number, segment in enumerate(edf.segments()):
            remaining = [layout[segment_number][0] for layout in layouts]
            while any(remaining):
                blocks = []
                for index, samples in enumerate(per_record):
                    count = min(
                        remaining[index], self._records_per_window * samples
                    )
                    remaining[index] -= count
                    blocks.append(np.full(count, np.nan, dtype=np.float32))
                yield blocks
            stop = segment.first_record + segment.n_records
            for first in range(
                segment.first_record, stop, self._records_per_window
            ):
                last = min(first + self._records_per_window, stop)
                window = edf.read_window(first, last)
                yield [
                    edf.signal_physical(window, s) for s in self._signal_indices
                ]


class EdfSources(NamedTuple):
    """Everything a bundle write needs from one EDF file.

    Attributes:
        continuous: One source for each group of data signals.
        events: The annotations source, or an empty list.
        recording: Content of recording.json.
    """

    continuous: list[EdfRecordSource]
    events: list[EdfAnnotationSource]
    recording: RecordingMeta


def start_us(header: EdfHeader, timezone: tzinfo) -> int:
    """Return the header start in microseconds since the Unix epoch.

    The header start is a wall clock without a zone, and timezone places it.
    An ambiguous wall clock takes its first occurrence.
    """
    return (_zoned_start(header, timezone) - _EPOCH) // _ONE_MICROSECOND


def _zoned_start(header: EdfHeader, timezone: tzinfo) -> datetime:
    """Return the header start placed in timezone, first occurrence."""
    return header.start.replace(tzinfo=timezone, fold=0)


def channel_layout(
    edf: EdfFile, signal_index: int
) -> tuple[list[tuple[int, int, int]], int]:
    """Return how a signal's recorded samples sit on its timeline.

    The list holds one (gap_samples_before, first_record, n_records) entry for
    each segment, and the second value is the total samples of the channel.
    The first segment starts at sample 0, which is the first record. Every later
    segment starts at its onset after the first record, which must be a
    multiple of the sample period within ONSET_TOLERANCE_S.

    Raises:
        EdfFormatError: A segment onset is not a multiple of the sample period
            of the signal, or a segment starts before the previous segment ends
            at the rate of the signal.
    """
    header = edf.header
    samples = header.signals[signal_index].samples_per_record
    rate = header.signals[signal_index].rate_hz(header.record_duration)
    layout: list[tuple[int, int, int]] = []
    end = 0
    origin = edf.first_onset_s()
    for segment in edf.segments():
        since_first = segment.onset_s - origin
        first_sample = round(since_first * rate)
        if abs(since_first - first_sample / rate) > ONSET_TOLERANCE_S:
            raise EdfFormatError(
                f"{edf.path}: segment at record {segment.first_record} starts "
                f"{since_first} s after the first record, which is not a "
                f"multiple of the sample period of signal {signal_index}"
            )
        gap = first_sample - end
        if gap < 0:
            raise EdfFormatError(
                f"{edf.path}: segment at record {segment.first_record} "
                f"overlaps the previous segment on signal {signal_index}"
            )
        layout.append((gap, segment.first_record, segment.n_records))
        end = first_sample + segment.n_records * samples
    return layout, end


def _provenance(header: EdfHeader, signal_index: int) -> dict[str, object]:
    """Return the provenance of one data signal."""
    signal = header.signals[signal_index]
    provenance: dict[str, object] = {
        "format": header.flavor,
        "signal_index": signal_index,
        "physical_min": signal.physical_min,
        "physical_max": signal.physical_max,
        "digital_min": signal.digital_min,
        "digital_max": signal.digital_max,
    }
    if signal.transducer:
        provenance["transducer"] = signal.transducer
    if signal.prefilter:
        provenance["prefilter"] = signal.prefilter
    return provenance


def _channel(
    edf: EdfFile, signal_index: int, start_microseconds: int
) -> EdfChannel:
    """Return the channel of one data signal."""
    header = edf.header
    signal = header.signals[signal_index]
    identifier = str(signal_index)
    return EdfChannel(
        id=identifier,
        name=signal.label or identifier,
        unit=fold_unit(signal.unit),
        signal_index=signal_index,
        _rate_hz=signal.rate_hz(header.record_duration),
        _start_us=start_microseconds,
        _num_samples=channel_layout(edf, signal_index)[1],
        _provenance=_provenance(header, signal_index),
    )


def build_sources_from_edf(
    edf: EdfFile,
    *,
    timezone: tzinfo,
    subject_fields: frozenset[str],
    max_sweep_channels: int | None = None,
) -> EdfSources:
    """Return the sources and recording metadata an EDF file supplies.

    Every data signal is one continuous channel, in storage order. Its id is
    the storage index of the signal as a decimal string, counting annotation
    signals, so the id is stable for the file. Its name is the signal label,
    or the id when the label is empty. Its unit is the file's unit with micro
    signs folded to "u", which the writer converts when it is a voltage. Sample 0
    of every channel is at the header start placed in timezone plus the onset
    of the first record, so every offset in the bundle is that onset. The
    reference time stays the header start. The continuous list holds one source for each group of at
    most max_sweep_channels consecutive channels, and None puts every channel
    in one group. Each group reads the records once. The grouping leaves
    channel order unchanged. The event list holds one source, with id
    "annotations", when the file holds any annotation text, and is empty
    otherwise. subject_fields chooses the patient sub-fields that reach
    recording.json.

    Raises:
        ValueError: max_sweep_channels is below 1.
        EdfFormatError: The file has no data signal, a segment starts at a time
            after the first record that is not a multiple of a sample period,
            or segments overlap.
    """
    if max_sweep_channels is not None and max_sweep_channels < 1:
        raise ValueError(
            f"max_sweep_channels must be 1 or more, got {max_sweep_channels}"
        )
    header = edf.header
    indices = header.data_indices
    if not indices:
        raise EdfFormatError(f"{edf.path}: the file has no data signal")
    start_microseconds = start_us(header, timezone)
    origin_us = round(edf.first_onset_s() * MICROSECONDS_PER_SECOND)
    if origin_us >= MICROSECONDS_PER_SECOND:
        logger.warning(
            "the first record starts %d us after the header start, which is "
            "more than the fraction of a second EDF+ allows",
            origin_us,
        )
    elif origin_us:
        logger.info(
            "the first record starts %d us after the header start", origin_us
        )
    channels = [
        _channel(edf, s, start_microseconds + origin_us) for s in indices
    ]
    size = len(indices) if max_sweep_channels is None else max_sweep_channels
    records_per_window = max(1, WINDOW_BYTES // header.record_bytes)
    continuous = [
        EdfRecordSource(
            edf,
            indices[first : first + size],
            channels[first : first + size],
            records_per_window=records_per_window,
        )
        for first in range(0, len(indices), size)
    ]
    events: list[EdfAnnotationSource] = []
    annotations = edf.annotations()
    if annotations:
        events.append(
            EdfAnnotationSource(
                annotations, start_us=start_microseconds, flavor=header.flavor
            )
        )
    else:
        logger.info("the file holds no annotations, so no event channel")
    recording = recording_meta(
        header, timezone=timezone, subject_fields=subject_fields
    )
    return EdfSources(continuous, events, recording)


def _subject(
    header: EdfHeader, subject_fields: frozenset[str]
) -> dict[str, object]:
    """Return the subject object of recording.json."""
    patient = header.patient()
    if patient is None:
        return {}
    subject: dict[str, object] = {}
    if "code" in subject_fields and patient.code:
        subject["id"] = patient.code
    if "sex" in subject_fields and patient.sex in _SEX_VALUES:
        subject["sex"] = patient.sex
    if "birthdate" in subject_fields and patient.birthdate is not None:
        subject["date_of_birth"] = patient.birthdate.isoformat()
    if "name" in subject_fields and patient.name:
        subject["name"] = patient.name
    return subject


def _zone_name(timezone: tzinfo) -> str:
    """Return the IANA key of a zone, or its string form without one."""
    return timezone.key if isinstance(timezone, ZoneInfo) else str(timezone)


def recording_meta(
    header: EdfHeader, *, timezone: tzinfo, subject_fields: frozenset[str]
) -> RecordingMeta:
    """Return the recording.json content an EDF header supplies.

    session_start_time is the header start placed in timezone, and
    reference_time is None, so the writer takes the session start as the
    reference time. A plain EDF file gives an empty subject and session.
    An EDF+ file gives the subject sub-fields named in subject_fields, and
    session.id from the administration code when it is not empty. The
    technician never enters the result. source names the converter, the
    writer, the EDF flavor, the timezone, and the equipment of an EDF+ file.
    """
    recording = header.recording()
    session: dict[str, object] = {}
    source: dict[str, object] = {
        "converter": f"timeseries-zarr-edf {version('timeseries-zarr-edf')}",
        "writer": f"timeseries-zarr {timeseries_zarr.__version__}",
        "format": header.flavor,
        "timezone": _zone_name(timezone),
    }
    if recording is not None:
        if recording.admin_code:
            session["id"] = recording.admin_code
        if recording.equipment:
            source["equipment"] = recording.equipment
    return RecordingMeta(
        session_start_time=_zoned_start(header, timezone),
        reference_time=None,
        subject=_subject(header, subject_fields),
        session=session,
        source=source,
    )
