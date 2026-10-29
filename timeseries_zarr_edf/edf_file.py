"""Reader for the data records and annotations of an EDF file."""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt

from timeseries_zarr_edf.edf_header import (
    EdfFormatError,
    EdfHeader,
    read_header,
)

logger = logging.getLogger(__name__)

ONSET_TOLERANCE_S: Final = 1e-6
"""Seconds by which a record step may differ from the record duration."""

MAX_TAL_BYTES: Final = 65536
"""Largest time-stamped annotation list, in bytes, the parser accepts."""

_TEXT_SEPARATOR: Final = "\x14"
_DURATION_SEPARATOR: Final = "\x15"


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of contiguous data records.

    Attributes:
        first_record: Index of the first record in the run.
        n_records: Number of records in the run.
        onset_s: Seconds from the header start to the first record.
    """

    first_record: int
    n_records: int
    onset_s: float


@dataclass(frozen=True, slots=True)
class EdfAnnotation:
    """One annotation text with its time.

    Attributes:
        onset_s: Seconds from the header start.
        duration_s: Duration in seconds, or None when the file gives none.
        text: Annotation text.
    """

    onset_s: float
    duration_s: float | None
    text: str


@dataclass(frozen=True, slots=True)
class Tal:
    """One time-stamped annotation list.

    Attributes:
        onset_s: Seconds from the header start.
        duration_s: Duration in seconds, or None when the list gives none.
        texts: Texts of the list. The time-keeping list holds one empty text.
    """

    onset_s: float
    duration_s: float | None
    texts: tuple[str, ...]


def _parse_seconds(text: str, what: str, where: str) -> float:
    """Parse a finite decimal number of seconds."""
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise EdfFormatError(f"{where}: annotation {what} is not a number")
    return number


def parse_tals(raw: bytes, *, where: str) -> list[Tal]:
    """Parse a buffer of time-stamped annotation lists.

    Args:
        raw: Bytes of one annotation signal in one data record.
        where: Description of the buffer, used in error messages.

    Raises:
        EdfFormatError: A list is longer than MAX_TAL_BYTES, lacks the text
            separator, or has an onset or duration that is not a number.
    """
    tals: list[Tal] = []
    for chunk in raw.split(b"\x00"):
        if not chunk:
            continue
        if len(chunk) > MAX_TAL_BYTES:
            raise EdfFormatError(
                f"{where}: annotation list exceeds {MAX_TAL_BYTES} bytes"
            )
        text = chunk.decode("utf-8", errors="replace")
        head, separator, tail = text.partition(_TEXT_SEPARATOR)
        if not separator:
            raise EdfFormatError(
                f"{where}: annotation list has no text separator"
            )
        onset_text, has_duration, duration_text = head.partition(
            _DURATION_SEPARATOR
        )
        if onset_text[:1] not in ("+", "-"):
            raise EdfFormatError(
                f"{where}: annotation onset does not start with + or -"
            )
        onset = _parse_seconds(onset_text, "onset", where)
        duration = (
            _parse_seconds(duration_text, "duration", where)
            if has_duration
            else None
        )
        texts = tail.split(_TEXT_SEPARATOR)
        if texts and texts[-1] == "":
            texts.pop()
        tals.append(Tal(onset, duration, tuple(texts)))
    return tals


class EdfFile:
    """An open EDF file whose data records are read on demand.

    Attributes:
        path: Path of the file.
        header: Header exactly as the file declares it. Its n_records is -1
            for a file still recording.
        n_records: Number of complete data records in the file.
    """

    def __init__(
        self,
        path: Path,
        header: EdfHeader,
        n_records: int,
        descriptor: int,
    ) -> None:
        """Wrap an open file descriptor. Use open to build one from a path."""
        self.path = path
        self.header = header
        self.n_records = n_records
        self._descriptor: int | None = descriptor
        self._onsets: npt.NDArray[np.float64] | None = None
        self._segments: list[Segment] | None = None
        self._annotations: list[EdfAnnotation] | None = None

    @classmethod
    def open(cls, path: Path) -> Self:
        """Open the file and resolve the record count.

        Raises:
            EdfFormatError: The header is malformed or the file holds fewer
                records than the header declares.
            OSError: The file cannot be opened.
        """
        header = read_header(path)
        data_bytes = path.stat().st_size - header.header_bytes
        record_bytes = header.record_bytes
        if header.n_records == -1:
            n_records = data_bytes // record_bytes
        else:
            n_records = header.n_records
            if data_bytes < n_records * record_bytes:
                raise EdfFormatError(
                    f"{path}: file is truncated: the header declares "
                    f"{n_records} records of {record_bytes} bytes and "
                    f"{data_bytes} data bytes follow the header"
                )
        if data_bytes > n_records * record_bytes:
            logger.warning(
                "%s: %d bytes follow the last complete record",
                path.name,
                data_bytes - n_records * record_bytes,
            )
        return cls(path, header, n_records, os.open(path, os.O_RDONLY))

    def close(self) -> None:
        """Close the file. A second call has no effect."""
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> Self:
        """Return the open file."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()

    def _read(self, offset: int, size: int) -> bytes:
        """Return size bytes at offset, raising when the file is closed.

        Raises:
            ValueError: The file is closed.
            EdfFormatError: The file ends before offset + size.
        """
        if self._descriptor is None:
            raise ValueError(f"{self.path}: file is closed")
        parts: list[bytes] = []
        while size:
            part = os.pread(self._descriptor, size, offset)
            if not part:
                raise EdfFormatError(
                    f"{self.path}: file ends inside a data record"
                )
            parts.append(part)
            offset += len(part)
            size -= len(part)
        return b"".join(parts)

    def _check_records(self, first_record: int, stop_record: int) -> None:
        """Raise ValueError for a record range outside the file."""
        if not 0 <= first_record <= stop_record <= self.n_records:
            raise ValueError(
                f"record range [{first_record}, {stop_record}) is outside "
                f"0..{self.n_records}"
            )

    def read_window(
        self, first_record: int, stop_record: int
    ) -> npt.NDArray[np.int16]:
        """Return the digital samples of records [first, stop).

        The array has one row per record and header.record_samples columns,
        with every signal at its sample_offset.

        Raises:
            ValueError: The file is closed or the range is out of range.
            EdfFormatError: The file is shorter than when it was opened.
        """
        self._check_records(first_record, stop_record)
        header = self.header
        count = stop_record - first_record
        raw = self._read(
            header.header_bytes + first_record * header.record_bytes,
            count * header.record_bytes,
        )
        return np.frombuffer(raw, dtype="<i2").reshape(
            count, header.record_samples
        )

    def signal_digital(
        self, window: npt.NDArray[np.int16], signal_index: int
    ) -> npt.NDArray[np.int16]:
        """Return one signal's digital samples from a read_window array.

        Raises:
            ValueError: The signal index is out of range.
        """
        self._check_signal(signal_index)
        start = self.header.sample_offset(signal_index)
        width = self.header.signals[signal_index].samples_per_record
        block = window[:, start : start + width]
        return np.ascontiguousarray(block, dtype=np.int16).reshape(-1)

    def signal_physical(
        self, window: npt.NDArray[np.int16], signal_index: int
    ) -> npt.NDArray[np.float32]:
        """Return one signal's physical values from a read_window array.

        Raises:
            ValueError: The signal index is out of range.
        """
        self._check_signal(signal_index)
        start = self.header.sample_offset(signal_index)
        signal = self.header.signals[signal_index]
        block = window[:, start : start + signal.samples_per_record]
        physical = block.astype(np.float64) * signal.gain + signal.offset
        return physical.astype(np.float32).reshape(-1)

    def _check_signal(self, signal_index: int) -> None:
        """Raise ValueError for a signal index outside the header."""
        if not 0 <= signal_index < len(self.header.signals):
            raise ValueError(
                f"signal index {signal_index} is outside "
                f"0..{len(self.header.signals) - 1}"
            )

    def num_samples(self, signal_index: int) -> int:
        """Return the recorded samples of a signal, gaps excluded."""
        self._check_signal(signal_index)
        return (
            self.n_records
            * self.header.signals[signal_index].samples_per_record
        )

    def read_records(
        self, signal_index: int, first_record: int, stop_record: int
    ) -> npt.NDArray[np.int16]:
        """Return one signal's digital samples for records [first, stop).

        Raises:
            ValueError: The file is closed or an index is out of range.
            EdfFormatError: The file is shorter than when it was opened.
        """
        self._check_signal(signal_index)
        window = self.read_window(first_record, stop_record)
        return self.signal_digital(window, signal_index)

    def read_physical(
        self, signal_index: int, first_record: int, stop_record: int
    ) -> npt.NDArray[np.float32]:
        """Return one signal's physical values for records [first, stop).

        Raises:
            ValueError: The file is closed or an index is out of range.
            EdfFormatError: The file is shorter than when it was opened.
        """
        self._check_signal(signal_index)
        window = self.read_window(first_record, stop_record)
        return self.signal_physical(window, signal_index)

    def _annotation_buffer(self, record: int, signal_index: int) -> bytes:
        """Return the raw bytes of one annotation signal in one record."""
        self._check_records(record, record + 1)
        header = self.header
        return self._read(
            header.header_bytes
            + record * header.record_bytes
            + 2 * header.sample_offset(signal_index),
            2 * header.signals[signal_index].samples_per_record,
        )

    def _where(self, record: int, signal_index: int) -> str:
        """Describe a record and signal for error messages."""
        return f"{self.path}: record {record} signal {signal_index}"

    def _record_onset(self, record: int, signal_index: int) -> float:
        """Return the stamp of the time-keeping list of one record.

        Raises:
            EdfFormatError: The record has no time-keeping list.
        """
        where = self._where(record, signal_index)
        first = self._annotation_buffer(record, signal_index).split(b"\x00", 1)[
            0
        ]
        if not first:
            raise EdfFormatError(f"{where}: no time-keeping annotation list")
        return parse_tals(first, where=where)[0].onset_s

    def record_onsets(self) -> npt.NDArray[np.float64]:
        """Return the onset of every record in seconds from the header start.

        A plain EDF file starts at 0. An EDF+C file starts at the stamp of its
        first record, and every later record follows its predecessor. An EDF+D
        file takes the stamp of each record.

        Raises:
            EdfFormatError: An EDF+ file has no annotation signal or a needed
                record without a time-keeping list.
        """
        if self._onsets is not None:
            return self._onsets
        header = self.header
        steps = (
            np.arange(self.n_records, dtype=np.float64) * header.record_duration
        )
        if header.is_discontinuous:
            if not header.annotation_indices:
                raise EdfFormatError(
                    f"{self.path}: EDF+D file has no annotation signal"
                )
            signal_index = header.annotation_indices[0]
            onsets = np.array(
                [
                    self._record_onset(record, signal_index)
                    for record in range(self.n_records)
                ],
                dtype=np.float64,
            )
        elif (
            header.flavor == "EDF+C"
            and header.annotation_indices
            and self.n_records
        ):
            onsets = self._record_onset(0, header.annotation_indices[0]) + steps
        else:
            onsets = steps
        self._onsets = onsets
        return onsets

    def segments(self) -> list[Segment]:
        """Return the runs of contiguous records.

        Raises:
            EdfFormatError: The first onset is negative or a record starts
                before the previous record ends.
        """
        if self._segments is not None:
            return self._segments
        if self.n_records == 0:
            self._segments = []
            return self._segments
        onsets = self.record_onsets()
        duration = self.header.record_duration
        if onsets[0] < 0:
            raise EdfFormatError(f"{self.path}: first record onset is negative")
        steps = np.diff(onsets)
        overlap = np.flatnonzero(steps < duration - ONSET_TOLERANCE_S)
        if overlap.size:
            raise EdfFormatError(
                f"{self.path}: record {int(overlap[0]) + 1} starts before "
                "the previous record ends"
            )
        breaks = np.flatnonzero(steps > duration + ONSET_TOLERANCE_S) + 1
        firsts = [0, *(int(index) for index in breaks)]
        stops = [*firsts[1:], self.n_records]
        self._segments = [
            Segment(first, stop - first, float(onsets[first]))
            for first, stop in zip(firsts, stops, strict=True)
        ]
        return self._segments

    def first_onset_s(self) -> float:
        """Return the onset of the first record in seconds from the header start.

        The value is 0.0 for a file with no records.

        Raises:
            EdfFormatError: The first onset is negative or the stamps are
                malformed.
        """
        segments = self.segments()
        return segments[0].onset_s if segments else 0.0

    def annotations(self) -> list[EdfAnnotation]:
        """Return every annotation text in file order.

        Order is record, then annotation signal, then list, then text. The
        time-keeping list and empty texts add nothing.

        Raises:
            EdfFormatError: An annotation list is malformed.
        """
        if self._annotations is None:
            found: list[EdfAnnotation] = []
            for record in range(self.n_records):
                for signal_index in self.header.annotation_indices:
                    buffer = self._annotation_buffer(record, signal_index)
                    where = self._where(record, signal_index)
                    for tal in parse_tals(buffer, where=where):
                        found.extend(
                            EdfAnnotation(tal.onset_s, tal.duration_s, text)
                            for text in tal.texts
                            if text
                        )
            self._annotations = found
        return list(self._annotations)
