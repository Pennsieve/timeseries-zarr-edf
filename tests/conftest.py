"""Shared test helpers, chiefly the synthetic EDF writer."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

ANNOTATION_LABEL = "EDF Annotations"

type SignalSpec = tuple[str, str, float, float, int, int, int]
"""label, unit, physical min, physical max, digital min, digital max, samples."""

type AnnotationSpec = tuple[float, float | None, str]
"""onset in seconds, duration in seconds or None, text."""


def _field(value: object, width: int) -> bytes:
    """Return the value as ASCII padded with spaces to the width."""
    text = str(value) if not isinstance(value, float) else f"{value:.8g}"
    data = text.encode("latin-1")
    assert len(data) <= width, f"{text!r} is wider than {width}"
    return data.ljust(width)


def encode_tal(
    onset_s: float, duration_s: float | None, texts: Sequence[str]
) -> bytes:
    """Encode one time-stamped annotation list, NUL terminated."""
    out = f"{onset_s:+}"
    if duration_s is not None:
        out += f"\x15{duration_s}"
    out += "\x14" + "".join(f"{text}\x14" for text in texts)
    return out.encode("utf-8") + b"\x00"


def annotation_samples(raw: bytes, samples: int) -> npt.NDArray[np.int16]:
    """Pad annotation bytes with NUL to the sample count and view as int16."""
    assert len(raw) <= 2 * samples
    return np.frombuffer(raw.ljust(2 * samples, b"\x00"), dtype="<i2")


def _annotation_buffers(
    onsets: Sequence[float],
    annotations: Sequence[AnnotationSpec],
    record_duration: float,
) -> list[bytes]:
    """Return the annotation bytes of every record, time-keeping list first."""
    per_record: list[list[bytes]] = [
        [encode_tal(onset, None, [""])] for onset in onsets
    ]
    for onset_s, duration_s, text in annotations:
        holders = [
            k
            for k, begin in enumerate(onsets)
            if begin <= onset_s < begin + record_duration
        ]
        if not holders:
            raise ValueError(f"no record holds onset {onset_s}")
        per_record[holders[0]].append(encode_tal(onset_s, duration_s, [text]))
    return [b"".join(parts) for parts in per_record]


def _header_bytes(
    all_signals: Sequence[SignalSpec],
    *,
    start: datetime,
    reserved: str,
    n_records_text: object,
    patient_field: str,
    recording_field: str,
    record_duration: float,
) -> bytes:
    """Return the fixed header and the field-major signal blocks."""
    n_signals = len(all_signals)
    header = bytearray()
    header += _field("0", 8)
    header += _field(patient_field, 80)
    header += _field(recording_field, 80)
    header += _field(start.strftime("%d.%m.%y"), 8)
    header += _field(start.strftime("%H.%M.%S"), 8)
    header += _field(256 * (n_signals + 1), 8)
    header += _field(reserved, 44)
    header += _field(n_records_text, 8)
    header += _field(record_duration, 8)
    header += _field(n_signals, 4)
    assert len(header) == 256
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    # Index into a SignalSpec for each header column; None writes blanks.
    source = (0, None, 1, 2, 3, 4, 5, None, 6, None)
    for width, index in zip(widths, source, strict=True):
        for spec in all_signals:
            if index is None:
                header += _field("", width)
            elif index in (2, 3):
                header += _field(float(spec[index]), width)
            else:
                header += _field(spec[index], width)
    assert len(header) == 256 * (n_signals + 1)
    return bytes(header)


def build_edf(
    path: Path,
    *,
    signals: Sequence[SignalSpec],
    records: Sequence[Sequence[npt.NDArray[np.int16]]],
    start: datetime,
    flavor: str = "EDF+C",
    annotations: Sequence[AnnotationSpec] = (),
    record_onsets: Sequence[float] | None = None,
    patient_field: str = "X X X X",
    recording_field: str = "Startdate X X X X",
    record_duration: float = 1.0,
    n_records_field: int | None = None,
) -> Path:
    """Write a synthetic EDF, EDF+C, or EDF+D file and return its path.

    An EDF+ flavor gets one extra signal labeled "EDF Annotations", last in
    storage order, sized for the time-keeping list plus the annotations. Each
    annotation goes in the record whose span holds its onset.
    """
    n_records = len(records)
    if flavor == "EDF" and annotations:
        raise ValueError("plain EDF cannot hold annotations")
    onsets = (
        list(record_onsets)
        if record_onsets is not None
        else [k * record_duration for k in range(n_records)]
    )
    assert len(onsets) == n_records
    reserved = "" if flavor == "EDF" else flavor
    buffers = (
        _annotation_buffers(onsets, annotations, record_duration)
        if reserved
        else []
    )
    annotation_spr = (
        max((len(buffer) + 1) // 2 for buffer in buffers) if buffers else 8
    )
    all_signals = list(signals)
    if reserved:
        all_signals.append(
            (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, annotation_spr)
        )
    header = _header_bytes(
        all_signals,
        start=start,
        reserved=reserved,
        n_records_text=n_records
        if n_records_field is None
        else n_records_field,
        patient_field=patient_field,
        recording_field=recording_field,
        record_duration=record_duration,
    )
    body = bytearray()
    for k, record in enumerate(records):
        assert len(record) == len(signals)
        for spec, samples in zip(signals, record, strict=True):
            assert len(samples) == spec[6], f"record {k}: wrong sample count"
            body += np.asarray(samples, dtype="<i2").tobytes()
        if reserved:
            body += annotation_samples(buffers[k], annotation_spr).tobytes()
    path.write_bytes(header + bytes(body))
    return path
