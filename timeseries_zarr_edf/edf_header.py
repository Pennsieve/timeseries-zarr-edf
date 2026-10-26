"""Header dataclasses and parser for EDF, EDF+C, and EDF+D files."""

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal

FIXED_HEADER_BYTES: Final = 256
"""Size of the fixed header at the start of the file."""

SIGNAL_HEADER_BYTES: Final = 256
"""Size of the header block each signal adds."""

ANNOTATION_LABEL: Final = "EDF Annotations"
"""Label that marks a signal as a buffer of time-stamped annotations."""

_PIVOT_YEAR: Final = 85
_FOUR_DIGIT_YEAR: Final = 100
_MONTHS: Final = {
    name: number
    for number, name in enumerate(
        (
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ),
        start=1,
    )
}
_START_DATE: Final = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{2})")
_START_TIME: Final = re.compile(r"(\d{2})[.:\-](\d{2})[.:\-](\d{2})")
_SUB_DATE: Final = re.compile(r"(\d{1,2})-([A-Za-z]{3})-(\d{2}|\d{4})")
_MICRO_SPELLINGS: Final = ("\u00c2\u00b5", "\u00b5", "\u03bc")


class EdfFormatError(ValueError):
    """The file is not an EDF the converter can read."""


@dataclass(frozen=True, slots=True)
class EdfSignalHeader:
    """Header block of one signal.

    Attributes:
        label: Signal label, stripped.
        transducer: Transducer description, stripped.
        unit: Physical dimension as written in the file, stripped.
        physical_min: Physical value that digital_min maps to.
        physical_max: Physical value that digital_max maps to. It may be
            smaller than physical_min, which gives a negative gain.
        digital_min: Digital value that maps to physical_min.
        digital_max: Digital value that maps to physical_max.
        prefilter: Prefiltering description, stripped.
        samples_per_record: Samples this signal holds in one data record.
    """

    label: str
    transducer: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefilter: str
    samples_per_record: int

    @property
    def is_annotation(self) -> bool:
        """Whether the signal is an annotation buffer, not samples."""
        return self.label == ANNOTATION_LABEL

    def rate_hz(self, record_duration: float) -> float:
        """Return the sample rate in hertz for a record of the given seconds."""
        return self.samples_per_record / record_duration

    @property
    def gain(self) -> float:
        """Physical units per digital step."""
        return (self.physical_max - self.physical_min) / (
            self.digital_max - self.digital_min
        )

    @property
    def offset(self) -> float:
        """Physical value of digital zero."""
        return self.physical_min - self.digital_min * self.gain


@dataclass(frozen=True, slots=True)
class EdfPatient:
    """EDF+ patient sub-fields.

    Attributes:
        code: Patient code, or "" when unknown.
        sex: "M" or "F", or "" when unknown.
        birthdate: Date of birth, or None when unknown.
        name: Patient name, or "" when unknown.
    """

    code: str
    sex: str
    birthdate: date | None
    name: str


@dataclass(frozen=True, slots=True)
class EdfRecordingField:
    """EDF+ recording sub-fields.

    Attributes:
        startdate: Start date written in the field, or None when unknown.
        admin_code: Administration code, or "" when unknown.
        technician: Technician code, or "" when unknown.
        equipment: Equipment code, or "" when unknown.
    """

    startdate: date | None
    admin_code: str
    technician: str
    equipment: str


@dataclass(frozen=True, slots=True)
class EdfHeader:
    """Parsed fixed header and per-signal headers.

    Attributes:
        version: Version field, always "0".
        patient_field: Local patient identification, stripped.
        recording_field: Local recording identification, stripped.
        start: Naive wall-clock start of the recording.
        header_bytes: Total header size in bytes.
        reserved: Reserved field, stripped. It names the EDF+ flavor.
        n_records: Declared number of data records, -1 while recording.
        record_duration: Seconds one data record spans.
        signals: Signal headers in storage order, annotation signals
            included.
    """

    version: str
    patient_field: str
    recording_field: str
    start: datetime
    header_bytes: int
    reserved: str
    n_records: int
    record_duration: float
    signals: tuple[EdfSignalHeader, ...]

    @property
    def flavor(self) -> Literal["EDF", "EDF+C", "EDF+D"]:
        """Format flavor named by the reserved field."""
        if self.reserved.startswith("EDF+C"):
            return "EDF+C"
        if self.reserved.startswith("EDF+D"):
            return "EDF+D"
        return "EDF"

    @property
    def is_edfplus(self) -> bool:
        """Whether the file is EDF+C or EDF+D."""
        return self.flavor != "EDF"

    @property
    def is_discontinuous(self) -> bool:
        """Whether the file is EDF+D, whose records may leave gaps."""
        return self.flavor == "EDF+D"

    @property
    def annotation_indices(self) -> tuple[int, ...]:
        """Storage indices of the annotation signals."""
        return tuple(
            index
            for index, signal in enumerate(self.signals)
            if signal.is_annotation
        )

    @property
    def data_indices(self) -> tuple[int, ...]:
        """Storage indices of the signals that hold samples."""
        return tuple(
            index
            for index, signal in enumerate(self.signals)
            if not signal.is_annotation
        )

    @property
    def record_samples(self) -> int:
        """Samples in one data record, summed over every signal."""
        return sum(signal.samples_per_record for signal in self.signals)

    @property
    def record_bytes(self) -> int:
        """Bytes in one data record."""
        return 2 * self.record_samples

    def sample_offset(self, signal_index: int) -> int:
        """Return the position of the signal's first sample in a record."""
        return sum(
            signal.samples_per_record for signal in self.signals[:signal_index]
        )

    def patient(self) -> EdfPatient | None:
        """Return the patient sub-fields, or None for plain EDF."""
        if not self.is_edfplus:
            return None
        tokens = self.patient_field.split(None, 3)
        tokens += [""] * (4 - len(tokens))
        code, sex, birth, name = tokens
        return EdfPatient(
            code=_known(code),
            sex=sex.upper() if sex.upper() in ("M", "F") else "",
            birthdate=_parse_sub_date(birth),
            name=_known(name),
        )

    def recording(self) -> EdfRecordingField | None:
        """Return the recording sub-fields, or None for plain EDF."""
        if not self.is_edfplus:
            return None
        tokens = self.recording_field.split(None, 4)
        if not tokens or tokens[0].lower() != "startdate":
            return EdfRecordingField(None, "", "", "")
        tokens += [""] * (5 - len(tokens))
        _, startdate, admin_code, technician, equipment = tokens
        return EdfRecordingField(
            startdate=_parse_sub_date(startdate),
            admin_code=_known(admin_code),
            technician=_known(technician),
            equipment=_known(equipment),
        )


def _known(token: str) -> str:
    """Return the token, or "" when it is the unknown marker "X"."""
    return "" if token == "X" else token


def _expand_year(year: int) -> int:
    """Map a two-digit year through the pivot, and keep a four-digit year."""
    if year >= _FOUR_DIGIT_YEAR:
        return year
    return 1900 + year if year >= _PIVOT_YEAR else 2000 + year


def _parse_sub_date(token: str) -> date | None:
    """Parse dd-MMM-yyyy from an EDF+ sub-field, or None when not a date."""
    match = _SUB_DATE.fullmatch(token)
    if match is None:
        return None
    day, month_name, year = match.groups()
    month = _MONTHS.get(month_name.upper())
    if month is None:
        return None
    try:
        return date(_expand_year(int(year)), month, int(day))
    except ValueError:
        return None


def fold_unit(unit: str) -> str:
    """Fold every micro sign spelling to "u" and strip the unit."""
    for spelling in _MICRO_SPELLINGS:
        unit = unit.replace(spelling, "u")
    return unit.strip()


def _text(buffer: bytes, start: int, width: int) -> str:
    """Return a header field decoded as latin-1 and stripped."""
    return buffer[start : start + width].decode("latin-1").strip()


def _to_int(text: str, field: str, where: str) -> int:
    """Parse an integer field, accepting an integral float such as "1.0"."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not number.is_integer():
        raise EdfFormatError(
            f"{where}: header field {field} must be an integer, got {text!r}"
        )
    return int(number)


def _to_float(text: str, field: str, where: str) -> float:
    """Parse a finite floating-point field."""
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise EdfFormatError(
            f"{where}: header field {field} must be a finite number, "
            f"got {text!r}"
        )
    return number


def _parse_start(date_text: str, time_text: str, where: str) -> datetime:
    """Parse the dd.mm.yy and hh.mm.ss start fields into a naive datetime."""
    date_match = _START_DATE.fullmatch(date_text)
    time_match = _START_TIME.fullmatch(time_text)
    if date_match is None:
        raise EdfFormatError(
            f"{where}: header field start date is not dd.mm.yy"
        )
    if time_match is None:
        raise EdfFormatError(
            f"{where}: header field start time is not hh.mm.ss"
        )
    day, month, year = (int(part) for part in date_match.groups())
    hour, minute, second = (int(part) for part in time_match.groups())
    try:
        return datetime(_expand_year(year), month, day, hour, minute, second)
    except ValueError:
        raise EdfFormatError(
            f"{where}: header field start date or time is out of range"
        ) from None


def _signal_count(fixed: bytes, where: str) -> int:
    """Return the signal count from the first 256 header bytes."""
    if len(fixed) < FIXED_HEADER_BYTES:
        raise EdfFormatError(
            f"{where}: file is shorter than the {FIXED_HEADER_BYTES}-byte "
            "fixed header"
        )
    count = _to_int(_text(fixed, 252, 4), "number of signals", where)
    if count < 1:
        raise EdfFormatError(
            f"{where}: header field number of signals must be at least 1, "
            f"got {count}"
        )
    return count


def parse_header(buffer: bytes, *, where: str = "buffer") -> EdfHeader:
    """Parse the header at the start of an EDF file.

    Args:
        buffer: The file bytes, at least the whole header.
        where: Name of the file, used in error messages.

    Raises:
        EdfFormatError: A field is missing, malformed, or inconsistent.
    """
    n_signals = _signal_count(buffer, where)
    version = _text(buffer, 0, 8)
    if version != "0":
        raise EdfFormatError(
            f"{where}: header field version must be 0, so BDF and other "
            "formats are not supported"
        )
    header_bytes = _to_int(_text(buffer, 184, 8), "header bytes", where)
    expected = FIXED_HEADER_BYTES + SIGNAL_HEADER_BYTES * n_signals
    if header_bytes != expected:
        raise EdfFormatError(
            f"{where}: header field header bytes is {header_bytes}, "
            f"expected {expected} for {n_signals} signals"
        )
    if len(buffer) < header_bytes:
        raise EdfFormatError(
            f"{where}: file holds {len(buffer)} bytes, "
            f"shorter than the {header_bytes}-byte header"
        )
    n_records = _to_int(_text(buffer, 236, 8), "number of records", where)
    if n_records < -1:
        raise EdfFormatError(
            f"{where}: header field number of records must be -1 or more, "
            f"got {n_records}"
        )
    record_duration = _to_float(_text(buffer, 244, 8), "record duration", where)
    if record_duration <= 0:
        raise EdfFormatError(
            f"{where}: header field record duration must be positive, "
            f"got {record_duration}"
        )
    signals = _parse_signals(buffer, n_signals, where)
    return EdfHeader(
        version=version,
        patient_field=_text(buffer, 8, 80),
        recording_field=_text(buffer, 88, 80),
        start=_parse_start(_text(buffer, 168, 8), _text(buffer, 176, 8), where),
        header_bytes=header_bytes,
        reserved=_text(buffer, 192, 44),
        n_records=n_records,
        record_duration=record_duration,
        signals=signals,
    )


def _parse_signals(
    buffer: bytes, n_signals: int, where: str
) -> tuple[EdfSignalHeader, ...]:
    """Parse the field-major per-signal blocks."""
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    columns: list[list[str]] = []
    position = FIXED_HEADER_BYTES
    for width in widths:
        columns.append(
            [
                _text(buffer, position + index * width, width)
                for index in range(n_signals)
            ]
        )
        position += width * n_signals
    labels, transducers, units, pmins, pmaxes, dmins, dmaxes, prefilters = (
        columns[:8]
    )
    samples = columns[8]
    signals: list[EdfSignalHeader] = []
    for index in range(n_signals):
        label = labels[index]
        name = f"signal {index}"
        physical_min = _to_float(pmins[index], f"{name} physical min", where)
        physical_max = _to_float(pmaxes[index], f"{name} physical max", where)
        digital_min = _to_int(dmins[index], f"{name} digital min", where)
        digital_max = _to_int(dmaxes[index], f"{name} digital max", where)
        if digital_min == digital_max:
            raise EdfFormatError(
                f"{where}: signal {index} digital min equals digital max "
                f"({digital_min})"
            )
        samples_per_record = _to_int(
            samples[index], f"{name} samples per record", where
        )
        if samples_per_record < 1:
            raise EdfFormatError(
                f"{where}: signal {index} samples per record must be at "
                f"least 1, got {samples_per_record}"
            )
        signals.append(
            EdfSignalHeader(
                label=label,
                transducer=transducers[index],
                unit=units[index],
                physical_min=physical_min,
                physical_max=physical_max,
                digital_min=digital_min,
                digital_max=digital_max,
                prefilter=prefilters[index],
                samples_per_record=samples_per_record,
            )
        )
    return tuple(signals)


def read_header(path: Path) -> EdfHeader:
    """Read and parse the header of the EDF file at the path.

    Only the header bytes are read.

    Raises:
        EdfFormatError: The header is malformed or the file is too short.
        OSError: The file cannot be opened.
    """
    where = str(path)
    with path.open("rb") as handle:
        fixed = handle.read(FIXED_HEADER_BYTES)
        n_signals = _signal_count(fixed, where)
        rest = handle.read(SIGNAL_HEADER_BYTES * n_signals)
    return parse_header(fixed + rest, where=where)
