"""Tests for the EDF header parser."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import SignalSpec, build_edf
from timeseries_zarr_edf.edf_header import (
    ANNOTATION_LABEL,
    EdfFormatError,
    EdfSignalHeader,
    fold_unit,
    parse_header,
    read_header,
)

START = datetime(2021, 3, 4, 5, 6, 7)

SIGNALS: list[SignalSpec] = [
    ("Fp1", "uV", -500.0, 500.0, -2048, 2047, 4),
    ("Temp", "degC", 30.0, 40.0, 0, 1000, 2),
]


def make_file(tmp_path: Path, **kwargs: object) -> Path:
    """Write a two-signal file and return its path."""
    options: dict[str, object] = {
        "signals": SIGNALS,
        "records": [
            [np.zeros(4, dtype=np.int16), np.zeros(2, dtype=np.int16)]
            for _ in range(3)
        ],
        "start": START,
    }
    options.update(kwargs)
    return build_edf(tmp_path / "x.edf", **options)  # type: ignore[arg-type]


def make_bytes(tmp_path: Path, **kwargs: object) -> bytearray:
    """Return the bytes of a synthetic file."""
    return bytearray(make_file(tmp_path, **kwargs).read_bytes())


def put(buffer: bytearray, offset: int, width: int, text: str) -> None:
    """Overwrite one header field with space-padded text."""
    buffer[offset : offset + width] = text.ljust(width).encode("latin-1")


def test_parses_every_fixed_field(tmp_path: Path) -> None:
    header = parse_header(
        make_bytes(
            tmp_path,
            patient_field="P1 F 02-MAR-1980 Jane",
            recording_field="Startdate 04-MAR-2021 A1 T1 EQ1",
            record_duration=2.0,
        )
    )
    assert header.version == "0"
    assert header.patient_field == "P1 F 02-MAR-1980 Jane"
    assert header.recording_field == "Startdate 04-MAR-2021 A1 T1 EQ1"
    assert header.start == START
    assert header.header_bytes == 256 * 4
    assert header.reserved == "EDF+C"
    assert header.n_records == 3
    assert header.record_duration == 2.0
    assert len(header.signals) == 3


def test_parses_per_signal_fields_field_major(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path))
    assert header.signals[0] == EdfSignalHeader(
        label="Fp1",
        transducer="",
        unit="uV",
        physical_min=-500.0,
        physical_max=500.0,
        digital_min=-2048,
        digital_max=2047,
        prefilter="",
        samples_per_record=4,
    )
    assert header.signals[1].label == "Temp"
    assert header.signals[1].unit == "degC"
    assert header.signals[1].physical_min == 30.0
    assert header.signals[1].digital_max == 1000
    assert header.signals[1].samples_per_record == 2
    assert header.signals[2].label == ANNOTATION_LABEL


@pytest.mark.parametrize(
    ("year_text", "year"),
    [("84", 2084), ("85", 1985), ("99", 1999), ("00", 2000)],
)
def test_pivots_two_digit_years_at_85(
    tmp_path: Path, year_text: str, year: int
) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 168, 8, f"04.03.{year_text}")
    assert parse_header(bytes(buffer)).start.year == year


def test_accepts_alternate_date_and_time_separators(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 168, 8, "04-03-21")
    put(buffer, 176, 8, "05:06:07")
    assert parse_header(bytes(buffer)).start == START
    put(buffer, 168, 8, "04/03/21")
    put(buffer, 176, 8, "05-06-07")
    assert parse_header(bytes(buffer)).start == START


def test_rejects_unparseable_start_date(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 168, 8, "31.02.21")
    with pytest.raises(EdfFormatError, match="out of range"):
        parse_header(bytes(buffer), where="x.edf")
    put(buffer, 168, 8, "garbage")
    with pytest.raises(EdfFormatError, match="start date"):
        parse_header(bytes(buffer), where="x.edf")


@pytest.mark.parametrize("bad", ["31.04.23", "1.4.2023"])
def test_start_date_errors_hold_neither_the_date_nor_the_year(
    tmp_path: Path, bad: str
) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 168, 8, bad)
    with pytest.raises(EdfFormatError) as raised:
        parse_header(bytes(buffer), where="x.edf")
    assert bad not in str(raised.value)
    assert "2023" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None or raised.value.__suppress_context__


def test_rejects_unparseable_start_time(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 176, 8, "noon")
    with pytest.raises(EdfFormatError, match="start time"):
        parse_header(bytes(buffer))


def test_rejects_a_non_zero_version(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    buffer[0:8] = b"\xffBIOSEMI"
    with pytest.raises(EdfFormatError, match="version"):
        parse_header(bytes(buffer))


def test_rejects_header_bytes_that_disagree_with_signal_count(
    tmp_path: Path,
) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 184, 8, "768")
    with pytest.raises(EdfFormatError, match="header bytes"):
        parse_header(bytes(buffer))


def test_rejects_a_buffer_shorter_than_the_header(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    with pytest.raises(EdfFormatError, match="shorter"):
        parse_header(bytes(buffer[:600]))
    with pytest.raises(EdfFormatError, match="shorter"):
        parse_header(bytes(buffer[:100]))


def test_rejects_zero_signals(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 252, 4, "0")
    with pytest.raises(EdfFormatError, match="number of signals"):
        parse_header(bytes(buffer))


def test_rejects_a_non_numeric_field(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 236, 8, "many")
    with pytest.raises(EdfFormatError, match="number of records"):
        parse_header(bytes(buffer))


def test_rejects_a_non_positive_record_duration(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 244, 8, "0")
    with pytest.raises(EdfFormatError, match="record duration"):
        parse_header(bytes(buffer))


def test_rejects_a_record_count_below_minus_one(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 236, 8, "-2")
    with pytest.raises(EdfFormatError, match="number of records"):
        parse_header(bytes(buffer))


def test_accepts_minus_one_records(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path, n_records_field=-1))
    assert header.n_records == -1


def test_rejects_zero_samples_per_record(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    samples_offset = 256 + (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80) * 3
    put(buffer, samples_offset, 8, "0")
    with pytest.raises(EdfFormatError, match="samples per record"):
        parse_header(bytes(buffer))


def test_rejects_a_zero_width_digital_range(tmp_path: Path) -> None:
    signals: list[SignalSpec] = [("A", "uV", -1.0, 1.0, 5, 5, 2)]
    buffer = make_bytes(
        tmp_path, signals=signals, records=[[np.zeros(2, dtype=np.int16)]]
    )
    with pytest.raises(EdfFormatError, match="digital min equals"):
        parse_header(bytes(buffer))


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "abc"])
def test_rejects_a_non_finite_physical_range(tmp_path: Path, bad: str) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 256 + (16 + 80 + 8) * 3, 8, bad)
    with pytest.raises(EdfFormatError, match="physical min"):
        parse_header(bytes(buffer))


def test_accepts_an_integer_field_written_as_a_float(tmp_path: Path) -> None:
    buffer = make_bytes(tmp_path)
    put(buffer, 236, 8, "3.0")
    assert parse_header(bytes(buffer)).n_records == 3
    put(buffer, 236, 8, "3.5")
    with pytest.raises(EdfFormatError, match="integer"):
        parse_header(bytes(buffer))


@pytest.mark.parametrize(
    ("flavor", "expected", "edfplus", "discontinuous"),
    [
        ("EDF", "EDF", False, False),
        ("EDF+C", "EDF+C", True, False),
        ("EDF+D", "EDF+D", True, True),
    ],
)
def test_flavor_follows_the_reserved_field(
    tmp_path: Path,
    flavor: str,
    expected: str,
    edfplus: bool,
    discontinuous: bool,
) -> None:
    header = parse_header(make_bytes(tmp_path, flavor=flavor))
    assert header.flavor == expected
    assert header.is_edfplus is edfplus
    assert header.is_discontinuous is discontinuous


def test_annotation_indices_find_every_annotation_signal(
    tmp_path: Path,
) -> None:
    signals: list[SignalSpec] = [
        (ANNOTATION_LABEL, "", -1.0, 1.0, -32768, 32767, 4),
        ("A", "uV", -1.0, 1.0, -10, 10, 2),
    ]
    records = [[np.zeros(4, dtype=np.int16), np.zeros(2, dtype=np.int16)]]
    header = parse_header(
        make_bytes(tmp_path, signals=signals, records=records)
    )
    assert header.annotation_indices == (0, 2)


def test_data_indices_exclude_annotation_signals(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path))
    assert header.data_indices == (0, 1)


def test_record_geometry_sums_every_signal(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path))
    spr_annotation = header.signals[2].samples_per_record
    assert header.record_samples == 4 + 2 + spr_annotation
    assert header.record_bytes == 2 * header.record_samples
    assert header.sample_offset(0) == 0
    assert header.sample_offset(1) == 4
    assert header.sample_offset(2) == 6


def test_rate_divides_by_the_record_duration(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path, record_duration=0.5))
    assert header.signals[0].rate_hz(header.record_duration) == 8.0
    assert header.signals[1].rate_hz(header.record_duration) == 4.0


def test_gain_and_offset_for_a_normal_range() -> None:
    signal = EdfSignalHeader("A", "", "uV", -100.0, 100.0, 0, 200, "", 1)
    assert signal.gain == 1.0
    assert signal.offset == -100.0


def test_gain_and_offset_for_an_inverted_range() -> None:
    signal = EdfSignalHeader("A", "", "uV", 100.0, -100.0, -10, 10, "", 1)
    assert signal.gain == -10.0
    assert signal.offset == 0.0


def test_patient_sub_fields_for_the_full_form(tmp_path: Path) -> None:
    header = parse_header(
        make_bytes(tmp_path, patient_field="PX7 F 02-MAR-1980 Jane Q Doe")
    )
    patient = header.patient()
    assert patient is not None
    assert patient.code == "PX7"
    assert patient.sex == "F"
    assert patient.birthdate == date(1980, 3, 2)
    assert patient.name == "Jane Q Doe"


def test_patient_sub_fields_for_the_all_unknown_form(tmp_path: Path) -> None:
    patient = parse_header(make_bytes(tmp_path)).patient()
    assert patient is not None
    assert patient.code == ""
    assert patient.sex == ""
    assert patient.birthdate is None
    assert patient.name == ""


def test_patient_sex_other_than_m_or_f_is_unknown(tmp_path: Path) -> None:
    patient = parse_header(
        make_bytes(tmp_path, patient_field="PX7 U X X")
    ).patient()
    assert patient is not None
    assert patient.sex == ""
    assert patient.code == "PX7"


def test_patient_birthdate_pivots_a_two_digit_year(tmp_path: Path) -> None:
    patient = parse_header(
        make_bytes(tmp_path, patient_field="X M 01-jan-90 X")
    ).patient()
    assert patient is not None
    assert patient.sex == "M"
    assert patient.birthdate == date(1990, 1, 1)


def test_patient_with_a_free_text_field_has_unknown_sub_fields(
    tmp_path: Path,
) -> None:
    patient = parse_header(make_bytes(tmp_path, patient_field="")).patient()
    assert patient is not None
    assert patient.code == ""
    assert patient.birthdate is None


def test_recording_sub_fields_with_startdate(tmp_path: Path) -> None:
    header = parse_header(
        make_bytes(
            tmp_path,
            recording_field="Startdate 04-MAR-2021 A1 T1 EQ one",
        )
    )
    recording = header.recording()
    assert recording is not None
    assert recording.startdate == date(2021, 3, 4)
    assert recording.admin_code == "A1"
    assert recording.technician == "T1"
    assert recording.equipment == "EQ one"


def test_recording_sub_fields_with_unknown_markers(tmp_path: Path) -> None:
    recording = parse_header(make_bytes(tmp_path)).recording()
    assert recording is not None
    assert recording.startdate is None
    assert recording.admin_code == ""
    assert recording.technician == ""
    assert recording.equipment == ""


def test_recording_without_startdate_has_no_sub_fields(
    tmp_path: Path,
) -> None:
    recording = parse_header(
        make_bytes(tmp_path, recording_field="Hospital room 4")
    ).recording()
    assert recording is not None
    assert recording.startdate is None
    assert recording.admin_code == ""
    assert recording.equipment == ""


def test_patient_and_recording_are_none_for_plain_edf(tmp_path: Path) -> None:
    header = parse_header(make_bytes(tmp_path, flavor="EDF"))
    assert header.patient() is None
    assert header.recording() is None


@pytest.mark.parametrize(
    ("unit", "folded"),
    [
        ("\u00b5V", "uV"),
        ("\u03bcV", "uV"),
        ("\u00c2\u00b5V", "uV"),
        ("  mV ", "mV"),
        ("%", "%"),
        ("bpm", "bpm"),
        ("", ""),
    ],
)
def test_fold_unit_maps_micro_signs_and_keeps_other_units(
    unit: str, folded: str
) -> None:
    assert fold_unit(unit) == folded


def test_read_header_reads_only_the_header_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_file(tmp_path)
    sizes: list[int] = []
    real_open = Path.open

    def spy(self: Path, *args: object, **kwargs: object) -> object:
        handle = real_open(self, *args, **kwargs)  # type: ignore[call-overload]
        real_read = handle.read

        def read(size: int = -1) -> bytes:
            data: bytes = real_read(size)
            sizes.append(len(data))
            return data

        handle.read = read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", spy)
    header = read_header(path)
    assert header.n_records == 3
    assert sum(sizes) == header.header_bytes


def test_read_header_rejects_a_short_file(tmp_path: Path) -> None:
    path = make_file(tmp_path)
    path.write_bytes(path.read_bytes()[:300])
    with pytest.raises(EdfFormatError, match="shorter"):
        read_header(path)
    path.write_bytes(b"0       ")
    with pytest.raises(EdfFormatError, match="fixed header"):
        read_header(path)
