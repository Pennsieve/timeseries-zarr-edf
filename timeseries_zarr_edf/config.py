"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from timeseries_zarr import (
    MAX_LEVELS,
    WINDOW_BATCH_BYTES,
    WriteOpts,
    channel_writer_bytes,
)

from timeseries_zarr_edf.edf_reader import WINDOW_BYTES

MAX_ZSTD_LEVEL: Final = 22
"""Highest Zstd compression level the codec accepts."""

DEFAULT_SWEEP_BYTES: Final = 2 * 2**30
"""Default bytes of memory for the writers and windows of one group of channels."""

WINDOW_RESERVE_BYTES: Final = 2 * (WINDOW_BATCH_BYTES + WINDOW_BYTES)
"""Bytes the sweep reserves for the two batches of windows that the writer holds."""

SUBJECT_FIELDS: Final = ("code", "sex", "birthdate", "name")
"""EDF+ patient sub-fields that ZARR_WRITER_SUBJECT_FIELDS may name."""

DEFAULT_SUBJECT_FIELDS: Final = frozenset({"code", "sex"})
"""Patient sub-fields written to recording.json when none are chosen."""


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    Attributes:
        edf_path: Input EDF file.
        output_path: Bundle directory to write.
        opts: Writer settings.
        timezone: Zone that places the EDF start time, which has no zone.
        subject_fields: Patient sub-fields written to recording.json.
        sweep_bytes: Bound on the memory of the channel writers and the two
            batches of windows while one group of channels is written.
    """

    edf_path: Path
    output_path: Path
    opts: WriteOpts
    timezone: ZoneInfo
    subject_fields: frozenset[str] = DEFAULT_SUBJECT_FIELDS
    sweep_bytes: int = DEFAULT_SWEEP_BYTES

    @property
    def sweep_channels(self) -> int:
        """The most channels whose writers fit in sweep_bytes, at least 1.

        The budget is sweep_bytes less WINDOW_RESERVE_BYTES. Each channel
        costs channel_writer_bytes under opts.
        """
        budget = self.sweep_bytes - WINDOW_RESERVE_BYTES
        return max(1, budget // channel_writer_bytes(self.opts))


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: the input EDF path and
    the output path. The writer settings come from the ZARR_WRITER_ prefixed
    variables; unset settings fall back to the WriteOpts defaults.
    ZARR_WRITER_ZSTD_LEVEL takes 0 through MAX_ZSTD_LEVEL.
    ZARR_WRITER_MAX_LEVELS takes 0 through MAX_LEVELS, and 0 writes raw
    samples only. ZARR_WRITER_MIN_BINS, ZARR_WRITER_INNER_LEN,
    ZARR_WRITER_TARGET_SHARD_BYTES, ZARR_WRITER_SWEEP_BYTES, and
    ZARR_WRITER_MAX_WORKERS take 1 or more. ZARR_WRITER_SWEEP_BYTES defaults
    to DEFAULT_SWEEP_BYTES, and ZARR_WRITER_MAX_WORKERS to the CPUs available.
    ZARR_WRITER_VAR takes 0 or 1. ZARR_WRITER_TIMEZONE is required and takes an
    IANA zone name. ZARR_WRITER_SUBJECT_FIELDS takes a comma-separated list of
    SUBJECT_FIELDS names, defaults to DEFAULT_SUBJECT_FIELDS, and an empty
    value writes no patient sub-field. Raises ValueError if argv does not hold
    exactly two entries, if ZARR_WRITER_TIMEZONE is unset, or if a setting lies
    outside its range.
    """
    edf_path, output_path = _resolve_paths(argv)

    defaults = WriteOpts()
    max_workers = (
        _int_env(env, "ZARR_WRITER_MAX_WORKERS", 1, 1)
        if "ZARR_WRITER_MAX_WORKERS" in env
        else defaults.max_workers
    )
    opts = WriteOpts(
        zstd_level=_int_env(
            env,
            "ZARR_WRITER_ZSTD_LEVEL",
            defaults.zstd_level,
            0,
            MAX_ZSTD_LEVEL,
        ),
        max_levels=_int_env(
            env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels, 0, MAX_LEVELS
        ),
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins, 1),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len, 1),
        target_shard_bytes=_int_env(
            env,
            "ZARR_WRITER_TARGET_SHARD_BYTES",
            defaults.target_shard_bytes,
            1,
        ),
        write_var=_flag_env(env, "ZARR_WRITER_VAR", defaults.write_var),
        max_workers=max_workers,
    )

    return Config(
        edf_path=edf_path,
        output_path=output_path,
        opts=opts,
        timezone=_timezone_env(env),
        subject_fields=_subject_fields_env(env),
        sweep_bytes=_int_env(
            env, "ZARR_WRITER_SWEEP_BYTES", DEFAULT_SWEEP_BYTES, 1
        ),
    )


def _resolve_paths(argv: Sequence[str]) -> tuple[Path, Path]:
    """Return the input EDF path and the output path from argv.

    Raises ValueError unless argv holds exactly those two entries.
    """
    positional_count = 2
    if len(argv) != positional_count:
        raise ValueError(
            "expected the input EDF path and the output path as two "
            f"arguments, got {len(argv)}"
        )
    return Path(argv[0]), Path(argv[1])


def _timezone_env(env: Mapping[str, str]) -> ZoneInfo:
    """Return the zone named by ZARR_WRITER_TIMEZONE.

    Raises ValueError if the variable is unset or its value is not an IANA
    zone name.
    """
    raw = env.get("ZARR_WRITER_TIMEZONE")
    if raw is None:
        raise ValueError(
            "ZARR_WRITER_TIMEZONE must be set to an IANA zone name such as "
            "UTC or America/New_York"
        )
    try:
        return ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(
            f"ZARR_WRITER_TIMEZONE must be an IANA zone name, got {raw!r}"
        ) from None


def _subject_fields_env(env: Mapping[str, str]) -> frozenset[str]:
    """Return the fields named by ZARR_WRITER_SUBJECT_FIELDS.

    An unset variable gives DEFAULT_SUBJECT_FIELDS and an empty one gives no
    field. Raises ValueError if an entry is not in SUBJECT_FIELDS.
    """
    key = "ZARR_WRITER_SUBJECT_FIELDS"
    raw = env.get(key)
    if raw is None:
        return DEFAULT_SUBJECT_FIELDS
    if not raw.strip():
        return frozenset()
    entries = [entry.strip() for entry in raw.split(",")]
    for entry in entries:
        if entry not in SUBJECT_FIELDS:
            raise ValueError(
                f"{key} entries must be among {', '.join(SUBJECT_FIELDS)}, "
                f"got {entry!r}"
            )
    return frozenset(entries)


def _int_env(
    env: Mapping[str, str],
    key: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return the integer env value for key, or default if it is unset.

    The value must lie in minimum through maximum inclusive. A maximum of
    None leaves it unbounded above. Raises ValueError if the value is present
    but is not an integer or lies outside that range.
    """
    raw = env.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None
    if maximum is None and value < minimum:
        raise ValueError(f"{key} must be {minimum} or more, got {value}")
    if maximum is not None and not minimum <= value <= maximum:
        raise ValueError(f"{key} must be {minimum} to {maximum}, got {value}")
    return value


def _flag_env(env: Mapping[str, str], key: str, default: bool) -> bool:
    """Return the flag env value for key, or default if it is unset.

    A flag is "0" for off or "1" for on. Raises ValueError for any other
    value, including an empty one.
    """
    raw = env.get(key)
    if raw is None:
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{key} must be 0 or 1, got {raw!r}")
    return raw == "1"
