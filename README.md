# timeseries-zarr-edf

Converts an EDF, EDF+C, or EDF+D recording into a TS-Zarr bundle, a static Zarr
v3 hierarchy that a browser reads over HTTP range requests to render time series
at any zoom level, from the whole recording down to single samples.

This package is an EDF input adapter for
[timeseries-zarr](https://github.com/Pennsieve/timeseries-zarr-py), which
specifies the format and writes the bundle. It parses the EDF header, maps the
data records, and runs the TS-Zarr bundle writer.

## Installation

```bash
pip install git+https://github.com/Pennsieve/timeseries-zarr-edf.git
```

The reader uses `os.pread`, so the converter runs on Linux and macOS and not on Windows.

## Usage

```bash
ZARR_WRITER_TIMEZONE=America/New_York timeseries-zarr-edf recording.edf /data/recording.zarr
```

`ZARR_WRITER_TIMEZONE=America/New_York python -m timeseries_zarr_edf` runs the same command.

The exit code is 0 on success, 2 for a bad invocation or setting,
and 1 for a failed conversion.

The output path must be absent, empty, or hold a bundle from an earlier run,
which the converter replaces.

## How an EDF+ file is converted

Every data signal becomes a continuous channel at its own sample rate.

The channel id is the signal's position in the file and the channel name is the
signal's label.

Voltage samples are stored in microvolts. A signal in any other unit retains its unit.

EDF+ annotations become one event channel named `annotations` with the text, the
duration, and a label for each distinct text. A file without annotations gets
no event channel.

Each segment of an EDF+D file is one run. The converter places a run at the
grid sample nearest its onset and fills the pause before it with NaN. A
channel of two or more runs gets a `runs` array that holds the exact onset of
each run in microseconds from the reference time. A segment may start at any
time, including one that is not a multiple of the sample period.

Each channel's provenance holds the file variant (EDF, EDF+C, or EDF+D),
the signal index, the physical and digital ranges, and the transducer and
prefilter text when present.

## Time

The EDF header stores the recording's start date and time without a time zone.
TS-Zarr requires timezone-aware datetimes. The converter reads the start in the
time zone named by `ZARR_WRITER_TIMEZONE`, an IANA name such as `America/New_York`.

The reference time of the bundle is the start date and time in the EDF header.
Each channel's `offset_us` holds the onset of the first data record, in
microseconds after the header start. A plain EDF file has no record onsets, so
its offset is zero. An EDF+ file states the onset of each record, and the first
record can start after the header start.

## Identity

The converter writes the subject, the session, the session start time, and the
reference time to `recording.json` at the bundle root and to no other metadata.
`ZARR_WRITER_SUBJECT_FIELDS` selects which patient sub-fields the subject holds.

The converter writes its own name and version, the writer, the file variant
(EDF, EDF+C, or EDF+D), the time zone, and the equipment code of an EDF+ file
to the `source` object of `recording.json`.

## Settings

Writer settings come from the environment under the `ZARR_WRITER_` prefix.
Every setting but `ZARR_WRITER_TIMEZONE` is optional.

| Variable | Governs | Range | Default |
|---|---|---|---|
| `ZARR_WRITER_TIMEZONE` | IANA zone of the header start | zone name | required |
| `ZARR_WRITER_SUBJECT_FIELDS` | patient sub-fields written to the subject | comma-separated `code`, `sex`, `birthdate`, `name`; empty for none | `code,sex` |
| `ZARR_WRITER_ZSTD_LEVEL` | Zstd compression level | 0 to 22 | 5 |
| `ZARR_WRITER_MAX_LEVELS` | maximum number of levels above the raw samples; 0 writes raw samples only | 0 to 15 | 15 |
| `ZARR_WRITER_MIN_BINS` | bin threshold for keeping a coarser level | 1 or more | 1024 |
| `ZARR_WRITER_INNER_LEN` | inner Zarr chunk length in samples | 1 or more | 8192 |
| `ZARR_WRITER_TARGET_SHARD_BYTES` | target outer shard size in bytes | 1 or more | 16 MiB |
| `ZARR_WRITER_VAR` | write the per-bin variance member | 0 or 1 | 0 |
| `ZARR_WRITER_MAX_WORKERS` | threads that fold and encode the channels of a window | 1 or more | available CPUs |
| `ZARR_WRITER_SWEEP_BYTES` | memory in bytes for the channel writers and their windows during one pass over the records | 1 or more | 2 GiB |

`ZARR_WRITER_VAR` sets the library's `write_var` option.
[The library README](https://github.com/Pennsieve/timeseries-zarr-py#usage) describes what
it writes and what it costs in memory.

## Memory

A pass over the records holds one writer per channel plus the windows of float32
samples decoded from at most 8 MiB of records. The converter makes one pass per
group of channels whose writers fit in `ZARR_WRITER_SWEEP_BYTES`, after it reserves
144 MiB for the windows. Each group re-reads the file. Each channel writer holds up
to 5.8 MiB at the defaults, so a group holds 329 channels and a file of that width
or less is read once.

## Known limitations

- BDF (24-bit BioSemi) is not supported.
- The `label_names` of the annotations channel come directly from the EDF+
  annotation text, without normalization or categorization.

## Development

Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset.

Runtime dependencies are `timeseries-zarr` and `numpy`.

```bash
make venv        # create the virtualenv and install this package with the library from PyPI
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # ruff check + format check + mypy + pytest
make pre-commit  # install the git hook
```

## License

Apache-2.0. See [LICENSE](LICENSE).
