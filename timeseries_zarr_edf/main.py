"""Command-line entry point: EDF file in, bundle out."""

import logging
import os
import sys
from collections.abc import Sequence

from timeseries_zarr import write_bundle

from timeseries_zarr_edf.config import load_config
from timeseries_zarr_edf.edf_file import EdfFile
from timeseries_zarr_edf.edf_reader import build_sources_from_edf

logger = logging.getLogger(__name__)


def main(argv: Sequence[str]) -> int:
    """Run one bundle build from the command line and return an exit code.

    Resolves a Config from the process environment and argv, then builds the
    bundle for the EDF file it names. Returns 0 on success, 2 on a bad
    invocation, 1 on a failed build.
    """
    try:
        cfg = load_config(os.environ, argv)
    except ValueError:
        logger.exception("invalid invocation")
        return 2

    try:
        # The sources read from the open file, so the whole write stays
        # inside the context manager.
        with EdfFile.open(cfg.edf_path) as edf:
            sources = build_sources_from_edf(
                edf,
                timezone=cfg.timezone,
                subject_fields=cfg.subject_fields,
                max_sweep_channels=cfg.sweep_channels,
            )
            logger.info(
                "writing %d continuous + %d event channels to %s",
                sum(len(source.channels()) for source in sources.continuous),
                len(sources.events),
                cfg.output_path,
            )
            write_bundle(
                sources.continuous,
                sources.events,
                output_path=cfg.output_path,
                opts=cfg.opts,
                recording=sources.recording,
            )
    except (OSError, ValueError):
        logger.exception("failed to build bundle from %s", cfg.edf_path)
        return 1

    logger.info("wrote bundle to %s", cfg.output_path)
    return 0


def run() -> None:
    """Run main on sys.argv[1:] with INFO logging and exit with its code."""
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
