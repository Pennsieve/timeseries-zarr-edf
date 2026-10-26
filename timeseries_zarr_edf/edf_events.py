"""Event channel over the annotations of an EDF+ file."""

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from timeseries_zarr import MAX_LABELS, EventColumn, text_width

from timeseries_zarr_edf.edf_file import EdfAnnotation
from timeseries_zarr_edf.edf_header import ANNOTATION_LABEL

ANNOTATIONS_CHANNEL_ID: Final = "annotations"
"""Id and name of the event channel that holds every annotation."""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting annotation seconds."""


class EdfAnnotationSource:
    """An event channel holding every annotation of one EDF+ file.

    Each annotation text is one event. Events are sorted by time, stably, so
    annotations with equal times keep file order. The channel id and name are
    ANNOTATIONS_CHANNEL_ID, and its unit is None. The text column always
    exists. The durations column exists when any annotation carries a
    duration. The labels column exists when the distinct texts number at most
    MAX_LABELS. The channels and waveform_mean columns do not exist.
    """

    def __init__(
        self,
        annotations: Sequence[EdfAnnotation],
        *,
        start_us: int,
        flavor: str,
    ) -> None:
        """Bind annotations as an event channel.

        start_us is the microseconds since the Unix epoch of the header start,
        which every annotation onset counts from. flavor is the EDF flavor,
        for the provenance. NUL is removed from every text. Raises ValueError
        if a duration is negative or an onset is not finite.
        """
        onsets = np.array([a.onset_s for a in annotations], dtype=np.float64)
        if not np.isfinite(onsets).all():
            raise ValueError("an annotation onset is not a finite number")
        order = np.argsort(onsets, kind="stable")
        offsets = np.rint(onsets * MICROSECONDS_PER_SECOND).astype(np.int64)
        self._events = (start_us + offsets)[order]
        texts = [a.text.replace("\x00", "") for a in annotations]
        self._texts = [texts[i] for i in order]
        self._durations: npt.NDArray[np.int64] | None = None
        if any(a.duration_s is not None for a in annotations):
            seconds = np.array(
                [a.duration_s or 0.0 for a in annotations], dtype=np.float64
            )
            if not np.isfinite(seconds).all() or (seconds < 0).any():
                raise ValueError(
                    "an annotation duration is negative or not finite"
                )
            lengths = np.rint(seconds * MICROSECONDS_PER_SECOND)
            self._durations = lengths.astype(np.int64)[order]
        self._labels: npt.NDArray[np.uint16] | None = None
        self._label_names: list[str] = []
        distinct = sorted(set(self._texts))
        if distinct and len(distinct) <= MAX_LABELS:
            index = {text: i for i, text in enumerate(distinct)}
            self._labels = np.array(
                [index[text] for text in self._texts], dtype=np.uint16
            )
            self._label_names = distinct
        self._provenance: dict[str, object] = {
            "format": flavor,
            "signal": ANNOTATION_LABEL,
        }

    @property
    def id(self) -> str:
        """The channel id, ANNOTATIONS_CHANNEL_ID."""
        return ANNOTATIONS_CHANNEL_ID

    @property
    def name(self) -> str:
        """The channel name, the same value as id."""
        return ANNOTATIONS_CHANNEL_ID

    @property
    def unit(self) -> str | None:
        """Always None: annotations have no measurement."""
        return None

    def provenance(self) -> dict[str, object]:
        """Return the EDF flavor and the label of the annotation signal."""
        return dict(self._provenance)

    def num_events(self) -> int:
        """Return the number of events."""
        return int(self._events.shape[0])

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns the channel provides.

        A channel with no events offers none.
        """
        if not self.num_events():
            return frozenset()
        present: set[EventColumn] = {"text"}
        if self._durations is not None:
            present.add("durations")
        if self._labels is not None:
            present.add("labels")
        return frozenset(present)

    def label_names(self) -> list[str] | None:
        """Return the sorted distinct texts, or None without labels."""
        return list(self._label_names) if self._labels is not None else None

    def num_labels(self) -> int:
        """Return the number of distinct texts, or 0 without labels."""
        return len(self._label_names)

    def max_duration_us(self) -> int | None:
        """Return the longest duration, or None without durations or events."""
        if self._durations is None or not self.num_events():
            return None
        return int(self._durations.max())

    def text_width(self) -> int:
        """Return the byte width of the longest text."""
        return text_width(self._texts)

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of event times in microseconds."""
        return self._events[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds.

        Raises:
            ValueError: No annotation carries a duration.
        """
        if self._durations is None:
            raise ValueError("no annotation has a duration")
        return self._durations[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label indices.

        Raises:
            ValueError: The distinct texts outnumber MAX_LABELS.
        """
        if self._labels is None:
            raise ValueError("the annotations have no labels column")
        return self._labels[start:stop]

    def read_text(self, start: int, stop: int) -> list[str]:
        """Return the [start, stop) window of texts."""
        return self._texts[start:stop]

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Raise ValueError, because annotations apply to the recording."""
        raise ValueError("the annotations have no channels column")

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Raise ValueError, because annotations have no snippets."""
        raise ValueError("the annotations have no waveform_mean column")

    def waveform_period_us(self) -> float:
        """Raise ValueError, because annotations have no snippets."""
        raise ValueError("the annotations have no waveform_mean column")
