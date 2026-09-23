"""BED interval loading and overlap queries.

Used for the confident-region gate (finding H4) and for the Group A annotation
features. Intervals are half-open [start, end), matching the BED convention.
"""

from __future__ import annotations

import bisect
import gzip
from collections import defaultdict
from pathlib import Path

from svml.utils.logging import get_logger

log = get_logger(__name__)


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_bed(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Load a BED file into chrom -> sorted, merged list of [start, end) intervals."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")

    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                raw[parts[0]].append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue

    merged = {chrom: _merge(intervals) for chrom, intervals in raw.items()}
    total = sum(len(v) for v in merged.values())
    log.info("loaded %s: %d merged intervals across %d contigs", path.name, total, len(merged))
    return merged


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping/adjacent intervals so overlap maths cannot double-count."""
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


class IntervalIndex:
    """Binary-search index over merged BED intervals.

    Merging at load time is what makes `overlap_bp` correct: overlapping source
    intervals (common in rmsk and segdup tracks) would otherwise be counted twice and
    produce overlap fractions above 1.0.
    """

    __slots__ = ("_by_chrom", "_starts", "name")

    def __init__(self, intervals: dict[str, list[tuple[int, int]]], name: str = "") -> None:
        self.name = name
        self._by_chrom = intervals
        self._starts = {chrom: [s for s, _ in ivs] for chrom, ivs in intervals.items()}

    @classmethod
    def from_bed(cls, path: str | Path, name: str = "") -> IntervalIndex:
        path = Path(path)
        return cls(load_bed(path), name=name or path.stem)

    def contigs(self) -> set[str]:
        return set(self._by_chrom)

    def _candidates(self, chrom: str, start: int, end: int) -> list[tuple[int, int]]:
        intervals = self._by_chrom.get(chrom)
        if not intervals:
            return []
        starts = self._starts[chrom]
        # First interval whose start is >= `end` cannot overlap; scan left from there.
        hi = bisect.bisect_left(starts, end)
        # Walk back far enough to catch a long interval that began well before `start`.
        lo = max(0, bisect.bisect_left(starts, start) - 1)
        return [iv for iv in intervals[lo:hi] if iv[1] > start and iv[0] < end]

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        if end <= start:
            end = start + 1
        return bool(self._candidates(chrom, start, end))

    def contains_point(self, chrom: str, pos: int) -> bool:
        return self.overlaps(chrom, pos, pos + 1)

    def overlap_bp(self, chrom: str, start: int, end: int) -> int:
        """Total bases of [start, end) covered by the index."""
        if end <= start:
            return 0
        return sum(
            min(end, iv_end) - max(start, iv_start)
            for iv_start, iv_end in self._candidates(chrom, start, end)
        )

    def overlap_fraction(self, chrom: str, start: int, end: int) -> float:
        """Covered fraction of [start, end). Zero-length spans return 0.0."""
        if end <= start:
            return 0.0
        return self.overlap_bp(chrom, start, end) / (end - start)

    def distance_to_nearest(self, chrom: str, pos: int, sentinel: int = 1_000_000_000) -> int:
        """Distance in bp from `pos` to the nearest interval; 0 if inside one."""
        intervals = self._by_chrom.get(chrom)
        if not intervals:
            return sentinel
        starts = self._starts[chrom]
        idx = bisect.bisect_right(starts, pos)

        best = sentinel
        if idx > 0:
            prev_start, prev_end = intervals[idx - 1]
            if prev_start <= pos < prev_end:
                return 0
            best = min(best, pos - prev_end + 1 if pos >= prev_end else 0)
        if idx < len(intervals):
            next_start, next_end = intervals[idx]
            if next_start <= pos < next_end:
                return 0
            best = min(best, next_start - pos)
        return max(0, int(best))


def check_contig_naming(*indexes: IntervalIndex, reference_contigs: set[str]) -> list[str]:
    """Return a list of human-readable warnings about contig-naming mismatches.

    Contig-naming mismatch (chr1 vs 1) is the most common silent failure in this kind
    of pipeline: every overlap query returns zero and the features look plausible but
    are all wrong. The Phase 2 exit test calls this.
    """
    problems: list[str] = []
    ref_has_chr = any(c.startswith("chr") for c in reference_contigs)

    for index in indexes:
        contigs = index.contigs()
        if not contigs:
            problems.append(f"{index.name}: no contigs loaded")
            continue
        idx_has_chr = any(c.startswith("chr") for c in contigs)
        if idx_has_chr != ref_has_chr:
            problems.append(
                f"{index.name}: contig naming mismatch "
                f"(index {'has' if idx_has_chr else 'lacks'} 'chr' prefix, "
                f"reference {'has' if ref_has_chr else 'lacks'} it)"
            )
        shared = contigs & reference_contigs
        if not shared:
            problems.append(
                f"{index.name}: no contigs shared with the reference "
                f"(index sample: {sorted(contigs)[:3]}, reference sample: "
                f"{sorted(reference_contigs)[:3]})"
            )
    return problems
