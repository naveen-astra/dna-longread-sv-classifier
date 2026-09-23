"""Entry point that runs truvari with a working `compress_index_vcf`.

Root cause (isolated by direct testing, 2026-09-22/23, WSL2/Ubuntu 24.04): both
`truvari bench` and `truvari collapse` call `truvari.compress_index_vcf()` to bgzip and
tabix-index their output VCFs at the end of a run. That function calls
`pysam.bcftools.sort()` -- pysam's own ctypes-bundled bcftools -- which crashes in this
environment with a garbled "Permission denied: Unknown error <garbage>" message. This
was confirmed reproducible across truvari 4.2.2, 4.3.1 and 5.4.0, and across pysam
0.22.1 and 0.24.1, so it is not a version-specific truvari or pysam bug; it is specific
to calling pysam's bundled bcftools this way under this environment.

The actual variant-comparison engine is unaffected: a synthetic test with a real,
spec-compliant candidate produced a correct match (TruScore, PctSizeSimilarity, etc.)
before the finalization step crashed. This module replaces only that one function with
an equivalent that shells out to the system `bcftools`/`bgzip`/`tabix` binaries, which
work correctly, and then runs truvari's normal CLI entry point.

Usage: `python -m svml._truvari_entry bench -b ... -c ... -o ...` (or `collapse ...`),
in place of calling `truvari` directly. `svml.labels.match.run_bench` and
`svml.candidates.merge.run_truvari_collapse` both invoke it this way.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _fixed_compress_index_vcf(fn: str, fout: str | None = None, remove: bool = True) -> None:
    if fout is None:
        fout = fn + ".gz"

    # bcftools sort's --temp-dir is a literal prefix it appends random characters to
    # (mkdtemp-style), not a directory to create inside. Without a trailing slash,
    # "/tmp" + "XXXXXX" resolves to a sibling of /tmp itself and fails trying to mkdir
    # directly under /. This was the second, distinct bug found while fixing the first.
    temp_dir_prefix = tempfile.gettempdir().rstrip("/") + "/"
    sorted_tmp = tempfile.mktemp(suffix=".vcf", dir=os.path.dirname(fn) or ".")

    proc = subprocess.run(
        ["bcftools", "sort", "-o", sorted_tmp, "--temp-dir", temp_dir_prefix, fn],
        check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bcftools sort failed ({proc.returncode}): {proc.stderr}")

    with open(fout, "wb") as out_fh:
        bgzip = subprocess.run(["bgzip", "-c", sorted_tmp], stdout=out_fh, stderr=subprocess.PIPE)
    if bgzip.returncode != 0:
        raise RuntimeError(f"bgzip failed ({bgzip.returncode}): {bgzip.stderr.decode()}")

    tabix = subprocess.run(
        ["tabix", "-f", "-p", "vcf", fout], check=False, capture_output=True, text=True
    )
    if tabix.returncode != 0:
        raise RuntimeError(f"tabix failed ({tabix.returncode}): {tabix.stderr}")

    if remove and os.path.exists(fn):
        os.remove(fn)
    if os.path.exists(sorted_tmp):
        os.remove(sorted_tmp)


def _apply_patch() -> None:
    import truvari

    truvari.compress_index_vcf = _fixed_compress_index_vcf
    try:
        import truvari.utils

        truvari.utils.compress_index_vcf = _fixed_compress_index_vcf
    except ImportError:
        pass


def main() -> int:
    _apply_patch()
    from truvari.__main__ import main as truvari_main

    return truvari_main()


if __name__ == "__main__":
    sys.exit(main())
