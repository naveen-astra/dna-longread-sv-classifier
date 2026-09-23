"""Regression tests for the truvari finalization workaround (finding D11).

`svml._truvari_entry` patches `truvari.compress_index_vcf`, which otherwise crashes in
this environment (see its module docstring and docs/06_execution_plan.md D11). These
tests exercise the patched function's logic directly, without needing truvari or a
real VCF -- the two bugs found were both in how the replacement invokes bcftools, not
in truvari itself, so that is what needs a regression guard.
"""

from __future__ import annotations

import gzip

import pytest

from svml._truvari_entry import _fixed_compress_index_vcf

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("bcftools") is None
    or __import__("shutil").which("bgzip") is None
    or __import__("shutil").which("tabix") is None,
    reason="requires bcftools/bgzip/tabix on PATH (WSL2/Linux toolchain)",
)


@pytest.fixture
def tiny_vcf(tmp_path):
    path = tmp_path / "in.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=248956422>\n"
        "##ALT=<ID=DEL,Description=\"Deletion\">\n"
        "##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"x\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1000\ta\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\n",
        encoding="utf-8",
    )
    return path


def test_compress_index_vcf_uses_a_trailing_slash_temp_dir(tiny_vcf, tmp_path):
    """Finding D11's second bug: bcftools sort --temp-dir needs a trailing slash, or
    it tries to mkdtemp a sibling of the temp dir itself and fails with permission
    denied. This is the regression that a plain `tempfile.gettempdir()` would reintroduce.
    """
    out = tmp_path / "out.vcf.gz"
    _fixed_compress_index_vcf(str(tiny_vcf), fout=str(out), remove=False)

    assert out.exists()
    with gzip.open(out, "rt") as fh:
        content = fh.read()
    assert "chr1\t1000\ta" in content
    assert out.with_suffix(out.suffix + ".tbi").exists() or (tmp_path / "out.vcf.gz.tbi").exists()


def test_compress_index_vcf_removes_input_when_asked(tiny_vcf, tmp_path):
    out = tmp_path / "out2.vcf.gz"
    _fixed_compress_index_vcf(str(tiny_vcf), fout=str(out), remove=True)
    assert not tiny_vcf.exists()
    assert out.exists()


def test_compress_index_vcf_raises_on_bad_input(tmp_path):
    missing = tmp_path / "does_not_exist.vcf"
    with pytest.raises(RuntimeError, match="bcftools sort failed"):
        _fixed_compress_index_vcf(str(missing), fout=str(tmp_path / "out3.vcf.gz"))
