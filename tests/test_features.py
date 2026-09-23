"""Feature registry, interval maths and the caller-derived feature groups."""

from __future__ import annotations

import math

import pytest

from svml.features import caller_features
from svml.features.annot_features import (
    _gc_fraction,
    _longest_homopolymer,
    _microhomology_length,
    classify_chrom,
)
from svml.features.registry import (
    FEATURES,
    Missing,
    feature_names,
    model_columns,
    summary,
)
from svml.io.bed import IntervalIndex, load_bed
from svml.io.vcf import CandidateRecord, normalise_svtype

# --- registry ---------------------------------------------------------------


def test_objective_o2_at_least_15_features():
    assert len(FEATURES) >= 15, "objective O2 requires at least 15 features per candidate"


def test_feature_names_are_unique():
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names))


def test_all_four_ablation_groups_are_populated():
    counts = summary()
    for group in ("B", "A", "C", "V"):
        assert counts.get(group, 0) > 0


def test_required_features_from_spec_are_present():
    """Spec section 6 names these explicitly as required."""
    names = set(feature_names())
    assert "dp_flank_ratio" in names          # read_depth_ratio
    assert "mean_mapq" in names               # MAPQ
    assert {"sniffles_support", "cutesv_support"} <= names   # breakpoint support
    assert {"tr_overlap_frac", "rmsk_overlap_frac", "segdup_overlap"} <= names  # repeat context


def test_independent_ablation_excludes_caller_groups():
    independent = set(feature_names(["B", "A"]))
    caller_derived = set(feature_names(["C", "V"]))
    assert not independent & caller_derived
    assert len(independent) >= 15, "the B+A ablation must still satisfy O2 on its own"


def test_model_columns_expand_categoricals_and_indicators():
    cols = model_columns(["V"])
    assert "svtype_called__DEL" in cols
    assert "svtype_called" not in cols
    assert "log10_svlen__isna" in cols


def test_stage_b_excludes_the_leaky_type_feature():
    """Finding H1: predicting DEL/INS from the caller's own SVTYPE is the identity function."""
    cols = model_columns(["B", "A", "C", "V"], exclude=["svtype_called"])
    assert not any(c.startswith("svtype_called") for c in cols)


def test_nan_indicator_features_declare_indicators():
    for feat in FEATURES:
        if feat.missing is Missing.NAN_INDICATOR:
            assert feat.needs_indicator


# --- interval maths ---------------------------------------------------------


def test_load_bed_merges_overlapping_intervals(tiny_bed):
    intervals = load_bed(tiny_bed)
    # (100,200) and (150,300) must collapse into (100,300).
    assert intervals["chr1"] == [(100, 300), (1000, 1100)]


def test_overlap_fraction_never_exceeds_one(tiny_bed):
    """Unmerged overlapping source intervals would double-count and exceed 1.0."""
    index = IntervalIndex.from_bed(tiny_bed)
    assert index.overlap_fraction("chr1", 100, 300) == pytest.approx(1.0)
    assert index.overlap_fraction("chr1", 0, 400) == pytest.approx(200 / 400)
    assert index.overlap_fraction("chr1", 5000, 6000) == 0.0


def test_overlap_fraction_of_zero_length_span_is_zero(tiny_bed):
    index = IntervalIndex.from_bed(tiny_bed)
    assert index.overlap_fraction("chr1", 150, 150) == 0.0


def test_contains_point(tiny_bed):
    index = IntervalIndex.from_bed(tiny_bed)
    assert index.contains_point("chr1", 150)
    assert not index.contains_point("chr1", 500)
    assert not index.contains_point("chrX", 150)


def test_distance_to_nearest(tiny_bed):
    index = IntervalIndex.from_bed(tiny_bed)
    assert index.distance_to_nearest("chr1", 150) == 0       # inside (100, 300)
    # Nearest covered base is 299, in the merged (100, 300) interval -- not the
    # interval starting at 1000, which is 600 away.
    assert index.distance_to_nearest("chr1", 400) == 101
    assert index.distance_to_nearest("chr1", 900) == 100     # to (1000, 1100)
    assert index.distance_to_nearest("chrZ", 1) == 1_000_000_000


# --- sequence helpers -------------------------------------------------------


def test_gc_fraction_ignores_ambiguous_bases():
    assert _gc_fraction("GCGC") == pytest.approx(1.0)
    assert _gc_fraction("ATAT") == pytest.approx(0.0)
    assert _gc_fraction("GCATNNNN") == pytest.approx(0.5)
    assert _gc_fraction("NNNN") is None
    assert _gc_fraction("") is None


def test_longest_homopolymer():
    assert _longest_homopolymer("ACGTTTTACG") == 4
    assert _longest_homopolymer("ACGT") == 1
    assert _longest_homopolymer("") == 0


def test_microhomology_length():
    assert _microhomology_length("AACCGG", "CCGGTT") == 4
    assert _microhomology_length("AAAA", "TTTT") == 0
    assert _microhomology_length("", "ACGT") == 0


def test_classify_chrom():
    assert classify_chrom("chr1") == "autosome"
    assert classify_chrom("chr22") == "autosome"
    assert classify_chrom("chrX") == "chrX"
    assert classify_chrom("chrY") == "chrY"
    assert classify_chrom("chrM") == "other"
    assert classify_chrom("chr1_KI270706v1_random") == "other"


# --- caller features --------------------------------------------------------


def _candidate(**overrides) -> CandidateRecord:
    payload = {
        "candidate_id": "SV_1",
        "chrom": "chr1",
        "pos": 1000,
        "end": 1500,
        "svtype": "DEL",
        "svlen": -500,
        "qual": 30.0,
        "filter_pass": True,
    }
    payload.update(overrides)
    return CandidateRecord(**payload)


def test_single_caller_disagreement_is_none_not_zero():
    """Zero would mean 'the callers agreed perfectly' -- the opposite of the truth."""
    cand = _candidate()
    cand.per_caller = {"sniffles2": {"support": 10, "pos": 1000, "svlen": -500, "qual": 30.0}}

    out = caller_features.extract(cand)
    assert out["n_callers"] == 1
    assert out["sniffles_support"] == 10
    assert out["cutesv_support"] is None
    assert out["bp_offset_between_callers"] is None
    assert out["svlen_disagreement"] is None


def test_two_caller_disagreement_is_computed():
    cand = _candidate()
    cand.per_caller = {
        "sniffles2": {"support": 10, "pos": 1000, "svlen": -500, "qual": 30.0},
        "cutesv": {"support": 8, "pos": 1012, "svlen": -520, "qual": 20.0},
    }

    out = caller_features.extract(cand)
    assert out["n_callers"] == 2
    assert out["bp_offset_between_callers"] == 12
    assert out["svlen_disagreement"] == 20
    assert out["caller_qual"] == pytest.approx(25.0)


def test_log10_svlen_uses_absolute_value():
    """The sign of SVLEN encodes DEL vs INS and must not leak into stage B."""
    deletion = _candidate(svtype="DEL", svlen=-500)
    insertion = _candidate(svtype="INS", svlen=500, end=1000)
    deletion.per_caller = insertion.per_caller = {}

    assert caller_features.extract(deletion)["log10_svlen"] == pytest.approx(
        caller_features.extract(insertion)["log10_svlen"]
    )
    assert caller_features.extract(deletion)["log10_svlen"] == pytest.approx(math.log10(501))


def test_insertion_has_zero_reference_span():
    insertion = _candidate(svtype="INS", svlen=500, end=1000)
    assert insertion.span_len == 0
    assert not insertion.has_reference_span

    deletion = _candidate()
    assert deletion.span_len == 500
    assert deletion.has_reference_span


# --- type normalisation -----------------------------------------------------


def test_normalise_svtype_maps_aliases():
    assert normalise_svtype("DEL") == "DEL"
    assert normalise_svtype("<DUP:TANDEM>") == "DUP"
    assert normalise_svtype("deletion") == "DEL"
    assert normalise_svtype(None) is None
    assert normalise_svtype("WEIRD") is None


def test_translocation_is_normalised_to_bnd():
    """Callers emit BND; reporting it as TRA would assert a claim they did not make."""
    assert normalise_svtype("TRA") == "BND"
    assert normalise_svtype("BND") == "BND"


# --- merge ID uniqueness ----------------------------------------------------


def test_assign_unique_ids_replaces_missing_and_duplicate_ids(tmp_path):
    """Truvari labels by ID; '.' or colliding IDs would attach labels to the wrong row."""
    from svml.candidates.merge import assign_unique_ids

    vcf = tmp_path / "merged.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr19\t100\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\n"
        "chr19\t200\tdup\tN\t<INS>\t.\tPASS\tSVTYPE=INS\n"
        "chr19\t300\tdup\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\n",
        encoding="utf-8",
    )
    assert assign_unique_ids(vcf) == 3

    ids = [line.split("\t")[2] for line in vcf.read_text().splitlines() if not line.startswith("#")]
    assert ids == ["SVML0000001", "SVML0000002", "SVML0000003"]
    assert vcf.read_text().startswith("##fileformat=VCFv4.2\n#CHROM")
