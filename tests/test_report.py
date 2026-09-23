"""Report honesty guards (spec section 28).

The report must never present an illustrative value as a result, and every stage C
number must carry its pseudo-label provenance in the same table as the number.
"""

from __future__ import annotations

import json
import re

import pytest

from svml.report.build import LIMITATIONS, PLACEHOLDER, build_report, write_summary_json


@pytest.fixture
def metadata() -> dict:
    return {
        "generated_utc": "2026-09-22T00:00:00+00:00",
        "config_digest": "abc123",
        "random_seed": 42,
        "svml_version": "0.1.0",
        "reference": {"id": "GRCh38_no_alt_analysis_set", "build": "GRCh38"},
        "truth": {"release": "GIAB HG002 v5.0q", "build": "GRCh38"},
        "samples": [
            {"sample_id": "HG002_ONT", "platform": "ONT", "role": "labelled",
             "truth": "v5_0q_stvar_grch38", "coverage": ""},
        ],
        "tool_versions": {"minimap2": "2.28-r1209", "truvari": None},
        "split_map_by_split": {"train": ["chr1"], "val": ["chr17"], "test": ["chr20"]},
    }


def test_empty_report_renders_tbd_not_invented_numbers(tmp_path, metadata):
    """With no benchmark, the report must say TBD rather than produce plausible numbers."""
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata)
    html = out.read_text(encoding="utf-8")

    assert PLACEHOLDER in html
    assert "Benchmark has not been run yet." in html
    assert "no conclusion is stated" in html

    # No precision/recall/F1 figure may appear anywhere in an unrun report.
    body = re.sub(r"config digest <code>[^<]*</code>", "", html)
    numbers = re.findall(r"\b0\.\d{3,4}\b", body)
    assert not numbers, f"unrun report contains numeric results: {numbers[:5]}"


def test_report_always_states_the_recall_ceiling_caveat(tmp_path, metadata):
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata)
    html = out.read_text(encoding="utf-8")
    assert "hard-capped" in html
    assert "intersection" in html


def test_stage_c_rows_carry_pseudo_label_provenance(tmp_path, metadata):
    """Finding B1: provenance sits in the table, not in a footnote."""
    stage_c = {
        "rows": [
            {"model": "rf", "label": "DUP", "precision": 0.5,
             "recall": 0.4, "f1": 0.44, "support": 12},
        ]
    }
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata, stage_c=stage_c)
    html = out.read_text(encoding="utf-8")

    assert "NOT GIAB-benchmarked" in html
    # The per-row provenance cell, not only the banner above the table.
    assert html.count("caller-consensus pseudo-label") >= 1


def test_unsupported_precision_claim_is_flagged_prominently(tmp_path, metadata):
    benchmark = {
        "recall_ceiling_union": 0.93,
        "headline_reference": "intersection",
        "view1_operating_points": {
            "intersection": {"precision": 0.9, "recall": 0.6, "f1": 0.72,
                             "tp": 60, "fp": 7, "fn": 40, "notes": []},
            "rf": {"precision": 0.8, "recall": 0.7, "f1": 0.75,
                   "tp": 70, "fp": 17, "fn": 30, "notes": []},
        },
        "claim_supported_by": [],
        "contribution": {"rf": {"reference": "intersection", "claim_supported": False}},
    }
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata, benchmark=benchmark)
    html = out.read_text(encoding="utf-8")

    assert "The precision claim is not supported by this run." in html
    assert "reported as measured" in html.lower()


def test_supported_claim_does_not_show_the_failure_banner(tmp_path, metadata):
    benchmark = {
        "recall_ceiling_union": 0.93,
        "headline_reference": "intersection",
        "view1_operating_points": {
            "intersection": {"precision": 0.7, "recall": 0.6, "f1": 0.65,
                             "tp": 60, "fp": 26, "fn": 40, "notes": []},
        },
        "claim_supported_by": ["rf"],
        "contribution": {"rf": {"reference": "intersection", "claim_supported": True}},
    }
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata, benchmark=benchmark)
    html = out.read_text(encoding="utf-8")

    assert "The precision claim is not supported by this run." not in html
    assert "precision claim is supported" in html


def test_limitations_cover_the_blocking_findings(tmp_path, metadata):
    out = build_report(out_path=tmp_path / "report.html", metadata=metadata)
    html = out.read_text(encoding="utf-8")

    assert len(LIMITATIONS) >= 6
    for required in ("DUP / INV / TRA", "one labelled sample", "draft benchmark",
                     "not clinically validated"):
        assert required.lower() in html.lower(), f"limitation missing: {required}"


def test_summary_json_is_machine_readable(tmp_path):
    """Output Q: a machine-readable summary."""
    path = write_summary_json(tmp_path / "summary.json", {"benchmark": {}, "gate_counts": {}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "benchmark" in payload
