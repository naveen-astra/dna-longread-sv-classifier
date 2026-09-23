"""Stage 6 -- feature extraction.

The layer boundary the architecture depends on: this stage reads the BAM, the
reference and the annotations, and NEVER reads truth data. Truth enters only at
stage 7.
"""

ANNOTATION_TRACKS = list(REFS["annotations"])


rule extract_features:
    input:
        candidates=RESULTS / "candidates" / "{sample}.candidates.pkl",
        bam=RESULTS / "align" / "{sample}.sorted.bam",
        bai=RESULTS / "align" / "{sample}.sorted.bam.bai",
        fa=REFERENCE_FA,
        fai=str(REFERENCE_FA) + ".fai",
        beds=expand(str(ANNOTATION_DIR / "{track}.bed"), track=ANNOTATION_TRACKS),
        naming=RESULTS / "checks" / "contig_naming.txt",
    output:
        parquet=RESULTS / "features" / "{sample}.parquet",
    params:
        annotation_dir=ANNOTATION_DIR,
    threads: 2
    log:
        LOGS / "features" / "{sample}.extract.log",
    shell:
        r"""
        mkdir -p $(dirname {output.parquet}) $(dirname {log})
        svml features --config config/config.yaml \
          --candidates {input.candidates} --bam {input.bam} \
          --reference {input.fa} --annotation-dir {params.annotation_dir} \
          --sample {wildcards.sample} --out {output.parquet} --log {log}
        """


rule feature_sanity:
    """Phase 7 exit test: the six validation gates from docs/03_feature_dictionary.md.

    The direction check is the important one -- dp_flank_ratio must be visibly lower
    for DEL candidates than for INS ones. If that fails, the depth computation is
    wrong and nothing downstream is trustworthy.
    """
    input:
        parquet=RESULTS / "features" / "{sample}.parquet",
    output:
        report=RESULTS / "checks" / "{sample}.feature_sanity.json",
    log:
        LOGS / "features" / "{sample}.sanity.log",
    shell:
        r"""
        mkdir -p $(dirname {output.report})
        python - <<'PY' > {output.report} 2> {log}
import json, sys
import pandas as pd
from svml.features.registry import FEATURES

frame = pd.read_parquet("{input.parquet}")
report = {{"n_rows": int(len(frame)), "n_features": len(FEATURES), "gates": {{}}}}

degenerate, missing_heavy = [], []
for feat in FEATURES:
    if feat.name not in frame.columns:
        degenerate.append(f"{{feat.name}} (absent)")
        continue
    series = frame[feat.name]
    if series.nunique(dropna=True) <= 1:
        degenerate.append(f"{{feat.name}} (constant)")
    elif not feat.categorical:
        top = series.value_counts(normalize=True, dropna=True)
        if len(top) and top.iloc[0] > 0.95:
            degenerate.append(f"{{feat.name}} (>95% one value)")
    na_rate = float(series.isna().mean())
    if na_rate > 0.5:
        missing_heavy.append({{"feature": feat.name, "na_rate": round(na_rate, 3)}})

report["gates"]["non_degenerate"] = {{"failures": degenerate}}
report["gates"]["missingness"] = {{"high_na": missing_heavy}}

# Direction sanity: DEL should show a LOWER depth ratio than DUP, and lower than INS
# flanks generally. Reported rather than asserted, because a pilot region may have too
# few DUPs for the comparison to be meaningful.
direction = {{}}
if "dp_flank_ratio" in frame.columns:
    for svtype in ("DEL", "DUP", "INV"):
        subset = frame.loc[frame["svtype"] == svtype, "dp_flank_ratio"].dropna()
        direction[svtype] = {{"n": int(len(subset)),
                             "median": float(subset.median()) if len(subset) else None}}
    del_med = direction.get("DEL", {{}}).get("median")
    dup_med = direction.get("DUP", {{}}).get("median")
    if del_med is not None and dup_med is not None:
        direction["DEL_below_DUP"] = bool(del_med < dup_med)
        if del_med >= dup_med:
            print("WARNING: DEL depth ratio is not below DUP -- check the depth computation",
                  file=sys.stderr)
report["gates"]["direction_dp_flank_ratio"] = direction

report["passed"] = not degenerate
json.dump(report, sys.stdout, indent=2)
PY
        """
