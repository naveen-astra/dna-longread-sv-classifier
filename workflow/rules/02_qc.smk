"""Stage 2 -- quality control.

Spec section 9 stage 2: no data is discarded without the filtering rule being written
down. The thresholds live in config.yaml and the applied rule is recorded in the
metrics JSON.
"""


rule nanoplot:
    input:
        fastq=RESOURCES / "raw" / "{sample}.region.fastq.gz",
    output:
        stats=RESULTS / "qc" / "{sample}" / "NanoStats.txt",
        directory=directory(RESULTS / "qc" / "{sample}" / "nanoplot"),
    threads: 4
    log:
        LOGS / "qc" / "{sample}.nanoplot.log",
    shell:
        r"""
        mkdir -p {output.directory} $(dirname {log})
        # --downsample 9000: the installed narwhals/plotly/pandas combination has a
        # length-mismatch bug in NanoPlot's yield-plot code (ShapeError, "Expected
        # object of length 10000, got length: N") whenever the read count exceeds
        # NanoPlot's own internal 10000-point downsampling threshold; that crash is
        # FATAL and NanoStats.txt is never written. Explicitly downsampling to 9000
        # avoids the mismatch. A second, unrelated plotly API bug
        # (`figure_factory.create_2d_density` renamed upstream) can still occur in the
        # bivariate density plot, but it is caught internally and does not stop
        # NanoStats.txt from being written or the process from exiting 0 -- verified
        # directly on both the affected sample and this fix. 9000 reads is still a
        # representative subsample for QC summary statistics; this pipeline does not
        # depend on NanoPlot's plot images, only NanoStats.txt.
        set +e
        NanoPlot --fastq {input.fastq} --outdir {output.directory} \
          --threads {threads} --N50 --tsv_stats --downsample 9000 > {log} 2>&1
        set -e
        test -s {output.directory}/NanoStats.txt || \
          (echo "NanoPlot did not produce NanoStats.txt -- see {log}" >&2; exit 1)
        cp {output.directory}/NanoStats.txt {output.stats}
        """


rule qc_metrics:
    """Parse NanoStats into machine-readable metrics and apply the documented QC rule."""
    input:
        stats=RESULTS / "qc" / "{sample}" / "NanoStats.txt",
    output:
        json=RESULTS / "qc" / "{sample}" / "qc_metrics.json",
    params:
        min_len=config["qc"]["min_read_length"],
        min_q=config["qc"]["min_mean_quality"],
    log:
        LOGS / "qc" / "{sample}.metrics.log",
    shell:
        r"""
        python - <<'PY' > {output.json} 2> {log}
import json, re, sys

metrics = {{}}
with open("{input.stats}") as fh:
    for line in fh:
        parts = [p.strip() for p in re.split(r"\t|:", line.strip()) if p.strip()]
        if len(parts) >= 2:
            key = parts[0].lower().replace(" ", "_").replace(">", "gt")
            value = parts[-1].replace(",", "")
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value

def pick(*names):
    # NanoPlot's key names differ between its text and --tsv_stats formats and across
    # versions, so accept every known alias rather than assuming one.
    for name in names:
        if metrics.get(name) not in (None, ""):
            return metrics[name]
    return None

out = {{
    "sample_id": "{wildcards.sample}",
    "n_reads": pick("number_of_reads"),
    "total_bases": pick("number_of_bases", "total_bases"),
    "read_n50": pick("n50", "read_length_n50"),
    "median_length": pick("median_read_length"),
    "mean_length": pick("mean_read_length"),
    "median_quality": pick("median_qual", "median_read_quality"),
    "mean_quality": pick("mean_qual", "mean_read_quality"),
    "qc_rule": {{
        "min_read_length": {params.min_len},
        "min_mean_quality": {params.min_q},
        "applied": "recorded only; reads are not filtered before alignment. Minimap2 "
                   "handles short and low-quality reads, and discarding them here "
                   "would change coverage without being reflected in the QC report.",
    }},
    "raw": metrics,
}}
json.dump(out, sys.stdout, indent=2)
PY
        """
