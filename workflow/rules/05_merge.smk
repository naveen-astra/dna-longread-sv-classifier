"""Stage 5 -- candidate merging (finding H5).

Merging is delegated to SURVIVOR, then provenance is reconciled back against the
per-caller VCFs so features C1-C5 (per-caller support, caller agreement, breakpoint
and size disagreement) can be computed.

Merging happens BEFORE splitting, so the same event cannot land in two splits
(docs/04_ml_task_definition.md, leakage control 1).
"""


rule merge_candidates:
    input:
        sniffles=RESULTS / "sv" / "{sample}.sniffles.vcf.gz",
        cutesv=RESULTS / "sv" / "{sample}.cutesv.vcf.gz",
        inspection=RESULTS / "sv" / "{sample}.header_inspection.txt",
    output:
        vcf=RESULTS / "candidates" / "{sample}.merged.vcf",
        pickle=RESULTS / "candidates" / "{sample}.candidates.pkl",
        summary=RESULTS / "candidates" / "{sample}.merge_summary.json",
    log:
        LOGS / "candidates" / "{sample}.merge.log",
    shell:
        r"""
        mkdir -p $(dirname {output.vcf}) $(dirname {log})
        svml merge --config config/config.yaml \
          --sniffles {input.sniffles} --cutesv {input.cutesv} \
          --out-vcf {output.vcf} --out-pickle {output.pickle} \
          --out-summary {output.summary} --log {log}
        """


rule compress_merged:
    input:
        vcf=RESULTS / "candidates" / "{sample}.merged.vcf",
    output:
        vcf=RESULTS / "candidates" / "{sample}.merged.vcf.gz",
        tbi=RESULTS / "candidates" / "{sample}.merged.vcf.gz.tbi",
    log:
        LOGS / "candidates" / "{sample}.compress.log",
    shell:
        r"""
        bcftools sort {input.vcf} -Oz -o {output.vcf} 2> {log}
        tabix -f -p vcf {output.vcf} 2>> {log}
        """


rule merge_check:
    """Phase 6 exit test: the merged set must be smaller than naive concatenation."""
    input:
        summary=RESULTS / "candidates" / "{sample}.merge_summary.json",
        sniffles=RESULTS / "sv" / "{sample}.sniffles.vcf.gz",
        cutesv=RESULTS / "sv" / "{sample}.cutesv.vcf.gz",
    output:
        ok=RESULTS / "checks" / "{sample}.merge_check.txt",
    log:
        LOGS / "candidates" / "{sample}.merge_check.log",
    shell:
        r"""
        mkdir -p $(dirname {output.ok})
        n_sniffles=$(bcftools view -H {input.sniffles} | wc -l)
        n_cutesv=$(bcftools view -H {input.cutesv} | wc -l)
        n_merged=$(python -c "import json;print(json.load(open('{input.summary}'))['n_candidates'])")
        concat=$((n_sniffles + n_cutesv))

        {{
          echo "sniffles2 calls : $n_sniffles"
          echo "cuteSV calls    : $n_cutesv"
          echo "naive concat    : $concat"
          echo "merged set      : $n_merged"
        }} > {output.ok}

        if [ "$n_merged" -ge "$concat" ] && [ "$concat" -gt 0 ]; then
          echo "MERGE CHECK FAILED: merged set is not smaller than naive concatenation" >&2
          exit 1
        fi
        echo "merge check passed" >> {output.ok}
        """
