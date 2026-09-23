"""Stage 3 -- alignment.

The reference is configurable, never hardcoded (spec section 9 stage 3).
`-Y` keeps supplementary alignments soft-clipped, which the clip-based Group B
features (n_clipped_reads, clip_frac) depend on.
"""


rule minimap2_align:
    input:
        fastq=RESOURCES / "raw" / "{sample}.region.fastq.gz",
        mmi=REFERENCE_MMI,
    output:
        bam=RESULTS / "align" / "{sample}.sorted.bam",
    params:
        preset=sample_preset,
        extra=config["align"]["extra"],
        platform=lambda w: "ONT" if SAMPLES.loc[w.sample, "platform"] == "ONT" else "PACBIO",
        split=lambda w: str(RESULTS / "align" / f"{w.sample}.mm2split"),
    threads: config["align"]["threads"]
    log:
        LOGS / "align" / "{sample}.minimap2.log",
    shell:
        r"""
        mkdir -p $(dirname {output.bam}) $(dirname {log})
        minimap2 -t {threads} -ax {params.preset} {params.extra} \
          --split-prefix {params.split} \
          -R "@RG\tID:{wildcards.sample}\tSM:{wildcards.sample}\tPL:{params.platform}" \
          {input.mmi} {input.fastq} 2> {log} \
          | samtools sort -@ {threads} -o {output.bam} - 2>> {log}
        samtools quickcheck {output.bam} || (echo "BAM failed quickcheck" >&2; exit 1)
        """


rule index_bam:
    input:
        bam=RESULTS / "align" / "{sample}.sorted.bam",
    output:
        bai=RESULTS / "align" / "{sample}.sorted.bam.bai",
    threads: 2
    log:
        LOGS / "align" / "{sample}.index.log",
    shell:
        "samtools index -@ {threads} {input.bam} 2> {log}"


rule alignment_stats:
    """Phase 5 exit test: mapping rate above 90% for ONT, coverage matching QC."""
    input:
        bam=RESULTS / "align" / "{sample}.sorted.bam",
        bai=RESULTS / "align" / "{sample}.sorted.bam.bai",
    output:
        stats=RESULTS / "align" / "{sample}.stats.txt",
        json=RESULTS / "align" / "{sample}.align_stats.json",
    threads: 2
    log:
        LOGS / "align" / "{sample}.stats.log",
    shell:
        r"""
        samtools stats -@ {threads} {input.bam} > {output.stats} 2> {log}
        samtools coverage {input.bam} > {output.stats}.coverage 2>> {log}

        total=$(grep -m1 '^SN\ssequences:' {output.stats} | cut -f3 || echo 0)
        mapped=$(grep -m1 '^SN\sreads mapped:' {output.stats} | cut -f3 || echo 0)
        depth=$(awk 'NR>1 {{sum+=$7; n++}} END {{if (n>0) printf "%.2f", sum/n; else print 0}}' \
                {output.stats}.coverage)
        rate=$(awk -v m="$mapped" -v t="$total" 'BEGIN {{ if (t>0) printf "%.4f", m/t; else print 0 }}')

        cat > {output.json} <<EOF
{{
  "sample_id": "{wildcards.sample}",
  "total_reads": $total,
  "mapped_reads": $mapped,
  "mapping_rate": $rate,
  "mean_depth": $depth
}}
EOF

        awk -v r="$rate" 'BEGIN {{ if (r < 0.90) {{
            print "WARNING: mapping rate " r " is below the 0.90 Phase 5 exit criterion" > "/dev/stderr"
        }} }}'
        """
