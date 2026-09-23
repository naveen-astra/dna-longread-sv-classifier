"""Stage 1 -- data acquisition (finding M2).

Two-tier strategy. Reads are acquired REGION-SLICED from GIAB's pre-aligned BAMs and
converted back to FASTQ, so the genuine FASTQ -> Minimap2 -> BAM path still runs (which
is the CO2/CO3 evidence) at a small fraction of whole-genome storage and runtime.

Raw downloads are never modified (spec section 9 stage 1); slicing writes new files.
"""

rule download_source_bam:
    """Fetch only the index plus the target region, not the whole genome.

    `samtools view <url> <region>` range-requests over HTTPS, so this transfers on the
    order of hundreds of MB rather than ~100 GB per sample.
    """
    output:
        bam=temp(RESOURCES / "raw" / "{sample}.region.bam"),
    params:
        url=lambda w: SAMPLES.loc[w.sample, "source_url"],
        regions=sample_regions,
    threads: 2
    log:
        LOGS / "acquire" / "{sample}.download.log",
    shell:
        r"""
        mkdir -p $(dirname {output.bam}) $(dirname {log})
        if [ "{params.url}" = "TODO" ] || [ -z "{params.url}" ]; then
          echo "source_url for {wildcards.sample} is still TODO in config/samples.tsv." >&2
          echo "Resolve it from the GIAB data index during Phase 3." >&2
          exit 1
        fi
        # htslib caches the remote .bai in the working directory, so run from the raw
        # directory rather than scattering index files through the repo.
        out=$(readlink -f {output.bam}); logf=$(readlink -f {log})
        cd $(dirname "$out")
        # One region at a time, each retried independently: remote reads over a slow link
        # get cut off, and retrying per window loses one window rather than the whole slice.
        # Windows are megabases apart, so no read can appear in two parts.
        parts=()
        i=0
        for region in {params.regions}; do
          i=$((i + 1)); part="$out.part$i.bam"; ok=0
          for attempt in 1 2 3 4 5 6; do
            if samtools view -@ {threads} -b "{params.url}" "$region" > "$part" 2>> "$logf" \
               && samtools quickcheck "$part"; then
              ok=1; break
            fi
            echo "region $region attempt $attempt failed; retrying" >> "$logf"; sleep 20
          done
          [ "$ok" = 1 ] || (echo "could not fetch $region" >&2; exit 1)
          parts+=("$part")
        done
        samtools cat -o "$out" "${{parts[@]}}" 2>> "$logf"
        rm -f "${{parts[@]}}"
        samtools quickcheck "$out" || (echo "sliced BAM failed quickcheck" >&2; exit 1)
        """


rule bam_to_fastq:
    input:
        bam=RESOURCES / "raw" / "{sample}.region.bam",
    output:
        fastq=RESOURCES / "raw" / "{sample}.region.fastq.gz",
    threads: 4
    log:
        LOGS / "acquire" / "{sample}.fastq.log",
    shell:
        r"""
        # Long reads are single-end: they carry neither the READ1 nor READ2 flag, so
        # `-0 /dev/null` would discard every one of them. Write to stdout instead, and
        # drop secondary/supplementary records (-F 0x900) so each read appears exactly
        # once, from its primary alignment, with its full sequence.
        samtools fastq -@ {threads} -F 0x900 {input.bam} 2> {log} \
          | gzip -c > {output.fastq}
        test -s {output.fastq} || (echo "empty FASTQ for {wildcards.sample}" >&2; exit 1)
        """


rule sample_metadata:
    """Record sample ID, platform, coverage, source and slice region (spec section 9)."""
    input:
        fastq=RESOURCES / "raw" / "{sample}.region.fastq.gz",
    output:
        json=RESULTS / "acquire" / "{sample}.metadata.json",
    params:
        platform=lambda w: SAMPLES.loc[w.sample, "platform"],
        role=lambda w: SAMPLES.loc[w.sample, "role"],
        truth=lambda w: SAMPLES.loc[w.sample, "truth"],
        url=lambda w: SAMPLES.loc[w.sample, "source_url"],
        regions=sample_regions,
        tier=config["regions"]["tier"],
    log:
        LOGS / "acquire" / "{sample}.metadata.log",
    shell:
        r"""
        mkdir -p $(dirname {output.json})
        cat > {output.json} <<EOF
{{
  "sample_id": "{wildcards.sample}",
  "platform": "{params.platform}",
  "role": "{params.role}",
  "truth": "{params.truth}",
  "source_url": "{params.url}",
  "regions": "{params.regions}",
  "region_tier": {params.tier},
  "fastq": "{input.fastq}"
}}
EOF
        """
