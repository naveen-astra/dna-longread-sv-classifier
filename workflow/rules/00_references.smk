"""Stage 0 -- reference and annotation preparation (finding M1).

Runs before data ingestion because everything downstream depends on it. The GRCh38
no-alt analysis set is used deliberately: ALT contigs fragment long-read alignments
and generate spurious SV signatures.
"""


rule download_reference:
    output:
        fa=REFERENCE_FA,
    params:
        url=REFS["genome"]["url"],
    log:
        LOGS / "references" / "download_reference.log",
    shell:
        r"""
        mkdir -p $(dirname {output.fa}) $(dirname {log})
        # Resumable with retries: on a slow link a ~900 MB transfer from NCBI is routinely
        # cut off mid-stream (curl exit 18). `-C -` continues from the partial file.
        ok=0
        for attempt in $(seq 1 20); do
          if curl -fsSL --retry 5 --retry-delay 10 -C - {params.url} -o {output.fa}.gz 2>> {log}; then
            ok=1; break
          fi
          echo "download attempt $attempt interrupted; resuming" >> {log}
          sleep 15
        done
        [ "$ok" = 1 ] || (echo "reference download failed after retries" >&2; exit 1)
        gzip -t {output.fa}.gz 2>> {log} || (echo "reference archive is corrupt" >&2; exit 1)
        gunzip -c {output.fa}.gz > {output.fa} 2>> {log}
        rm -f {output.fa}.gz
        """


rule index_reference:
    input:
        fa=REFERENCE_FA,
    output:
        fai=str(REFERENCE_FA) + ".fai",
    log:
        LOGS / "references" / "faidx.log",
    shell:
        "samtools faidx {input.fa} 2> {log}"


rule minimap2_index:
    input:
        fa=REFERENCE_FA,
    output:
        mmi=REFERENCE_MMI,
    params:
        batch=config["align"].get("index_batch", "8G"),
    threads: config["align"]["threads"]
    log:
        LOGS / "references" / "minimap2_index.log",
    shell:
        "minimap2 -t {threads} -I {params.batch} -d {output.mmi} {input.fa} 2> {log}"


rule download_annotation:
    """UCSC tracks -> sorted, merged BED. Required for the Group A repeat_context features."""
    output:
        bed=ANNOTATION_DIR / "{track}.bed",
    params:
        url=lambda w: REFS["annotations"][w.track]["url"],
    log:
        LOGS / "references" / "annotation_{track}.log",
    shell:
        r"""
        mkdir -p $(dirname {output.bed}) $(dirname {log})
        for attempt in $(seq 1 10); do
          curl -fsSL --retry 5 --retry-delay 10 -C - {params.url} -o {output.bed}.txt.gz 2>> {log} && break
          sleep 15
        done
        gzip -t {output.bed}.txt.gz 2>> {log} || (echo "annotation archive corrupt" >&2; exit 1)
        # UCSC table dumps vary in column layout; genoName/genoStart/genoEnd sit at
        # columns 6-8 for rmsk and 2-4 for simpleRepeat/gap/genomicSuperDups. Detect
        # which by looking for the first field that parses as a chrom name.
        zcat {output.bed}.txt.gz \
          | awk 'BEGIN{{OFS="\t"}} {{
                if ($6 ~ /^chr/ && $7 ~ /^[0-9]+$/) print $6,$7,$8;
                else if ($2 ~ /^chr/ && $3 ~ /^[0-9]+$/) print $2,$3,$4;
                else if ($1 ~ /^chr/ && $2 ~ /^[0-9]+$/) print $1,$2,$3;
            }}' \
          | sort -k1,1 -k2,2n \
          | bedtools merge -i - > {output.bed} 2>> {log}
        rm -f {output.bed}.txt.gz
        test -s {output.bed} || (echo "empty annotation BED for {wildcards.track}" >> {log}; exit 1)
        """


rule download_truth:
    """GIAB v5.0q stvar on GRCh38 (finding B2).

    Filenames are resolved from the release listing during Phase 2 and recorded in
    config/references.yaml. The rule fails loudly while they are still TODO rather
    than silently fetching the wrong benchmark -- note that NISTv4.2.1 is the SMALL
    VARIANT benchmark, not the SV one (finding B1c).
    """
    output:
        vcf=TRUTH_VCF,
        tbi=str(TRUTH_VCF) + ".tbi",
        bed=CONFIDENT_BED,
    params:
        base=lambda w: REFS["truth"][config["labels"]["truth"]]["base_url"],
        vcf_name=lambda w: REFS["truth"][config["labels"]["truth"]]["vcf"],
        bed_name=lambda w: REFS["truth"][config["labels"]["truth"]]["bed"],
        md5_vcf=lambda w: REFS["truth"][config["labels"]["truth"]]["md5"]["vcf"],
        md5_bed=lambda w: REFS["truth"][config["labels"]["truth"]]["md5"]["bed"],
    log:
        LOGS / "references" / "download_truth.log",
    shell:
        r"""
        mkdir -p $(dirname {output.vcf}) $(dirname {log})
        if [ "{params.vcf_name}" = "TODO" ] || [ "{params.bed_name}" = "TODO" ]; then
          echo "config/references.yaml still has TODO for the truth filenames." >&2
          echo "Resolve them from {params.base} during Phase 2 (see docs/06_execution_plan.md)." >&2
          exit 1
        fi
        curl -fsSL "{params.base}{params.vcf_name}" -o {output.vcf} 2> {log}
        curl -fsSL "{params.base}{params.bed_name}" -o {output.bed} 2>> {log}

        # Verify against GIAB's published checksums before anything downstream trusts
        # these files -- labels and every benchmark number derive from them.
        echo "{params.md5_vcf}  {output.vcf}" | md5sum -c - >> {log} 2>&1 \
          || (echo "truth VCF md5 mismatch" >&2; exit 1)
        echo "{params.md5_bed}  {output.bed}" | md5sum -c - >> {log} 2>&1 \
          || (echo "confident-region BED md5 mismatch" >&2; exit 1)

        tabix -f -p vcf {output.vcf} 2>> {log}
        """


rule check_contig_naming:
    """Phase 2 exit test.

    Contig-naming mismatch (chr1 vs 1) is the most common silent failure in this kind
    of pipeline: every overlap query returns zero, and the features look plausible but
    are all wrong.
    """
    input:
        fai=str(REFERENCE_FA) + ".fai",
        beds=expand(str(ANNOTATION_DIR / "{track}.bed"), track=list(REFS["annotations"])),
        truth=TRUTH_VCF,
        confident=CONFIDENT_BED,
    output:
        report=RESULTS / "checks" / "contig_naming.txt",
    log:
        LOGS / "references" / "check_contig_naming.log",
    shell:
        r"""
        mkdir -p $(dirname {output.report})
        {{
          echo "== reference contigs (first 5) =="
          cut -f1 {input.fai} | head -5
          for bed in {input.beds}; do
            echo "== $bed (first 3) =="
            cut -f1 "$bed" | sort -u | head -3
          done
          echo "== truth VCF contigs (first 5) =="
          bcftools index --stats {input.truth} 2>/dev/null | cut -f1 | head -5 \
            || zcat {input.truth} | grep -v '^#' | cut -f1 | sort -u | head -5
          echo "== confident BED (first 3) =="
          cut -f1 {input.confident} | sort -u | head -3
        }} > {output.report} 2> {log}

        ref_chr=$(cut -f1 {input.fai} | head -1 | grep -c '^chr' || true)
        bed_chr=$(cut -f1 {input.confident} | head -1 | grep -c '^chr' || true)
        if [ "$ref_chr" != "$bed_chr" ]; then
          echo "CONTIG NAMING MISMATCH between reference and confident-region BED" >&2
          exit 1
        fi
        echo "contig naming consistent" >> {output.report}
        """
