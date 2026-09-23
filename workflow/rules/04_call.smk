"""Stage 4 -- SV calling with Sniffles2 and cuteSV.

Both callers run permissively (low min-support). Filtering is the ML layer's job;
a caller tuned for precision here would remove the very candidates the model is
supposed to learn to reject, and would raise the union baseline artificially.
"""


def cutesv_params(wildcards):
    platform = SAMPLES.loc[wildcards.sample, "platform"]
    return config["sv_calling"]["cutesv"].get(platform, "")


rule sniffles:
    input:
        bam=RESULTS / "align" / "{sample}.sorted.bam",
        bai=RESULTS / "align" / "{sample}.sorted.bam.bai",
        fa=REFERENCE_FA,
    output:
        vcf=RESULTS / "sv" / "{sample}.sniffles.raw.vcf",
    params:
        minsupport=config["sv_calling"]["sniffles"]["minsupport"],
        extra=config["sv_calling"]["sniffles"]["extra"],
        minlen=config["sv_calling"]["min_sv_length"],
    threads: config["align"]["threads"]
    log:
        LOGS / "sv" / "{sample}.sniffles.log",
    shell:
        r"""
        mkdir -p $(dirname {output.vcf}) $(dirname {log})
        sniffles --input {input.bam} --vcf {output.vcf} --reference {input.fa} \
          --threads {threads} --minsupport {params.minsupport} \
          --minsvlen {params.minlen} {params.extra} > {log} 2>&1
        """


rule cutesv:
    input:
        bam=RESULTS / "align" / "{sample}.sorted.bam",
        bai=RESULTS / "align" / "{sample}.sorted.bam.bai",
        fa=REFERENCE_FA,
    output:
        vcf=RESULTS / "sv" / "{sample}.cutesv.raw.vcf",
    params:
        extra=cutesv_params,
        min_support=config["sv_calling"]["cutesv"]["min_support"],
        minlen=config["sv_calling"]["min_sv_length"],
        workdir=lambda w: str(RESULTS / "sv" / f"{w.sample}_cutesv_work"),
    threads: config["align"]["threads"]
    log:
        LOGS / "sv" / "{sample}.cutesv.log",
    shell:
        r"""
        mkdir -p {params.workdir} $(dirname {log})
        cuteSV {input.bam} {input.fa} {output.vcf} {params.workdir} \
          --threads {threads} --min_support {params.min_support} \
          --min_size {params.minlen} --genotype {params.extra} > {log} 2>&1
        rm -rf {params.workdir}
        """


rule normalise_vcf:
    """bcftools norm + size filter (finding H5).

    Normalising both callers the same way before merging means the merge is comparing
    like with like, rather than reconciling two different representations.
    """
    input:
        vcf=RESULTS / "sv" / "{sample}.{caller}.raw.vcf",
        fa=REFERENCE_FA,
    output:
        vcf=RESULTS / "sv" / "{sample}.{caller}.vcf.gz",
        tbi=RESULTS / "sv" / "{sample}.{caller}.vcf.gz.tbi",
    params:
        minlen=config["sv_calling"]["min_sv_length"],
    wildcard_constraints:
        caller="sniffles|cutesv",
    log:
        LOGS / "sv" / "{sample}.{caller}.norm.log",
    shell:
        r"""
        bcftools sort {input.vcf} 2> {log} \
          | bcftools norm -f {input.fa} -c w -Ou 2>> {log} \
          | bcftools filter -i '(INFO/SVTYPE="BND") || (abs(INFO/SVLEN) >= {params.minlen})' \
              -Oz -o {output.vcf} 2>> {log}
        tabix -f -p vcf {output.vcf} 2>> {log}
        """


rule inspect_caller_output:
    """Phase 6 exit test: inspect real INFO fields before any parser assumes names.

    Spec section 9 stage 4 requires this explicitly -- the supporting-read field is
    SUPPORT for Sniffles2 and RE for cuteSV, and both have changed across versions.
    """
    input:
        sniffles=RESULTS / "sv" / "{sample}.sniffles.vcf.gz",
        cutesv=RESULTS / "sv" / "{sample}.cutesv.vcf.gz",
    output:
        report=RESULTS / "sv" / "{sample}.header_inspection.txt",
    params:
        minlen=config["sv_calling"]["min_sv_length"],
    log:
        LOGS / "sv" / "{sample}.inspect.log",
    shell:
        r"""
        {{
          echo "######## SNIFFLES2 ########"
          svml inspect-vcf {input.sniffles} --min-sv-length {params.minlen}
          echo
          echo "######## CUTESV ########"
          svml inspect-vcf {input.cutesv} --min-sv-length {params.minlen}
        }} > {output.report} 2> {log}
        cat {output.report}
        """
