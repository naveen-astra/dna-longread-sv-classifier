"""svml -- AI-based structural variant detection using long-read sequencing data.

The package is importable and testable independently of Snakemake. Workflow rules are
thin wrappers that call into `svml.cli`.

Layer boundary that must not be violated (see docs/01_system_architecture.md):
the feature-extraction layer never reads truth data. Truth enters only at the
labelling (`svml.labels`) and benchmarking (`svml.eval`) layers.
"""

__version__ = "0.1.0"
