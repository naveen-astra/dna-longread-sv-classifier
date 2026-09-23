"""Feature layer.

`registry` is the single source of truth for the schema and is always importable.
`extract` pulls in the BAM/FASTA readers, which need pysam, so it is imported on
demand (finding M4).
"""

from svml.features.registry import (
    FEATURES,
    FEATURES_BY_NAME,
    Feature,
    Group,
    Missing,
    feature_names,
    indicator_name,
    model_columns,
    summary,
    to_markdown,
)

__all__ = [
    "FEATURES",
    "FEATURES_BY_NAME",
    "Feature",
    "Group",
    "Missing",
    "feature_names",
    "indicator_name",
    "model_columns",
    "summary",
    "to_markdown",
    "build_feature_matrix",
    "load_annotations",
    "write_matrix",
]

_LAZY = {
    "build_feature_matrix": "svml.features.extract",
    "load_annotations": "svml.features.extract",
    "write_matrix": "svml.features.extract",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'svml.features' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
