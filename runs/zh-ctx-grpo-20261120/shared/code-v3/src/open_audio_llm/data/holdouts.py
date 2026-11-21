"""Leave out training records built from recordings an evaluation set holds out.

The catalog declares both sides (audio-data-contract ``source_keys``): an
evaluation dataset lists the source recordings it holds out, and a training
dataset says where its records name their source recordings. A source whose
dataset declares source keys drops every record that shares one with a
holdout, with no per-config exclusion list to remember.

Data configs often point ``catalog`` at one file or a prepared subset that does
not contain the evaluation datasets, so holdouts can be read from elsewhere::

    holdouts:
      catalog: /workspace/data/registry/catalog
      roots: /workspace/data/registry/roots.json

A source that declares source keys while no holdout is found is an error, not
an unchecked pass.
"""

from __future__ import annotations

from audio_data_contract import load_catalog, load_holdouts, load_roots


def config_holdouts(config, resolver):
    """Holdouts from ``config["holdouts"]``, else from the data catalog itself."""
    settings = config.get("holdouts") or {}
    catalog = load_catalog(settings["catalog"]) if "catalog" in settings else resolver.catalog
    roots = load_roots(settings["roots"]) if "roots" in settings else resolver.roots
    return load_holdouts(catalog, roots)


def require_holdouts(holdouts, config, where):
    if not holdouts.sources:
        found_in = (config.get("holdouts") or {}).get("catalog", config["catalog"])
        raise ValueError(
            f"{where} declares source keys, but {found_in} declares no holdout; set "
            "holdouts.catalog (and holdouts.roots) to the registry catalog"
        )
    return holdouts
