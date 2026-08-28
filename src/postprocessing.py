#!/usr/bin/env python3
"""
postprocessing.py

Inputs (defaults assume the derived-data files sit alongside this script's data
directory; override with the flags):
    --collapsed   relations_validated_collapsed.csv
    --full        relations_validated_full.csv
    --merged      relations_graph_ready_identifier_merged.csv
    --pubchem-cache pubchem_cache.json
    --pubtator    pubtator_relations.csv   (optional; enables stage 5)

Usage
    python src/reproduce_postprocessing.py
    python src/reproduce_postprocessing.py --pubtator pubtator_comparison/pubtator_relations.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

NA = dict(keep_default_na=False, na_values=[], low_memory=False)

SYMMETRIC_LABELS = {"Association", "HPRD-Association", "EUADR-Association",
                    "AIMED-Association", "Comparison", "Bind", "PPIm", "int"}
GENERIC_ORGANISMS = {"patients", "humans", "human", "dogs", "rats", "mice"}


def as_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")


def normalize_key(value):
    """The chemical-name key used when the PubChem cache was built."""
    text = str(value or "").strip().lower().replace("®", "").replace("™", "")
    text = text.strip(" .,;:()[]{}\"'")
    return re.sub(r"\s+", " ", text)


INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def has_inchikey(value):
    """True if the field contains a structurally valid InChIKey."""
    return bool(INCHIKEY.search(str(value)))


# --------------------------------------------------------------------------- #
# Stage 1-2: collapse cleanup and retention
# --------------------------------------------------------------------------- #
def pair_key(row):
    a = f"{row['entity1_canonical_text']}||{row['entity1_type']}"
    b = f"{row['entity2_canonical_text']}||{row['entity2_type']}"
    if row["pred_label"] in SYMMETRIC_LABELS:
        return " -- ".join(sorted([a, b]))
    return f"{a} --> {b}"


def collapse_cleanup(collapsed):
    clean = collapsed[
        collapsed["entity1_canonical_text"].str.lower()
        != collapsed["entity2_canonical_text"].str.lower()
    ].copy()
    clean["relation_pair_key"] = clean.apply(pair_key, axis=1)
    clean = (clean.groupby(["relation_pair_key", "pred_label"], dropna=False)
             .agg(entity1_canonical_text=("entity1_canonical_text", "first"),
                  entity1_type=("entity1_type", "first"),
                  entity2_canonical_text=("entity2_canonical_text", "first"),
                  entity2_type=("entity2_type", "first"),
                  has_validated_pesticide=("has_validated_pesticide",
                                           lambda s: bool(as_bool(s).max())),
                  has_unvalidated_chemical=("has_unvalidated_chemical",
                                            lambda s: bool(as_bool(s).max())))
             .reset_index())
    return clean


def retention(clean):
    """Apply the three criteria in order; return the kept set and the split."""
    n0 = len(clean)
    keep = clean[clean["has_validated_pesticide"]]
    removed_no_pesticide = n0 - len(keep)

    step = keep[~keep["has_unvalidated_chemical"]]
    removed_unvalidated = len(keep) - len(step)

    vague = (step["entity1_canonical_text"].str.lower().isin(GENERIC_ORGANISMS)
             | step["entity2_canonical_text"].str.lower().isin(GENERIC_ORGANISMS))
    kept = step[~vague]
    removed_generic = len(step) - len(kept)

    split = {
        "no_validated_pesticide": removed_no_pesticide,
        "unvalidated_chemical": removed_unvalidated,
        "generic_organism": removed_generic,
    }
    return kept, split


# --------------------------------------------------------------------------- #
# Stage 4: chemical-identifier coverage
# --------------------------------------------------------------------------- #
def chemical_coverage(full, cache):
    """Distinct chemical mentions entering relation extraction, and the share
    resolving to a structural identifier via the dictionary or PubChem."""
    frames = []
    for pre in ("entity1", "entity2"):
        m = full[f"{pre}_type"] == "ChemicalEntity"
        d = full.loc[m, [f"{pre}_text", f"{pre}_pesticide_inchikey"]].copy()
        d.columns = ["mention", "dict_inchikey"]
        frames.append(d)
    chem = pd.concat(frames, ignore_index=True)
    chem["key"] = chem["mention"].astype(str).str.strip().str.lower()

    grp = chem.groupby("key")
    dict_resolved = grp["dict_inchikey"].apply(
        lambda s: any(has_inchikey(x) for x in s))

    def pubchem_resolved(name):
        entry = cache.get(normalize_key(name))
        return bool(entry and has_inchikey(entry.get("pubchem_inchikey", "")))

    keys = list(grp.groups.keys())
    resolved = sum(bool(dict_resolved[k]) or pubchem_resolved(k) for k in keys)
    n = len(keys)
    return {"distinct": n, "resolved": resolved,
            "pct": resolved / n * 100 if n else 0.0}


# --------------------------------------------------------------------------- #
# Stage 5: eligibility of the yield gap (needs the PubTator pull)
# --------------------------------------------------------------------------- #
def eligibility(full, pubtator_relations, cache):
    """Of the abstracts where only this workflow returned relations, how many
    contain only relations PubTator cannot express (every relation has an
    organism endpoint, or a chemical endpoint with no structural identifier)."""
    our_pmids = set(full["pmid"].astype(str).str.strip())
    pt = pd.read_csv(pubtator_relations, **NA)
    pt_pmids = set(pt["pmid"].astype(str).str.strip())
    gap = our_pmids - pt_pmids

    def chem_resolved(mention, dict_ik):
        if has_inchikey(dict_ik):
            return True
        entry = cache.get(normalize_key(mention))
        return bool(entry and has_inchikey(entry.get("pubchem_inchikey", "")))

    def endpoint_beyond(row, pre):
        t = row[f"{pre}_type"]
        if t == "OrganismTaxon":
            return True
        if t == "ChemicalEntity":
            return not chem_resolved(row[f"{pre}_text"], row[f"{pre}_pesticide_inchikey"])
        return False  # gene / disease: expressible by PubTator

    sub = full[full["pmid"].astype(str).str.strip().isin(gap)].copy()
    beyond = [endpoint_beyond(r, "entity1") or endpoint_beyond(r, "entity2")
              for _, r in sub.iterrows()]
    sub = sub.assign(beyond=beyond)
    per_pmid = sub.groupby(sub["pmid"].astype(str).str.strip())["beyond"].all()
    ineligible = int(per_pmid.sum())
    return {"gap": len(gap), "ineligible": ineligible,
            "pct": ineligible / len(gap) * 100 if len(gap) else 0.0}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parents[1]
    data = here / "data"
    ap.add_argument("--collapsed", default=str(data / "relations_validated_collapsed.csv"))
    ap.add_argument("--full", default=str(data / "relations_validated_full.csv"))
    ap.add_argument("--merged", default=str(data / "relations_graph_ready_identifier_merged.csv"))
    ap.add_argument("--pubchem-cache", default=str(data / "pubchem_cache.json"))
    ap.add_argument("--pubtator", default="",
                    help="pubtator_relations.csv from pubtator_compare.py (enables the eligibility split)")
    args = ap.parse_args()

    for path in (args.collapsed, args.full):
        if not Path(path).exists():
            sys.exit(f"missing input: {path}")

    collapsed = pd.read_csv(args.collapsed, **NA)
    full = pd.read_csv(args.full, **NA)
    cache = json.loads(Path(args.pubchem_cache).read_text(encoding="utf-8")) \
        if Path(args.pubchem_cache).exists() else {}

    print("# Post-processing reproduction\n")

    # Stage 1-2
    clean = collapse_cleanup(collapsed)
    kept, split = retention(clean)
    print("## Cascade")
    print(f"collapsed relation rows            : {len(collapsed):,}")
    print(f"after self/directional cleanup     : {len(clean):,}")
    print(f"strict graph-ready (all criteria)  : {len(kept):,}")
    print("retention removals, by criterion:")
    print(f"  no validated pesticide           : {split['no_validated_pesticide']:,}")
    print(f"  unvalidated chemical endpoint    : {split['unvalidated_chemical']:,}")
    print(f"  generic organism endpoint        : {split['generic_organism']:,}")
    print(f"  (total removed                   : {sum(split.values()):,})")

    # Stage 3
    if Path(args.merged).exists():
        merged = pd.read_csv(args.merged, **NA)
        print(f"after identifier merge (shipped)   : {len(merged):,}")
    print()

    # Stage 4
    if cache:
        cov = chemical_coverage(full, cache)
        print("## Chemical-identifier coverage")
        print(f"distinct chemical mentions         : {cov['distinct']:,}")
        print(f"resolved (dictionary or PubChem)   : {cov['resolved']:,} "
              f"({cov['pct']:.1f}%)\n")
    else:
        print("## Chemical-identifier coverage: skipped (no PubChem cache)\n")

    # Stage 5
    if args.pubtator and Path(args.pubtator).exists():
        elig = eligibility(full, args.pubtator, cache)
        print("## Eligibility of the yield gap")
        print(f"abstracts only this workflow covers: {elig['gap']:,}")
        print(f"  of which beyond PubTator's reach : {elig['ineligible']:,} "
              f"({elig['pct']:.0f}%)\n")
    else:
        print("## Eligibility of the yield gap: skipped "
              "(pass --pubtator pubtator_relations.csv to enable)\n")


if __name__ == "__main__":
    main()
