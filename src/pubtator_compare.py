#!/usr/bin/env python3
"""
pubtator_compare.py: Compare tthe relation output against PubTator 3.0's own 
annotations for the same PubMed abstracts.

Inputs
    data/pubmed_papers.csv

Outputs (under --outdir)
    pubtator_relations.csv    one row per retrieved relation
    pubtator_chemicals.csv    chemical annotations, for the coverage comparison
    pubtator_summary.md       the comparison tables

Requirements
    requests, pandas; scipy is optional (only the Mann-Whitney test uses it).

Usage
    python src/pubtator_compare.py --relations path/to/relations_readable.csv
    python src/pubtator_compare.py --relations <file> --limit 20   # quick check
    python src/pubtator_compare.py --pmids all                     # whole corpus
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS = REPO_ROOT / "data" / "pubmed_papers.csv"

BASE_URL_CANDIDATES = [
    "https://www.ncbi.nlm.nih.gov/research/pubtator-api",
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api",
]

BATCH_SIZE = 100
SLEEP_BETWEEN = 0.4
MAX_RETRIES = 4
TIMEOUT = 90
HEADERS = {"User-Agent": "pesticide-text-mining-pipeline/1.0"}

# PubTator surface forms mapped onto the label names used by the workflow.
CANON = {
    "association": "Association",
    "associate": "Association",
    "hprd-association": "HPRD-Association",
    "euadr-association": "EUADR-Association",
    "aimed-association": "AIMED-Association",
    "positive_correlation": "Positive_Correlation",
    "positive_correlate": "Positive_Correlation",
    "negative_correlation": "Negative_Correlation",
    "negative_correlate": "Negative_Correlation",
    "bind": "Bind",
    "cotreatment": "Cotreatment",
    "cotreat": "Cotreatment",
    "comparison": "Comparison",
    "compare": "Comparison",
    "drug_interaction": "Drug_Interaction",
    "drug_interact": "Drug_Interaction",
    "conversion": "Conversion",
    "inhibit": "Inhibitor",
    "inhibitor": "Inhibitor",
    "stimulate": "Stimulator",
    "cause": "Cause",
    "treat": "Treat",
    "prevent": "Prevent",
    "interact": "int",
    "ppim": "PPIm",
}

# Broad vs specific partition used for the score/specificity test.
BROAD = {"Association", "HPRD-Association", "EUADR-Association",
         "AIMED-Association", "Positive_Correlation", "Negative_Correlation"}

# BioRED-native restriction, for the robustness check.
NATIVE_BROAD = {"Association", "Positive_Correlation", "Negative_Correlation"}
NATIVE_SPEC = {"Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"}

# Labels collapsed on an unordered pair key; all others keep direction.
SYMMETRIC_LABELS = {"Association", "HPRD-Association", "EUADR-Association",
                    "AIMED-Association", "Comparison", "Bind", "PPIm", "int"}

# Entity biotypes normalised to the four workflow types, so PubTator's surface
# strings (Chemical, Species, ...) and the workflow's BioREx types
# (ChemicalEntity, OrganismTaxon, ...) can be compared on the same footing.
BIOTYPE_CANON = {
    "chemical": "chemical", "chemicalentity": "chemical",
    "gene": "gene", "geneorgeneproduct": "gene",
    "disease": "disease", "diseaseorphenotypicfeature": "disease",
    "species": "organism", "organismtaxon": "organism",
}


def canon_biotype(t):
    return BIOTYPE_CANON.get(str(t).strip().lower(), str(t).strip().lower())


def pairing_key(t1, t2):
    """Unordered entity-type pairing on the four normalised workflow types."""
    return "-".join(sorted([canon_biotype(t1), canon_biotype(t2)]))


def rank_biserial(u, n1, n2):
    """Rank-biserial correlation from a Mann-Whitney U (U is for group 1).
    r = 2 * P(x1 > x2) - 1, where P is estimated as U / (n1 * n2).
    Positive r means group 1 tends to score above group 2."""
    if not n1 or not n2:
        return None
    return 2.0 * (u / (n1 * n2)) - 1.0


def canon(label):
    if not label:
        return "UNKNOWN"
    return CANON.get(str(label).strip().lower(), str(label).strip())


def read_pmids(path, column="pmid"):
    df = pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False)
    if column not in df.columns:
        sys.exit(f"Column '{column}' not found in {path}")
    pmids = (df[column].astype(str).str.strip()
             .str.replace(r"\.0$", "", regex=True))
    pmids = sorted({p for p in pmids if p.isdigit()})
    print(f"[corpus] {len(pmids)} unique PMIDs from {os.path.basename(path)}")
    return pmids


def load_pmids(args):
    if args.pmids == "all":
        return read_pmids(args.papers)
    if not args.relations:
        sys.exit("--pmids relation_producing needs --relations "
                 "(the PMIDs are taken from the workflow's relation output)")
    return read_pmids(args.relations)


def parse_payload(text):
    """Accept either a dict keyed 'PubTator3' or newline-delimited JSON."""
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("PubTator3", "pubtator3", "documents"):
                if key in obj:
                    return obj[key]
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def fetch_batch(pmids, base_url):
    params = {"pmids": ",".join(pmids), "full": "false"}
    url = f"{base_url}/publications/export/biocjson"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return parse_payload(r.text)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"    HTTP {r.status_code}; retrying in {wait}s")
                time.sleep(wait)
                last_err = f"HTTP {r.status_code}"
                continue
            return None
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"    {type(exc).__name__}; retrying in {wait}s")
            time.sleep(wait)
            last_err = str(exc)
    raise RuntimeError(f"batch failed after {MAX_RETRIES} attempts: {last_err}")


def pick_base_url(sample_pmids):
    for base in BASE_URL_CANDIDATES:
        try:
            recs = fetch_batch(sample_pmids, base)
            if isinstance(recs, list) and recs:
                print(f"[api] using {base}")
                return base
        except Exception as exc:
            print(f"[api] {base} failed: {exc}")
    sys.exit("No PubTator base URL responded. Check network access.")


def fetch_all(pmids, outdir):
    cache_dir = os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    batches = [pmids[i:i + BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]

    base = None
    records = []
    for i, batch in enumerate(batches):        
        tag = hashlib.md5(",".join(batch).encode()).hexdigest()[:10]
        cache_path = os.path.join(cache_dir, f"batch_{i:04d}_{tag}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                recs = json.load(fh)
            print(f"[{i + 1}/{len(batches)}] cached ({len(recs)} records)")
        else:
            if base is None:
                base = pick_base_url(batch[:5])
            recs = fetch_batch(batch, base)
            if not isinstance(recs, list):
                recs = []
            with open(cache_path, "w") as fh:
                json.dump(recs, fh)
            print(f"[{i + 1}/{len(batches)}] fetched {len(recs)} records")
            time.sleep(SLEEP_BETWEEN)
        records.extend(recs)
    return records


def extract(records):
    """Pull relations and chemical annotations out of the BioC-JSON payload."""
    rel_rows, ent_rows, seen_pmids = [], [], set()

    for rec in records:
        pmid = str(rec.get("pmid") or rec.get("id") or "").split("|")[0].strip()
        if not pmid:
            continue
        seen_pmids.add(pmid)

        for rel in rec.get("relations", []) or []:
            infons = rel.get("infons", {}) or {}
            r1 = infons.get("role1", {}) or {}
            r2 = infons.get("role2", {}) or {}
            try:
                score = float(infons.get("score"))
            except (TypeError, ValueError):
                score = None
            rel_rows.append({
                "pmid": pmid,
                "raw_type": infons.get("type"),
                "label": canon(infons.get("type")),
                "score": score,
                "role1_name": r1.get("name"),
                "role1_biotype": r1.get("biotype"),
                "role1_id": r1.get("identifier"),
                "role2_name": r2.get("name"),
                "role2_biotype": r2.get("biotype"),
                "role2_id": r2.get("identifier"),
            })

        for passage in rec.get("passages", []) or []:
            for ann in passage.get("annotations", []) or []:
                ai = ann.get("infons", {}) or {}
                biotype = (ai.get("biotype") or ai.get("type") or "").lower()
                if biotype != "chemical":
                    continue
                ident = ai.get("identifier") or ai.get("normalized_id") or ""
                ent_rows.append({
                    "pmid": pmid,
                    "text": ann.get("text"),
                    "identifier": ident,
                    "normalised": bool(ident) and str(ident).lower() not in ("-", "none", ""),
                })

    return pd.DataFrame(rel_rows), pd.DataFrame(ent_rows), seen_pmids


def collapse_edges(rel):
    """Collapse relation records into edges using the following rule: an
    unordered identifier pair for symmetric labels, a directed pair otherwise,
    then group by (pair, label). Keyed on role identifiers, not names."""
    if not len(rel):
        return pd.DataFrame(columns=["a", "b", "label", "records", "n_pmids", "score"])

    keys = []
    for r1, r2, lab in zip(rel["role1_id"], rel["role2_id"], rel["label"]):
        a, b = str(r1), str(r2)
        if lab in SYMMETRIC_LABELS:
            a, b = sorted([a, b])
        keys.append((a, b))
    rel = rel.assign(a=[k[0] for k in keys], b=[k[1] for k in keys])

    edges = (rel.groupby(["a", "b", "label"])
             .agg(records=("pmid", "size"),
                  n_pmids=("pmid", lambda s: s.nunique()),
                  score=("score", "max"),
                  role1_biotype=("role1_biotype", "first"),
                  role2_biotype=("role2_biotype", "first"))
             .reset_index())
    edges["pairing"] = [pairing_key(t1, t2)
                        for t1, t2 in zip(edges["role1_biotype"], edges["role2_biotype"])]
    return edges


def load_our_relations(path):
    """Read the workflow's relation output and adapt to its schema. The
    graph-ready file carries collapsed edges scored with max_pred_score; the
    readable file carries relation rows scored with pred_score."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False)
    if "max_pred_score" in df.columns:
        score_col, unit = "max_pred_score", "edges"
    elif "pred_score" in df.columns:
        score_col, unit = "pred_score", "relation rows"
    else:
        score_col, unit = None, "rows"

    if "pred_label" not in df.columns:
        sys.exit(f"{path} has no 'pred_label' column; not a relation output file")

    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    counts = df["pred_label"].value_counts()
    top3 = list(counts.head(3).index)
    n = len(df)
    labels_above_1pct = int((counts / n * 100 > 1).sum()) if n else 0

    per_pair, shared_top3 = {}, None
    if {"entity1_type", "entity2_type"}.issubset(df.columns):
        pairing = pd.Series(
            [pairing_key(a, b) for a, b in zip(df["entity1_type"], df["entity2_type"])],
            index=df.index)
        paired = df.assign(pair=pairing)
        for pair, grp in paired.groupby("pair"):
            per_pair[pair] = {
                "n": len(grp),
                "share": len(grp) / n * 100,
                "top3_share": grp["pred_label"].isin(top3).mean() * 100,
            }        
        shared = paired[paired["pair"].str.contains("chemical")
                        & ~paired["pair"].str.contains("organism")]
        if len(shared):
            sc = shared["pred_label"].value_counts()
            shared_top3 = sc.head(3).sum() / len(shared) * 100

    density = None
    rc = pd.to_numeric(df["relation_count"], errors="coerce") if "relation_count" in df.columns else None
    upc = pd.to_numeric(df["unique_pmid_count"], errors="coerce") if "unique_pmid_count" in df.columns else None
    if rc is not None:
        density = {
            "single_row": (rc == 1).mean() * 100,
            "mean_rows": rc.mean(),
            "single_abstract": (upc == 1).mean() * 100 if upc is not None else None,
        }

    scores = None
    if score_col:
        scores = (df[["pred_label", score_col]]
                  .rename(columns={"pred_label": "label", score_col: "score"})
                  .dropna(subset=["score"]))

    return {
        "unit": unit,
        "total": n,
        "distinct_labels": int(counts.size),
        "labels_above_1pct": labels_above_1pct,
        "top3_share": counts.head(3).sum() / n * 100 if n else 0.0,
        "top3": counts.head(3).to_dict(),
        "shared_top3": shared_top3,
        "per_pair": per_pair,
        "density": density,
        "scores": scores,
    }


def mannwhitney(a, b):
    """Return (U, p) for group a against group b, or None if scipy is absent.
    U is the statistic for a, so it feeds rank_biserial directly."""
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        return None


def _mw_line(a, b):
    """Format 'U = ..., p = ..., r = ...' for group a against group b."""
    res = mannwhitney(a, b)
    if res is None:
        return "(scipy unavailable)"
    u, p = res
    r = rank_biserial(u, len(a), len(b))
    frac = (r + 1) / 2 if r is not None else None
    tail = f", r = {r:.2f} (broad outscores specific in {frac * 100:.0f}% of pairings)" \
        if r is not None else ""
    return f"U = {u:,.0f}, p = {p:.3g}{tail}"


def score_block(add, scored, unit_label):
    b = scored[scored["label"].isin(BROAD)]
    s = scored[~scored["label"].isin(BROAD)]
    add(f"- Broad labels: n = {len(b):,}, median = {b.score.median():.4f}"
        if len(b) else "- Broad labels: none")
    add(f"- Specific labels: n = {len(s):,}, median = {s.score.median():.4f}"
        if len(s) else "- Specific labels: none")
    if len(b) and len(s):
        add(f"- Mann-Whitney ({unit_label}): {_mw_line(b.score, s.score)}")
    bn = scored[scored["label"].isin(NATIVE_BROAD)]
    sn = scored[scored["label"].isin(NATIVE_SPEC)]
    if len(bn) and len(sn):
        add(f"- BioRED-native restriction: broad n = {len(bn):,} "
            f"(median {bn.score.median():.4f}) vs specific n = {len(sn):,} "
            f"(median {sn.score.median():.4f}); {_mw_line(bn.score, sn.score)}")
    add("")


def summarise(rel, ent, edges, seen, requested, ours, outdir):
    L = []
    add = L.append

    add("# PubTator 3.0 comparison on the study corpus\n")
    add(f"- PMIDs requested: **{len(requested):,}**")
    add(f"- PMIDs returned by PubTator: **{len(seen):,}**")
    add(f"- Not returned: **{len(set(requested) - seen):,}**")
    add(f"- Relation records retrieved: **{len(rel):,}**")
    if not len(rel):
        add("\n**No relations returned; check the API response.**\n")
        Path(outdir, "pubtator_summary.md").write_text("\n".join(L))
        return
    add(f"- Collapsed edges: **{len(edges):,}**")
    add(f"- PMIDs with >=1 relation: **{rel.pmid.nunique():,}**\n")

    # Q1: label distribution
    add("## Q1. Relation label distribution\n")
    rec_counts = rel["label"].value_counts()
    edge_counts = edges["label"].value_counts()
    edge_above_1pct = int((edge_counts / len(edges) * 100 > 1).sum())
    add(f"Records: {rec_counts.size} distinct labels, "
        f"top-3 share {rec_counts.head(3).sum() / len(rel) * 100:.1f}%.")
    add(f"Edges: {edge_counts.size} distinct labels, "
        f"{edge_above_1pct} above 1% of edges, "
        f"top-3 share {edge_counts.head(3).sum() / len(edges) * 100:.1f}%.\n")
    add("| Label | Records | Edges | Edge share |")
    add("|---|---:|---:|---:|")
    for lab in edge_counts.index:
        add(f"| {lab} | {int(rec_counts.get(lab, 0)):,} | {int(edge_counts[lab]):,} "
            f"| {edge_counts[lab] / len(edges) * 100:.2f}% |")
    add("")
    if ours:
        add(f"This workflow, for comparison ({ours['unit']}):\n")
        add(f"- {ours['total']:,} {ours['unit']}, {ours['distinct_labels']} distinct labels, "
            f"{ours['labels_above_1pct']} above 1%, "
            f"top-3 share **{ours['top3_share']:.1f}%**")
        add(f"- top three: {ours['top3']}\n")

    # Q2: score vs specificity
    add("## Q2. Score vs label specificity\n")
    add("PubTator scores are normalised confidences in [0, 1]. This workflow's "
        "scores are BioREx maximum logits and are not comparable in level, only "
        "in the direction of the broad/specific gap.\n")
    scored = rel.dropna(subset=["score"])
    if len(scored):
        add(f"PubTator, record level (n = {len(scored):,}, "
            f"range {scored.score.min():.4f}-{scored.score.max():.4f}):")
        score_block(add, scored, "records")
        edge_scored = edges.dropna(subset=["score"])
        if len(edge_scored):
            add(f"PubTator, edge level (n = {len(edge_scored):,}, score = max over the "
                f"records collapsed into each edge):")
            score_block(add, edge_scored, "edges")
        add("Per-label median record score:\n")
        add("| Label | n | Median |")
        add("|---|---:|---:|")
        for lab, grp in scored.groupby("label"):
            add(f"| {lab} | {len(grp):,} | {grp.score.median():.4f} |")
        add("")
    else:
        add("No numeric scores present in the response.\n")
    if ours and ours["scores"] is not None and len(ours["scores"]):
        add(f"This workflow ({ours['unit']}, maximum logit):")
        score_block(add, ours["scores"], ours["unit"])

    # Q3: chemical identifier coverage
    add("## Q3. Chemical identifier coverage\n")
    add("Restricted to chemical mentions: a structural identifier is undefined "
        "for organisms, genes and diseases.\n")
    if len(ent):
        key = ent["text"].astype(str).str.strip().str.lower()
        norm = ent["normalised"].astype(str).str.lower().eq("true")
        by_key = norm.groupby(key).any()
        distinct = len(by_key)
        resolved = int(by_key.sum())
        pct = resolved / distinct * 100 if distinct else 0.0
        add(f"- PubTator distinct chemical mentions (case-folded): **{distinct:,}**")
        add(f"- Carrying a normalised MeSH identifier: **{pct:.1f}%**")
        add(f"- Chemical annotations in total: {len(ent):,}\n")
    add("This workflow's chemical normalisation coverage is produced by the "
        "pesticide-dictionary and PubChem validation stage and is reported in the "
        "paper; it is not recomputed here.\n")

    # Q3b: evidential density
    add("## Q3b. Evidential density\n")
    add("Support behind each collapsed edge: contributing relation records, and "
        "the number of distinct abstracts those records come from.\n")
    if len(edges):
        single_abs = (edges["n_pmids"] == 1).mean() * 100
        single_row = (edges["records"] == 1).mean() * 100
        mean_rows = edges["records"].mean()
        add(f"PubTator: {single_abs:.1f}% of edges rest on a single abstract, "
            f"{single_row:.1f}% on a single record, mean {mean_rows:.2f} records per edge.")
    if ours and ours["density"]:
        d = ours["density"]
        sa = f"{d['single_abstract']:.1f}%" if d["single_abstract"] is not None else "n/a"
        add(f"This workflow ({ours['unit']}): {sa} of edges rest on a single abstract, "
            f"{d['single_row']:.1f}% on a single record, mean {d['mean_rows']:.2f} records per edge.")
    add("")

    # Q4: entity-type pairings
    add("## Q4. Entity-type pairings\n")
    rel = rel.assign(pair=rel.role1_biotype.astype(str) + "-" + rel.role2_biotype.astype(str))
    pairs = rel["pair"].value_counts()
    add("PubTator:\n")
    add("| Pair | Records | Share |")
    add("|---|---:|---:|")
    for pr, n in pairs.head(15).items():
        add(f"| {pr} | {n:,} | {n / len(rel) * 100:.1f}% |")
    add("")
    sp = rel[rel.role1_biotype.astype(str).str.lower().eq("species")
             | rel.role2_biotype.astype(str).str.lower().eq("species")]
    add(f"**Relations with a species endpoint: {len(sp):,} "
        f"({len(sp) / len(rel) * 100:.1f}% of {len(rel):,}).**\n")
    
    shared_pairs = {p for p in edges["pairing"].unique()
                    if "organism" not in p and "chemical" in p}
    if ours and ours["per_pair"]:
        ours_pairs = {p for p in ours["per_pair"]
                      if "organism" not in p and "chemical" in p}
        shared_pairs &= ours_pairs
    shared_edges = edges[edges["pairing"].isin(shared_pairs)]
    if len(shared_edges):
        sc = shared_edges["label"].value_counts()
        pt_shared_top3 = sc.head(3).sum() / len(shared_edges) * 100
        add(f"Top-three share over shared pairings "
            f"({', '.join(sorted(shared_pairs))}): PubTator {pt_shared_top3:.1f}%"
            + (f", this workflow {ours['shared_top3']:.1f}%."
               if ours and ours.get("shared_top3") is not None else "."))
        add("")

    if ours and ours["per_pair"]:
        add(f"This workflow, by pairing ({ours['unit']}), with the share carried by "
            f"its top three labels:\n")
        add("| Pair | Count | Share | Top-3 share |")
        add("|---|---:|---:|---:|")
        for pr, d in sorted(ours["per_pair"].items(), key=lambda kv: -kv[1]["n"]):
            add(f"| {pr} | {d['n']:,} | {d['share']:.1f}% | {d['top3_share']:.1f}% |")
        add("")
        add("The chemical-chemical, chemical-gene and chemical-disease rows are the "
            "shared pairings; chemical-organism edges have no PubTator counterpart.\n")

    text = "\n".join(L)
    Path(outdir, "pubtator_summary.md").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relations",
                    help="workflow relation output (relations_readable.csv or a "
                         "collapsed graph-ready file); source of the "
                         "relation-producing PMIDs and of this workflow's side")
    ap.add_argument("--papers", default=str(DEFAULT_PAPERS),
                    help="processed abstract corpus (default: data/pubmed_papers.csv)")
    ap.add_argument("--pmids", choices=["relation_producing", "all"],
                    default="relation_producing",
                    help="which PMIDs to query PubTator for")
    ap.add_argument("--outdir", default="pubtator_comparison")
    ap.add_argument("--limit", type=int, default=0,
                    help="only fetch the first N PMIDs (quick check)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pmids = load_pmids(args)
    if args.limit:
        pmids = pmids[:args.limit]
        print(f"[limit] restricted to {len(pmids)} PMIDs")

    records = fetch_all(pmids, args.outdir)
    print(f"[parse] {len(records)} records returned")

    rel, ent, seen = extract(records)
    edges = collapse_edges(rel)
    rel.to_csv(os.path.join(args.outdir, "pubtator_relations.csv"), index=False)
    if len(ent):
        ent.to_csv(os.path.join(args.outdir, "pubtator_chemicals.csv"), index=False)

    ours = load_our_relations(args.relations) if args.relations else None
    summarise(rel, ent, edges, seen, pmids, ours, args.outdir)
    print(f"\nWritten to {args.outdir}/")


if __name__ == "__main__":
    main()
