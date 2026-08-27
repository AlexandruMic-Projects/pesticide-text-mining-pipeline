import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# CONFIG FROM ENV
# =========================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path.home() / "grad_project"))).resolve()
CSV_PATH = Path(os.environ["CSV_PATH"]).resolve()
PEST_DICT_PATH = Path(os.environ["PEST_DICT_PATH"]).resolve()

AIONER_DIR = Path(os.environ["AIONER_DIR"]).resolve()
BIOREX_DIR = Path(os.environ["BIOREX_DIR"]).resolve()
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"]).resolve()

AIONER_SRC_DIR = AIONER_DIR / "src"
AIONER_PYTHON = os.environ.get("AIONER_PYTHON", sys.executable)
BIOREX_PYTHON = os.environ.get("BIOREX_PYTHON", sys.executable)

AIONER_MODEL_SOURCE = AIONER_DIR / "pretrained_models" / "AIONER" / "Bioformer-softmax-AIONER.h5"
AIONER_MODEL_BASENAME = "Bioformer-softmax-AIONER.h5"
AIONER_MODEL_RUNTIME = AIONER_SRC_DIR / AIONER_MODEL_BASENAME
AIONER_VOCAB_ARG = "../vocab/AIO_label.vocab"

BIOREX_MODEL_DIRNAME = os.environ.get("BIOREX_MODEL_DIRNAME", "pretrained_model_biolinkbert")
START_ROW_ENV = os.environ.get("START_ROW", "0").strip()
START_ROW = int(START_ROW_ENV) if START_ROW_ENV else 0

SAVE_DECODED_ALL = os.environ.get("SAVE_DECODED_ALL", "False").strip().lower() in ("1", "true", "yes")
SAVE_ENTITY_TABLES = os.environ.get("SAVE_ENTITY_TABLES", "False").strip().lower() in ("1", "true", "yes")

# None = full dataset
START_ROW = int(os.environ.get("START_ROW", "0"))

N_ROWS_ENV = os.environ.get("N_ROWS", "").strip()
N_ROWS = None if N_ROWS_ENV in ("", "None", "none", "ALL", "all") else int(N_ROWS_ENV)
print("START_ROW:", START_ROW)

# batch AIONER input so huge datasets do not become one giant run folder
AIONER_BATCH_SIZE = int(os.environ.get("AIONER_BATCH_SIZE", "500"))

GENERIC_CHEMICALS = {
    "pesticide", "pesticides",
    "herbicide", "herbicides",
    "fungicide", "fungicides",
    "insecticide", "insecticides",
    "acaricide", "acaricides",
    "chemical", "chemicals",
    "residue", "residues",
    "neonicotinoid", "neonicotinoids",
    "qoi", "qoi fungicides"
}

BIORED_TYPE_MAP = {
    "Chemical": "ChemicalEntity",
    "Disease": "DiseaseOrPhenotypicFeature",
    "Gene": "GeneOrGeneProduct",
    "Species": "OrganismTaxon",
}

LABELS = [
    'None', 'Association', 'Bind', 'Comparison', 'Conversion', 'Cotreatment',
    'Drug_Interaction', 'Negative_Correlation', 'Positive_Correlation',
    'None-CID', 'CID', 'None-PPIm', 'PPIm', 'None-AIMED', 'None-DDI',
    'None-BC7', 'None-phargkb', 'None-GDA', 'None-DISGENET', 'None-EMU_BC',
    'None-EMU_PC', 'None-HPRD50', 'None-PHARMGKB', 'ACTIVATOR', 'AGONIST',
    'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR', 'ANTAGONIST',
    'DIRECT-REGULATOR', 'INDIRECT-DOWNREGULATOR', 'INDIRECT-UPREGULATOR',
    'INHIBITOR', 'PART-OF', 'PRODUCT-OF', 'SUBSTRATE',
    'SUBSTRATE_PRODUCT-OF', 'mechanism', 'int', 'effect', 'advise',
    'AIMED-Association', 'HPRD-Association', 'EUADR-Association',
    'None-EUADR', 'Indirect_conversion', 'Non_conversion',
    'None-Conversion'
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== PATH CHECK ===")
print("CSV_PATH:", CSV_PATH, CSV_PATH.exists())
print("PEST_DICT_PATH:", PEST_DICT_PATH, PEST_DICT_PATH.exists())
print("AIONER_DIR:", AIONER_DIR, AIONER_DIR.exists())
print("BIOREX_DIR:", BIOREX_DIR, BIOREX_DIR.exists())
print("OUTPUT_DIR:", OUTPUT_DIR)
print("AIONER_PYTHON:", AIONER_PYTHON)
print("BIOREX_PYTHON:", BIOREX_PYTHON)
print("START_ROW:", START_ROW)
print("N_ROWS:", N_ROWS)
print("AIONER_BATCH_SIZE:", AIONER_BATCH_SIZE)
print("SAVE_DECODED_ALL:", SAVE_DECODED_ALL)
print("SAVE_ENTITY_TABLES:", SAVE_ENTITY_TABLES)
print("==================")

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def safe_fill(df):
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["pmid"] = df["pmid"].astype(str)
    df["text"] = df["title"] + "\n" + df["abstract"]
    return df

def run_cmd(cmd, cwd=None, env=None, allow_fail=False):
    print("\nRUNNING:")
    print(" ".join(map(str, cmd)))
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=env
    )
    if result.stdout:
        print("\nSTDOUT:\n", result.stdout[:12000])
    if result.stderr:
        print("\nSTDERR:\n", result.stderr[:12000])
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result

def load_csv(csv_path, start_row=0, n_rows=None):
    """
    Load a chunk of the input CSV.

    start_row = first data row to process, excluding the header.
    n_rows = number of rows to process. If None, process from start_row until end.
    """
    if start_row == 0:
        df = pd.read_csv(csv_path, nrows=n_rows)
    else:
        df = pd.read_csv(
            csv_path,
            skiprows=range(1, start_row + 1),
            nrows=n_rows
        )

    return safe_fill(df)

def load_pesticide_terms(path):
    terms = set()
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        header = [h.lower().strip() for h in header]
        name_idx = header.index("name")
        syn_idx = header.index("synonyms")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if name_idx < len(parts):
                name = normalize(parts[name_idx])
                if name:
                    terms.add(name)

            if syn_idx < len(parts) and parts[syn_idx].strip():
                for syn in parts[syn_idx].split(";"):
                    syn = normalize(syn)
                    if syn:
                        terms.add(syn)
    return terms

def ensure_aioner_runtime_files():
    if not AIONER_MODEL_SOURCE.exists():
        raise FileNotFoundError(f"Missing source model: {AIONER_MODEL_SOURCE}")
    if not AIONER_MODEL_RUNTIME.exists():
        shutil.copy2(AIONER_MODEL_SOURCE, AIONER_MODEL_RUNTIME)
        print("Copied runtime model to:", AIONER_MODEL_RUNTIME)
    else:
        print("Runtime model already exists:", AIONER_MODEL_RUNTIME)

def write_aioner_input_files(df, input_dir):
    input_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        pmid = str(row["pmid"])
        title = str(row["title"])
        abstract = str(row["abstract"])
        out_path = input_dir / f"{pmid}.pubtator"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{pmid}|t|{title}\n")
            f.write(f"{pmid}|a|{abstract}\n\n")

def run_aioner(input_dir, output_dir, entity_type="ALL"):
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_aioner_runtime_files()

    env = os.environ.copy()

    # IMPORTANT:
    # Do not force AIONER to CPU.
    # Let Slurm expose the allocated GPUs.
    # Optional override: export AIONER_CUDA_VISIBLE_DEVICES="0,1"
    if "AIONER_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["AIONER_CUDA_VISIBLE_DEVICES"]

    cmd = [
        AIONER_PYTHON,
        "AIONER_Run.py",
        "-i", str(input_dir) + "/",
        "-m", AIONER_MODEL_BASENAME,
        "-v", AIONER_VOCAB_ARG,
        "-e", entity_type,
        "-o", str(output_dir) + "/",
    ]

    return run_cmd(cmd, cwd=AIONER_SRC_DIR, env=env)

def parse_aioner_pubtator_output(output_dir, source_df):
    source_lookup = (
        source_df[["pmid", "title", "abstract", "text"]]
        .drop_duplicates(subset=["pmid"])
        .copy()
    )
    source_lookup["pmid"] = source_lookup["pmid"].astype(str)
    source_lookup = source_lookup.set_index("pmid").to_dict(orient="index")

    rows = []
    for fpath in output_dir.iterdir():
        if not fpath.is_file():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if "|t|" in line or "|a|" in line:
                    continue

                parts = line.split("\t")
                if len(parts) < 5:
                    continue

                pmid = str(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                mention = parts[3]
                entity_label = parts[4]

                meta = source_lookup.get(pmid, {"title": "", "abstract": "", "text": ""})
                rows.append({
                    "pmid": pmid,
                    "title": meta["title"],
                    "abstract": meta["abstract"],
                    "full_text": meta["text"],
                    "text": mention,
                    "entity_label": entity_label,
                    "score": np.nan,
                    "start": start,
                    "end": end
                })
    return pd.DataFrame(rows)

def batched(iterable_df, batch_size):
    n = len(iterable_df)
    for start in range(0, n, batch_size):
        yield start, iterable_df.iloc[start:start + batch_size].copy()

def extract_entities_with_aioner(df, working_dir, entity_type="ALL"):
    all_parts = []
    for start_idx, batch_df in batched(df, AIONER_BATCH_SIZE):
        batch_dir = working_dir / f"aioner_batch_{start_idx:07d}"
        input_dir = batch_dir / "input"
        output_dir = batch_dir / "output"

        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== AIONER batch {start_idx} .. {start_idx + len(batch_df) - 1} ===")
        write_aioner_input_files(batch_df, input_dir)
        run_aioner(input_dir, output_dir, entity_type=entity_type)
        batch_entities = parse_aioner_pubtator_output(output_dir, batch_df)
        all_parts.append(batch_entities)

    if not all_parts:
        return pd.DataFrame()

    return pd.concat(all_parts, ignore_index=True)

def prepare_entities(entities_df, pesticide_terms):
    df = entities_df.copy()
    df["text_norm"] = df["text"].apply(normalize)
    df["is_pesticide"] = df["text_norm"].isin(pesticide_terms)
    df["is_generic_chemical"] = (
        df["entity_label"].astype(str).str.lower().eq("chemical") &
        df["text_norm"].isin(GENERIC_CHEMICALS)
    )
    df["pubtator_type"] = df["entity_label"].map(BIORED_TYPE_MAP)

    entities_all_df = df.drop_duplicates(
        subset=["pmid", "start", "end", "text", "entity_label"]
    ).copy()

    entities_biorex_df = entities_all_df.dropna(subset=["pubtator_type"]).copy()
    return entities_all_df, entities_biorex_df

def verify_offsets(entities_df, source_df):
    kept = []
    dropped = []

    source_unique = (
        source_df[["pmid", "title", "abstract", "text"]]
        .drop_duplicates(subset=["pmid"])
        .copy()
    )
    source_unique["pmid"] = source_unique["pmid"].astype(str)
    source_lookup = source_unique.set_index("pmid").to_dict(orient="index")

    for _, ent in entities_df.iterrows():
        pmid = str(ent["pmid"])
        if pmid not in source_lookup:
            dropped.append({
                "pmid": pmid, "text": ent["text"], "start": ent["start"],
                "end": ent["end"], "substring": None, "reason": "pmid_not_found"
            })
            continue

        full_text = source_lookup[pmid]["text"]
        start = int(ent["start"])
        end = int(ent["end"])
        surface = str(ent["text"])

        if 0 <= start < end <= len(full_text) and full_text[start:end] == surface:
            kept.append(ent.to_dict())
        else:
            dropped.append({
                "pmid": pmid,
                "text": surface,
                "start": start,
                "end": end,
                "substring": full_text[start:end] if 0 <= start < end <= len(full_text) else None,
                "reason": "offset_mismatch"
            })

    return pd.DataFrame(kept), pd.DataFrame(dropped)
  
def sanitize_for_biorex(df):
    """
    Fixes issues that crash BioREx:
    - multiple labels per same span → keep one
    - overlapping spans → keep longest
    """

    if df.empty:
        return df

    df = df.copy()

    # --- 1. One label per span ---
    df = df.sort_values(["pmid", "start", "end"])
    df = df.drop_duplicates(subset=["pmid", "start", "end"], keep="first")

    # --- 2. Remove overlaps (keep longest span) ---
    cleaned = []

    for pmid, group in df.groupby("pmid"):
        group = group.sort_values(["start", "end"])
        kept = []

        for _, row in group.iterrows():
            start, end = row["start"], row["end"]

            overlap = False
            for k in kept:
                if not (end <= k["start"] or start >= k["end"]):
                    # overlap exists → keep longer span
                    current_len = end - start
                    kept_len = k["end"] - k["start"]

                    if current_len > kept_len:
                        k.update(row.to_dict())
                    overlap = True
                    break

            if not overlap:
                kept.append(row.to_dict())

        cleaned.extend(kept)

    return pd.DataFrame(cleaned)
  
def strict_offset_filter(entities_df, source_df):
    source_lookup = (
        source_df[["pmid", "text"]]
        .drop_duplicates(subset=["pmid"])
        .set_index("pmid")["text"]
        .to_dict()
    )

    keep = []

    for _, row in entities_df.iterrows():
        pmid = str(row["pmid"])
        if pmid not in source_lookup:
            continue

        text = source_lookup[pmid]
        start = int(row["start"])
        end = int(row["end"])
        surface = str(row["text"])

        if 0 <= start < end <= len(text) and text[start:end] == surface:
            keep.append(row)

    return pd.DataFrame(keep)

def write_biored_pubtator(source_df, entities_df, out_path):
    source_df = source_df.copy()
    source_df["pmid"] = source_df["pmid"].astype(str)

    entities_df = entities_df.copy()
    entities_df["pmid"] = entities_df["pmid"].astype(str)

    lines = []
    docs_written = 0
    ann_lines_written = 0

    for _, row in source_df.iterrows():
        pmid = row["pmid"]
        title = str(row["title"])
        abstract = str(row["abstract"])

        doc_entities = entities_df[entities_df["pmid"] == pmid].copy()
        doc_entities = doc_entities.sort_values(["start", "end"])

        doc_entities = doc_entities[
            (doc_entities["start"] < doc_entities["end"]) &
            (doc_entities["start"] >= 0)
        ]

        if len(doc_entities) == 0:
            continue

        lines.append(f"{pmid}|t|{title}")
        lines.append(f"{pmid}|a|{abstract}")
        docs_written += 1

        for i, (_, ent) in enumerate(doc_entities.iterrows(), start=1):
            start = int(ent["start"])
            end = int(ent["end"])
            text = str(ent["text"])
            ent_type = str(ent["pubtator_type"])
            norm_id = f"{ent_type}:{pmid}:{i}"

            lines.append(
                f"{pmid}\t{start}\t{end}\t{text}\t{ent_type}\t{norm_id}"
            )
            ann_lines_written += 1

        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("BioREx input written:", out_path)
    print("BioREx input exists:", out_path.exists())
    print("BioREx input size:", out_path.stat().st_size if out_path.exists() else 0)
    print("Documents written to BioREx input:", docs_written)
    print("Annotation lines written:", ann_lines_written)

    if not out_path.exists():
        raise FileNotFoundError(f"BioREx input was not created: {out_path}")

    if out_path.stat().st_size == 0:
        raise RuntimeError(
            f"BioREx input was created but is empty: {out_path}. "
            "This means no documents had valid BioREx-compatible entities."
        )

def run_biorex_prediction(pubtator_in, biorex_dir):
    """
    Runs BioREx prediction for one chunk.

    Important:
    All temporary and model output files are written inside OUTPUT_DIR,
    not inside the shared BioREx folder. This makes chunked Slurm jobs safer.
    """
    biorex_work_dir = OUTPUT_DIR / "biorex_runtime"
    biorex_work_dir.mkdir(parents=True, exist_ok=True)

    out_processed_tsv = biorex_work_dir / "out_processed.tsv"
    pred_tsv = biorex_work_dir / "pred_test_results.tsv"
    biorex_model_out = biorex_work_dir / "biorex_model"
    test_results_path = biorex_model_out / "test_results.tsv"

    for p in [out_processed_tsv, pred_tsv]:
        if p.exists():
            p.unlink()

    if biorex_model_out.exists():
        shutil.rmtree(biorex_model_out)

    run_cmd([
        BIOREX_PYTHON,
        "src/dataset_format_converter/convert_pubtator_2_tsv.py",
        "--exp_option", "biored_pred",
        "--in_pubtator_file", str(pubtator_in),
        "--out_tsv_file", str(out_processed_tsv)
    ], cwd=biorex_dir)

    if not out_processed_tsv.exists() or out_processed_tsv.stat().st_size == 0:
        print("No BioREx candidates were generated. Writing empty outputs.")
        return None, None

    run_cmd([
        BIOREX_PYTHON,
        "src/run_ncbi_rel_exp.py",
        "--task_name", "biorex",
        "--test_file", str(out_processed_tsv),
        "--use_balanced_neg", "False",
        "--to_add_tag_as_special_token", "True",
        "--model_name_or_path", BIOREX_MODEL_DIRNAME,
        "--output_dir", str(biorex_model_out),
        "--num_train_epochs", "10",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "32",
        "--do_predict",
        "--logging_steps", "10",
        "--evaluation_strategy", "steps",
        "--save_steps", "10",
        "--overwrite_output_dir",
        "--max_seq_length", "512"
    ], cwd=biorex_dir)

    if not test_results_path.exists():
        raise RuntimeError("BioREx prediction finished but test_results.tsv was not created.")

    shutil.copy2(test_results_path, pred_tsv)
    return out_processed_tsv, pred_tsv

def decode_biorex_predictions(cand_df, pred_scores_df, labels):
    score_matrix = pred_scores_df.values
    pred_idx = score_matrix.argmax(axis=1)

    max_idx = int(pred_idx.max()) if len(pred_idx) else -1
    if max_idx >= len(labels):
        raise RuntimeError(
            f"Prediction index {max_idx} exceeds LABELS length {len(labels)}. "
            f"Your label mapping does not match the BioREx model output dimension "
            f"({score_matrix.shape[1]} classes)."
        )

    pred_label = [labels[i] for i in pred_idx]
    pred_score = score_matrix.max(axis=1)

    decoded_df = cand_df.copy()
    decoded_df["pred_label"] = pred_label
    decoded_df["pred_score"] = pred_score
    return decoded_df

def keep_non_none_relations(decoded_df):
    return decoded_df[~decoded_df["pred_label"].astype(str).str.startswith("None")].copy()

def parse_input_pubtator_annotations(pubtator_path):
    anns = []
    with open(pubtator_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|t|" in line or "|a|" in line:
                continue
            parts = line.split("\t")
            if len(parts) == 6:
                pmid, start, end, text, ent_type, ent_id = parts
                anns.append({
                    "pmid": str(pmid),
                    "entity_id": ent_id,
                    "entity_text": text,
                    "entity_type": ent_type
                })
    return pd.DataFrame(anns).drop_duplicates()

def make_all_relations_readable(positive_df, pubtator_in, papers_df, pesticide_terms):
    ann_input_df = parse_input_pubtator_annotations(pubtator_in)

    rel_df = positive_df.copy()
    rel_df["pmid"] = rel_df["pmid"].astype(str)

    ann1 = ann_input_df.rename(columns={
        "entity_id": "id1",
        "entity_text": "entity1_text",
        "entity_type": "entity1_type"
    })
    ann2 = ann_input_df.rename(columns={
        "entity_id": "id2",
        "entity_text": "entity2_text",
        "entity_type": "entity2_type"
    })

    rel_df = rel_df.merge(ann1, on=["pmid", "id1"], how="left")
    rel_df = rel_df.merge(ann2, on=["pmid", "id2"], how="left")

    rel_df["entity1_is_pesticide"] = rel_df["entity1_text"].apply(
        lambda x: normalize(x) in pesticide_terms if pd.notna(x) else False
    )
    rel_df["entity2_is_pesticide"] = rel_df["entity2_text"].apply(
        lambda x: normalize(x) in pesticide_terms if pd.notna(x) else False
    )

    papers_meta = papers_df.copy()
    papers_meta["pmid"] = papers_meta["pmid"].astype(str)

    rel_df = rel_df.merge(
        papers_meta[["pmid", "title", "abstract"]],
        on="pmid",
        how="left"
    )

    return rel_df[
        [
            "pmid", "title", "abstract",
            "type1", "type2",
            "id1", "entity1_text", "entity1_type",
            "pred_label",
            "id2", "entity2_text", "entity2_type",
            "pred_score",
            "same_sentence", "sentence_window",
            "entity1_is_pesticide", "entity2_is_pesticide"
        ]
    ].copy()

def write_relation_pubtator(source_pubtator, positive_df, out_path):
    """
    Writes a PubTator-style file with original entity lines plus relation lines:
    PMID<TAB>RELTYPE<TAB>Arg1:ID1<TAB>Arg2:ID2
    """
    with open(source_pubtator, "r", encoding="utf-8") as f:
        original = f.read().rstrip("\n")

    rel_lines = []
    for _, row in positive_df.iterrows():
        rel_lines.append(
            f"{row['pmid']}\t{row['pred_label']}\tArg1:{row['id1']}\tArg2:{row['id2']}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        if original:
            f.write(original)
            f.write("\n")
        for line in rel_lines:
            f.write(line + "\n")

def main():
    papers_df = load_csv(CSV_PATH, start_row=START_ROW, n_rows=N_ROWS)
    pesticide_terms = load_pesticide_terms(PEST_DICT_PATH)

    print(f"Loaded papers: {len(papers_df)}")
    print(f"Loaded pesticide terms: {len(pesticide_terms)}")

    # ---------- NER ----------
    entities_raw_df = extract_entities_with_aioner(
        papers_df,
        working_dir=OUTPUT_DIR,
        entity_type="ALL"
    )

    entities_all_df, entities_biorex_df = prepare_entities(entities_raw_df, pesticide_terms)
    entities_all_valid_df, entities_all_dropped_df = verify_offsets(entities_all_df, papers_df)
    entities_biorex_valid_df, entities_biorex_dropped_df = verify_offsets(entities_biorex_df, papers_df)

    # 🔥 NEW STEP (critical)
    # entities_biorex_valid_df = strict_offset_filter(entities_biorex_valid_df, papers_df)
    entities_biorex_valid_df = sanitize_for_biorex(entities_biorex_valid_df)

    if SAVE_ENTITY_TABLES:
        entities_raw_df.to_csv(OUTPUT_DIR / "entities_raw.csv", index=False)
        entities_all_df.to_csv(OUTPUT_DIR / "entities_all_prepared.csv", index=False)
        entities_biorex_df.to_csv(OUTPUT_DIR / "entities_biorex_prepared.csv", index=False)
        entities_all_valid_df.to_csv(OUTPUT_DIR / "entities_all_valid.csv", index=False)
        entities_biorex_valid_df.to_csv(OUTPUT_DIR / "entities_biorex_valid.csv", index=False)
    else:
        print("Skipping large entity tables because SAVE_ENTITY_TABLES=False")

    entities_all_dropped_df.to_csv(OUTPUT_DIR / "entities_all_dropped.csv", index=False)
    entities_biorex_dropped_df.to_csv(OUTPUT_DIR / "entities_biorex_dropped.csv", index=False)

    print("Raw entities:", len(entities_raw_df))
    print("BioREx-valid entities:", len(entities_biorex_valid_df))

    # ---------- BIOREX INPUT ----------
    pubtator_in = OUTPUT_DIR / "biorex_input.pubtator"
    write_biored_pubtator(papers_df, entities_biorex_valid_df, pubtator_in)
    
    if not pubtator_in.exists():
        raise FileNotFoundError(f"Missing BioREx input after writing: {pubtator_in}")

    print("Confirmed BioREx input before prediction:", pubtator_in)

    # ---------- RELATION EXTRACTION ----------
    out_processed_tsv, pred_tsv = run_biorex_prediction(pubtator_in, BIOREX_DIR)

    if out_processed_tsv is None or pred_tsv is None:
        print("No relation candidates. Writing empty relation outputs.")
        pd.DataFrame().to_csv(OUTPUT_DIR / "relations_decoded_all.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_DIR / "relations_positive_only.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_DIR / "relations_readable.csv", index=False)
        write_relation_pubtator(pubtator_in, pd.DataFrame(columns=["pmid", "pred_label", "id1", "id2"]), OUTPUT_DIR / "biorex_output.pubtator")
        print("Pipeline finished cleanly.")
        return

    cand_df = pd.read_csv(out_processed_tsv, sep="\t", header=None)
    pred_scores_df = pd.read_csv(pred_tsv, sep="\t", header=None)

    cand_df.columns = [
        "pmid", "type1", "type2", "id1", "id2",
        "same_sentence", "sentence_window", "model_input",
        "neighbors", "label"
    ]

    if len(cand_df) != len(pred_scores_df):
        raise RuntimeError(
            f"Candidate/prediction row mismatch: {len(cand_df)} vs {len(pred_scores_df)}"
        )

    decoded_df = decode_biorex_predictions(cand_df, pred_scores_df, LABELS)
    positive_df = keep_non_none_relations(decoded_df)

    if SAVE_DECODED_ALL:
        decoded_df.to_csv(
        OUTPUT_DIR / "relations_decoded_all.csv.gz",
        index=False,
        compression="gzip"
        )
    else:
        print("Skipping relations_decoded_all.csv because SAVE_DECODED_ALL=False")

    positive_df.to_csv(OUTPUT_DIR / "relations_positive_only.csv", index=False)

    relations_readable_df = make_all_relations_readable(
        positive_df=positive_df,
        pubtator_in=pubtator_in,
        papers_df=papers_df,
        pesticide_terms=pesticide_terms
    )
    relations_readable_df.to_csv(OUTPUT_DIR / "relations_readable.csv", index=False)

    # Save a final relation PubTator-like file WITHOUT depending on run_pubtator_eval.py
    final_pubtator = OUTPUT_DIR / "biorex_output.pubtator"
    write_relation_pubtator(pubtator_in, positive_df, final_pubtator)

    print("Decoded rows:", len(decoded_df))
    print("Positive relations:", len(positive_df))
    print("Final PubTator:", final_pubtator)
    print("Pipeline finished cleanly.")

if __name__ == "__main__":
    main()
