# Pesticide Text-Mining Pipeline

This repository contains the code, input data, derived relation tables, pesticide reference dictionary, and supporting evaluation data for the pesticide-related text-mining pipeline presented in the accompanying paper.

## Repository contents

### `src/extraction_pipeline.py`

Main processing pipeline used to extract and process pesticide-related entities and relations from scientific literature.

### `src/reproduce_postprocessing.py`

Post-processing over the validated relation tables. It removes self-relations and directional duplicates, applies the retention criteria used to build the graph-ready relation set, confirms the final edge count, and reports the chemical-identifier coverage. Reads the tables in `data/` and requires only `pandas`.

### `src/pubtator_compare.py`

Compares the relation output of this workflow against PubTator 3.0's own annotations for the same abstracts. Requires `pandas` and `requests` (and, optionally, `scipy`), and needs internet access to the PubTator API.

### `data/pubmed_papers.csv`

Input dataset used by the text-mining pipeline.

The dataset contains scientific articles retrieved from PubMed. Relevant literature was identified through keyword-based searches, followed by manual reading of the retrieved abstracts to assess their relevance for the study.

The dataset contains:

* `pmid` – PubMed identifier
* `title` – article title
* `abstract` – article abstract

### Derived relation tables (`data/`)

Relation tables produced by the pipeline, provided so that the post-processing and comparison scripts can be run without re-executing AIONER and BioREx.

* `relations_validated_full.csv` – row-level relation output after `None`-label removal and pesticide-dictionary and PubChem validation. One row per extracted relation, with both entities and their types, the predicted relation label, the prediction score, sentence-proximity information, validation flags, and the assigned dictionary and PubChem identifiers. The `title` and `abstract` columns are omitted from this copy to reduce file size and can be recovered from the `pmid`.
* `relations_validated_collapsed.csv` – the same relations after collapsing duplicate rows that share the same entity pair and relation label.
* `relations_graph_ready_strict.csv` – the relation set after retention filtering, where each row is one candidate graph edge.
* `relations_graph_ready_identifier_merged.csv` – the final graph edge set, after merging endpoints that resolve to the same chemical identifier. This is the file loaded by the dashboard.

### PubChem caches (`data/`)

* `pubchem_cache.json`, `pubchem_graph_ready_cache.json` – cached PubChem lookups used during chemical validation and identifier merging, so these steps can be reproduced without new PubChem queries.

### `resources/pesticide_dictionary.tsv`

Reference pesticide dictionary used during entity post-processing and pesticide identification.

The dictionary contains:

* `name`
* `synonyms`

Multiple synonyms are separated by semicolons.

### `evaluation/Qualitative_study.xlsx`

Data associated with the qualitative evaluation reported in the accompanying paper.

Some responses take the literal value `None`. When reading the file with pandas, pass `keep_default_na=False` so that these values are preserved rather than treated as missing.

## Pipeline

The pipeline performs the following main steps:

1. Load the PubMed literature dataset.
2. Convert article titles and abstracts into PubTator input files.
3. Perform biomedical named entity recognition using AIONER.
4. Validate entity offsets and prepare BioREx-compatible entities.
5. Identify pesticide entities using the pesticide reference dictionary.
6. Convert the recognized entities and articles into BioREx-compatible PubTator input.
7. Generate relation candidates and perform relation extraction using BioREx.
8. Decode the BioREx predictions and remove predictions corresponding to `None` relation classes.
9. Generate readable relation tables and PubTator-formatted relation output.

## External models

The pipeline relies on two external NCBI tools:

* **AIONER** for named entity recognition:
  https://github.com/ncbi/AIONER/

* **BioREx** for relation extraction:
  https://github.com/ncbi/BioREx/

The original repositories contain the model files, dependency specifications, and additional documentation.

Because AIONER and BioREx use different software environments, it is recommended to install them in separate Python environments.

## AIONER installation

Clone the official AIONER repository:

```bash
git clone https://github.com/ncbi/AIONER.git
cd AIONER
```

AIONER was tested by its authors using Python 3.7. Its documented main dependencies include TensorFlow 2.3.0, Transformers 4.18.0, and Stanza 1.4.0.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The AIONER pre-trained models must also be downloaded following the instructions in the official AIONER repository:

https://github.com/ncbi/AIONER/

The pipeline uses:

```text
pretrained_models/AIONER/Bioformer-softmax-AIONER.h5
```

and the vocabulary file:

```text
vocab/AIO_label.vocab
```

The `pretrained_models` folder should therefore be placed inside the cloned AIONER repository as described in the AIONER documentation.

The pipeline runs AIONER with entity type:

```text
ALL
```

allowing the model to recognize the biomedical entity types supported by AIONER.

## BioREx installation

Clone the official BioREx repository:

```bash
git clone https://github.com/ncbi/BioREx.git
cd BioREx
```

The current BioREx repository specifies Python 3.12 and supports Linux or WSL2.

The environment can be created according to the official BioREx instructions:

```bash
conda create -n biorex python=3.12
conda activate biorex

pip install --upgrade pip setuptools wheel

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

BioREx provides pre-trained models through its official repository:

https://github.com/ncbi/BioREx/

This pipeline uses the **BioREx BioLinkBERT model**, which is the preferred pre-trained model listed by BioREx.

After downloading and extracting the model, the default expected model directory is:

```text
pretrained_model_biolinkbert
```

inside the BioREx repository.

A different model directory can be supplied through the `BIOREX_MODEL_DIRNAME` environment variable.

## Python requirements for the pipeline script

The main `Extraction_pipeline.py` script directly requires:

```text
numpy
pandas
```

AIONER and BioREx should use the dependencies specified by their respective repositories.

The `reproduce_postprocessing.py` script requires `pandas`. The `pubtator_compare.py` script requires `pandas` and `requests`, and optionally `scipy`.

## Running the pipeline

The pipeline is configured through environment variables.

### Required environment variables

The following variables must be defined:

* `CSV_PATH` – path to `data/pubmed_papers.csv`
* `PEST_DICT_PATH` – path to `resources/pesticide_dictionary.tsv`
* `AIONER_DIR` – path to the cloned and configured AIONER repository
* `BIOREX_DIR` – path to the cloned and configured BioREx repository
* `OUTPUT_DIR` – directory in which pipeline results will be written

### Python environments

Because AIONER and BioREx may require different Python versions and dependency environments, their Python executables can be specified independently:

* `AIONER_PYTHON` – Python executable from the AIONER environment
* `BIOREX_PYTHON` – Python executable from the BioREx environment

If these variables are not specified, the pipeline uses the Python executable with which `Extraction_pipeline.py` itself was started.

### Additional optional variables

* `BIOREX_MODEL_DIRNAME` – BioREx model directory; default: `pretrained_model_biolinkbert`
* `START_ROW` – first data row to process; default: `0`
* `N_ROWS` – number of articles to process; if omitted, all remaining articles are processed
* `AIONER_BATCH_SIZE` – number of articles included in each AIONER batch; default: `500`
* `SAVE_DECODED_ALL` – save the complete set of decoded BioREx predictions; default: `False`
* `SAVE_ENTITY_TABLES` – save intermediate entity tables; default: `False`
* `AIONER_CUDA_VISIBLE_DEVICES` – optional GPU selection for AIONER

## Example configuration

Example on Linux:

```bash
export CSV_PATH=/path/to/pesticide-text-mining-pipeline/data/pubmed_papers.csv
export PEST_DICT_PATH=/path/to/pesticide-text-mining-pipeline/resources/pesticide_dictionary.tsv

export AIONER_DIR=/path/to/AIONER
export BIOREX_DIR=/path/to/BioREx

export OUTPUT_DIR=/path/to/pipeline_output

export AIONER_PYTHON=/path/to/aioner/environment/bin/python
export BIOREX_PYTHON=/path/to/biorex/environment/bin/python
```

The complete dataset can then be processed using:

```bash
python src/Extraction_pipeline.py
```

If `N_ROWS` is not specified, the complete input dataset is processed.

## Processing a smaller subset

For testing purposes, only part of the dataset can be processed.

For example:

```bash
export START_ROW=0
export N_ROWS=10

python src/Extraction_pipeline.py
```

This processes only the first 10 articles.

To return to processing the full dataset, remove or unset `N_ROWS`.

## AIONER processing

For each article, the pipeline combines the title and abstract and generates PubTator-formatted input.

AIONER is then executed through:

```text
AIONER_Run.py
```

using:

```text
Bioformer-softmax-AIONER.h5
AIO_label.vocab
entity type = ALL
```

The resulting named entities are read back into the pipeline. Entity offsets are checked against the original article text, duplicate and incompatible annotations are handled, and supported entity types are converted into the terminology required by BioREx.

## BioREx processing

The pipeline automatically prepares PubTator input containing the original article text and AIONER annotations.

BioREx processing is performed using the BioREx scripts:

```text
src/dataset_format_converter/convert_pubtator_2_tsv.py
```

followed by:

```text
src/run_ncbi_rel_exp.py
```

The pipeline therefore handles the conversion from AIONER output to BioREx input automatically.

Users do not need to manually run the BioREx prediction scripts when using `Extraction_pipeline.py`.

## Pipeline output

The main output files are:

### `relations_positive_only.csv`

Contains BioREx relation predictions after predictions whose labels begin with `None` have been removed.

### `relations_readable.csv`

Human-readable relation table containing information including:

* PMID
* article title
* article abstract
* entity 1
* entity 1 type
* entity 2
* entity 2 type
* predicted relation
* BioREx prediction score
* sentence proximity information
* whether either entity was identified as a pesticide

### `biorex_output.pubtator`

Final PubTator-style output containing entity annotations and predicted relations.

## Intermediate outputs

The pipeline also creates intermediate files required for communication between AIONER and BioREx and for validation.

These include:

```text
biorex_input.pubtator
entities_all_dropped.csv
entities_biorex_dropped.csv
```

When:

```text
SAVE_ENTITY_TABLES=True
```

the pipeline additionally saves:

```text
entities_raw.csv
entities_all_prepared.csv
entities_biorex_prepared.csv
entities_all_valid.csv
entities_biorex_valid.csv
```

When:

```text
SAVE_DECODED_ALL=True
```

the complete decoded BioREx prediction table is also saved in compressed form.

## Post-processing and comparison scripts

Once the relation tables in `data/` are available (either produced by the extraction pipeline or taken from this repository), the two additional scripts can be run without AIONER or BioREx installed.

Post-processing (runs offline):

```bash
python src/reproduce_postprocessing.py
```

Comparison against PubTator 3.0 (requires internet access):

```bash
python src/pubtator_compare.py --relations data/relations_validated_full.csv
```

The comparison writes its outputs to a `pubtator_comparison/` directory. Both scripts also print their results to the console.

## Reproducing the workflow

To reproduce the extraction workflow:

1. Clone this repository.
2. Clone and install AIONER using the instructions provided by NCBI.
3. Download the AIONER pre-trained models and ensure that `Bioformer-softmax-AIONER.h5` is available.
4. Clone and install BioREx using the instructions provided by NCBI.
5. Download and extract the BioREx BioLinkBERT pre-trained model.
6. Use `data/pubmed_papers.csv` as the literature input.
7. Use `resources/pesticide_dictionary.tsv` as the pesticide reference dictionary.
8. Define the required environment variables.
9. Run `src/Extraction_pipeline.py`.
10. Retrieve the resulting relation tables from the configured output directory.

The repository provides the study input literature, derived relation tables, pesticide reference dictionary, extraction and post-processing code, and evaluation data needed to reproduce the workflow described in the accompanying paper.
