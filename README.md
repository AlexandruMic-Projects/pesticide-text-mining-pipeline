# Pesticide Text-Mining Pipeline

This repository contains the code, input data, and supporting evaluation data for the pesticide-related text-mining pipeline presented in the accompanying paper.

## Repository contents

### `src/Extraction_pipeline.py`

Main processing pipeline used to extract and process pesticide-related entities and relations from scientific literature.

### `data/pubmed_papers.csv`

Input dataset used by the text-mining pipeline.

The dataset contains 2,526 scientific articles retrieved from PubMed. Relevant literature was identified through keyword-based searches, followed by manual reading of the retrieved abstracts to assess their relevance for the study.

The dataset includes the information required as input to the extraction pipeline:

* `pmid` – PubMed identifier
* `title` – article title
* `abstract` – article abstract

### `evaluation/Qualitative_study.xlsx`

Data associated with the qualitative evaluation reported in the paper.

## Pipeline

The pipeline performs the main processing steps described in the paper, including:

1. Processing scientific literature
2. Named entity recognition
3. Relation extraction
4. Entity and relation post-processing
5. Generation of the final relations used for analysis

## Models and installation

The pipeline uses **AIONER** for named entity recognition and **BioREx** for relation extraction.

Installation instructions, model files, dependencies, and documentation on how to run these models are available in their official NCBI repositories:

* **AIONER:** https://github.com/ncbi/AIONER/
* **BioREx:** https://github.com/ncbi/BioREx/

These repositories should be consulted for the setup and use of the respective models.
