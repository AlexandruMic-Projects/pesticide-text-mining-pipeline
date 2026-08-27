# Pesticide Text-Mining Pipeline

This repository contains the code and supporting evaluation data for the
pesticide-related text-mining pipeline presented in the accompanying paper.

## Repository contents

### `src/extraction_pipeline.py`

Main processing pipeline used to extract and process pesticide-related
entities and relations from scientific literature.

### `evaluation/qualitative_study.xlsx`

Data associated with the qualitative evaluation reported in the paper.

## Pipeline

The pipeline performs the main processing steps described in the paper,
including:

1. Processing scientific literature
2. Named entity recognition
3. Relation extraction
4. Entity and relation post-processing
5. Generation of the final relations used for analysis
