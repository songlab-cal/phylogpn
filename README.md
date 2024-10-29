# PhyloGPN: A Phylogenetic Approach to Genomic Language Modeling
A framework for training genomic language models (gLMs) that controls for phylogenetic biases

## Setup
To setup the environment for PhyloGPN, follow these instructions:

1. Install a Python virtual environment with the necessary packages and activate it by running `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

2. Download the PhyloGPN checkpoint:
    - `gdown "https://drive.google.com/uc?id=1MSxLYbZKSnWjbM_w1cHGVFffrwh8j64V" -O ./PhyloGPN/checkpoint.pt`

For a tutorial on how to obtain likelihoods, rate parameters, and embeddings from PhyloGPN, refer to  `notebooks/example.ipynb`

## Embedding-Based Evaluation

For instructions on how to benchmark PhyloGPN and PhyloGPN-pooled in BEND, 
refer to  `notebooks/bend.ipynb`

## Likelihood-Based Evaluation

To reproduce our results for the variant effect prediction evaluation, follow these instructions:

1. Download the required raw data:
    - `invoke download-hg38`
    - `invoke download-omim`
    - `invoke download-latest-clinvar`
    - `invoke download-dms-data`
    - `invoke download-and-process-gnomad`

2. Process the raw data:
    - `invoke chunk-hg38`
    - `invoke process-clinvar`
    - `invoke process-dms-data`

3. Generate log likelihood ratios:
    - `invoke generate-vep-results --model phylogpn`
    - `invoke generate-vep-results --model caduceus_131k`
    - `invoke generate-vep-results --model hyenadna_medium_160k`
    - `invoke generate-vep-results --model nucleotide_transformer_v2_500m`

4. Process the results:
    - `invoke merge-clinvar-results`
    - `invoke merge-dms-results`
    - `invoke merge-omim-results`

The results should be in `data/clinvar_eval.csv`, `data/omim_eval.csv`, and `data/dms_eval.csv`.


