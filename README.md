To reproduce our results for the variant effect prediction evaluation, follow these instructions:

1. Install a Python virtual environment with the necessary packages and activate it by running `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

2. Download the PhyloGPN checkpoint:
    - `gdown "https://drive.google.com/uc?id=1MSxLYbZKSnWjbM_w1cHGVFffrwh8j64V" -O ./PhyloGPN/checkpoint.pt`


2. Download the required raw data by running the following commands:
    - `invoke download-hg38`
    - `invoke download-omim`
    - `invoke download-latest-clinvar`
    - `invoke download-dms-data`
    - `invoke download-and-process-gnomad`

3. Process the raw data by running the following commands:
    - `invoke chunk-hg38`
    - `invoke process-clinvar`
    - `invoke process-dms-data`

4. Generate log-likelihood ratios for models:
    - `invoke generate-vep-results --model phylogpn`
    - `invoke generate-vep-results --model caduceus_131k`
    - `invoke generate-vep-results --model hyenadna_medium_160k`
    - `invoke generate-vep-results --model nucleotide_transformer_v2_500m`

4. Process the results:
    - `invoke merge-clinvar-results`
    - `invoke merge-dms-results`
    - `invoke merge-omim-gnomad-results`
