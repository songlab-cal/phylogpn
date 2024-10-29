from collections import defaultdict
import gzip
import multiprocessing
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, Iterator, List
import glob
from multiprocessing import Pool
import allel

import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.special import softmax, expit, logsumexp
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
import torch.nn.functional as F
import yaml
from invoke import task
from tqdm import tqdm
from cyvcf2 import VCF
from torch.utils.data import DataLoader
import urllib.request
from torch.nn.utils.rnn import pad_sequence
import pyBigWig
import h5py

from fasta_utils import ChunkedSequenceReader, chunk_fasta
from models import HyenaDNA, NucleotideTransformer, NucleotideTransformerV2, PhyloGPN, Caduceus
from data import GenomeChunkDataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


@task
def download_hg38(context):
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    urllib.request.urlretrieve(url, "hg38.fa.gz")


@task
def download_omim(context):
    df = pd.read_parquet("hf://datasets/songlab/omim/test.parquet")
    mask = df["label"] == True
    df = df.loc[mask]
    df["id"] = "chr" + df["chrom"] + ":g." + df["pos"].astype(str) + df["ref"] + ">" + df["alt"]
    
    with open(config["data"]["omim"], "w") as f:
        for line in df["id"]:
            f.write(f"{line}\n")

@task
def download_latest_clinvar(context):
    output_dir_path = config["data"]["clinvar_dir"]

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    tbi_url = vcf_url + ".tbi"

    for url in [vcf_url, tbi_url]:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_path = output_dir_path + "/" + url.split("/")[-1]

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


@task
def process_clinvar(context):
    rows = []
    chrom_list = [str(i) for i in range(1, 23)] + ["X", "Y"]

    review_status_to_num_stars = {
        "criteria_provided,_single_submitter": 1,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "no_assertion_criteria_provided": 0,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_conflicting_interpretations": 1,
        "practice_guideline": 4,
    }

    input_file_path = config["data"]["clinvar_dir"] + "/clinvar.vcf.gz"

    for variant in VCF(input_file_path):
        skip = variant.INFO.get("CLNVC") != "single_nucleotide_variant"
        skip |= len(variant.ALT) != 1
        skip |= variant.INFO.get("MC") is None
        skip |= variant.INFO.get("CLNSIG") is None
        skip |= variant.CHROM not in chrom_list
        skip |= variant.ALT is None

        if skip:
            continue

        skip |= variant.ALT[0] == "N"
        skip |= variant.INFO.get("CLNSIG").lower() not in [
            "benign",
            "likely_benign",
            "benign/likely_benign",
            "pathogenic",
            "likely_pathogenic",
            "pathogenic/likely_pathogenic",
        ]

        if skip:
            continue

        row = {}

        for key in variant.INFO.get("CLNSIG").lower().split("/"):
            row[key] = 1

        for key in [x.split("|")[1].replace("_variant", "") for x in variant.INFO.get("MC").split(",")]:
            key = key.lower().replace("non-coding", "noncoding")

            if key == "intron":
                key = "intronic"
            elif key == "genic_upstream_transcript":
                key = "upstream_of_transcript"
            elif key == "genic_downstream_transcript":
                key = "downstream_of_transcript"
            elif key == "initiator_codon":
                key = "start_codon"
            elif key == "noncoding_transcript":
                key = "noncoding"

            if key == "no_sequence_alteration":
                continue

            row[key] = 1

        row["id"] = f"chr{variant.CHROM}:g.{variant.POS}{variant.REF}>{ variant.ALT[0]}"

        row["num_stars"] = review_status_to_num_stars[variant.INFO.get("CLNREVSTAT")]

        rows.append(row)

    df = pd.DataFrame(rows)
    cols = [
        "id",
        "num_stars",
        "benign",
        "likely_benign",
        "pathogenic",
        "likely_pathogenic",
        "noncoding",
        "intronic",
        "synonymous",
        "missense",
        "nonsense",
        "5_prime_utr",
        "3_prime_utr",
        "splice_donor",
        "splice_acceptor",
        "start_codon",
        "stop_lost",
        "upstream_of_transcript",
        "downstream_of_transcript",
    ]
    df = df[cols]
    df.fillna(0, inplace=True)
    df[cols[1:]] = df[cols[1:]].astype(int)

    df.to_csv(config["data"]["processed_clinvar"], index=False)


@task
def download_dms_data(context):
    output_dir_path = config["data"]["dms_dir"]

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    base_names = [
        "DMS_A4_HUMAN_Seuma_2021",
        "DMS_ADRB2_HUMAN_Jones_2020",
        "DMS_BRCA1_HUMAN_Findlay_2018",
        "DMS_CALM1_HUMAN_Weile_2017",
        "DMS_CP2C9_HUMAN_Amorosi_abundance_2021",
        "DMS_CP2C9_HUMAN_Amorosi_activity_2021",
        "DMS_DLG4_HUMAN_Faure_2021",
        "DMS_GRB2_HUMAN_Faure_2021",
        "DMS_KCNH2_HUMAN_Kozek_2020",
        "DMS_MK01_HUMAN_Brenan_2016",
        "DMS_MSH2_HUMAN_Jia_2020",
        "DMS_NUD15_HUMAN_Suiter_2020",
        "DMS_P53_HUMAN_Giacomelli_NULL_Etoposide_2018",
        "DMS_P53_HUMAN_Giacomelli_NULL_Nutlin_2018",
        "DMS_P53_HUMAN_Giacomelli_WT_Nutlin_2018",
        "DMS_P53_HUMAN_Kotler_2018",
        "DMS_PTEN_HUMAN_Matreyek_2021",
        "DMS_PTEN_HUMAN_Mighell_2018",
        "DMS_SC6A4_HUMAN_Young_2021",
        "DMS_SCN5A_HUMAN_Glazer_2019",
        "DMS_SRC_HUMAN_Ahler_CD_2019",
        "DMS_SUMO1_HUMAN_Weile_2017",
        "DMS_SYUA_HUMAN_Newberry_2020",
        "DMS_TADBP_HUMAN_Bolognesi_2019",
        "DMS_TPK1_HUMAN_Weile_2017",
        "DMS_TPMT_HUMAN_Matreyek_2018",
        "DMS_TPOR_HUMAN_Bridgford_S505N_2020",
        "DMS_UBC9_HUMAN_Weile_2017",
        "DMS_VKOR1_HUMAN_Chiasson_abundance_2020",
        "DMS_VKOR1_HUMAN_Chiasson_activity_2020",
        "DMS_YAP1_HUMAN_Araya_2012",
    ]

    for base_name in base_names:
        print(f"Downloading data for `{base_name}`...")

        url = f"https://kircherlab.bihealth.org/download/CADD-development/v1.7/validation/esm/{base_name}.vcf.gz"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if request was successful

        base_name_parts = base_name.split("_")
        formatted_base_name = f"{base_name_parts[1]}_{base_name_parts[3]}_{base_name_parts[-1]}"

        for x in base_name_parts[4:-1]:
            formatted_base_name += f"_{x}"

        formatted_base_name = formatted_base_name.lower()

        output_file_path = f"{output_dir_path}/{formatted_base_name}.vcf.gz"

        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


@task
def process_dms_data(context):
    file_paths = glob.glob(config["data"]["dms_dir"] + "/*.vcf.gz")
    df_list = []

    for file_path in file_paths:
        base_name = os.path.basename(file_path).split(".")[0]

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            dtype={"chrom": "str"},
            names=["chrom", "pos", "id", "ref", "alt", "label"],
        )
        df["id"] = "chr" + df["chrom"] + ":g." + df["pos"].astype(str) + df["ref"] + ">" + df["alt"]
        df["study"] = base_name
        df_list.append(df[["study", "id", "label"]])

    df = pd.concat(df_list)
    df.to_csv(config["data"]["processed_dms"], index=False)


def process_gnomad_vcf(input_path: str, output_path: str, chunk_size: int = 100000):
    fields = ["CHROM", "POS", "REF", "ALT", "PASS", "AC", "AN",]
    groups = ["afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "sas"]

    for group in groups:
        fields.append(f"AC_{group}")
        fields.append(f"AN_{group}")

    *__, chunks = allel.iter_vcf_chunks(input_path, fields=fields, chunk_length=chunk_size)
    df_list = []

    try:
        for chunk in chunks:
            data = {k.lstrip("variants/").lower(): v.tolist() for k, v in chunk[0].items()}

            for key in ["alt", "ac"] + [f"ac_{p}" for p in groups]:
                data[key] = [x[0] for x in data[key]]

            # Filter out non-substitutions
            df = pd.DataFrame(data)
            mask = (df["an"] > 0) & (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1) & df["filter_pass"]
            df = df.loc[mask]
            df["id"] = df["chrom"] + ":g." + df["pos"].astype(str) + df["ref"] + ">" + df["alt"]

            cols = ["id", "ac", "an"]

            for group in groups:
                cols.append(f"ac_{group}")
                cols.append(f"an_{group}")

            df = df[cols]
            df_list.append(df)
    except EOFError:
        pass
    
    df = pd.concat(df_list).reset_index(drop=True)
    compression = "gzip" if output_path.endswith(".gz") else None
    df.to_csv(output_path, index=False, compression=compression)


@task
def download_and_process_gnomad(context):
    output_dir_path = config["data"]["gnomad_dir"]
    os.makedirs(output_dir_path, exist_ok=True)

    chroms = [str(x) for x in range(1, 23)] + ["X", "Y"]
    urls = [f"https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/genomes/gnomad.genomes.v4.1.sites.chr{x}.vcf.bgz" for x in chroms]

    for url, chrom in zip(urls, chroms):
        file_name = f"chr{chrom}.csv.gz"
        output_file_path = Path(output_dir_path) / file_name
        touch_path = Path(output_dir_path) / f"{file_name}.downloaded"

        if touch_path.exists():
            print(f"Skipping `{url}`, already downloaded.")
            return

        print(f"Downloading data for `chr{chrom}`...")

        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with NamedTemporaryFile(dir=output_dir_path, suffix=".vcf.bgz") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                
                print(f"Processing data for `chr{chrom}`...")
                
                output_file_path = f"{output_dir_path}/chr{chrom}.csv.gz"
                process_gnomad_vcf(temp_file.name, output_file_path)
                touch_path.touch()
        else:
            raise Exception(f"Failed to download `{url}`")
        

@task
def chunk_hg38(context, chunk_size: int = 1000000, line_length: int = 80):
    output_dir_path = config["data"]["genome_chunk_dir"]
    index_file_path = config["data"]["genome_chunk_index"]

    index_df_list = []

    def format_hg38_source(x: str) -> str:
        chrom, *x = x.split("_")

        if x:
            return x[0].replace("v", ".")

        return chrom
    
    genome = "hg38"
    genome_file_path = config["data"]["hg38"]

    index_df = chunk_fasta(genome_file_path, output_dir_path, chunk_size, line_length)

    if genome == "hg38":
        index_df["src"] = index_df["src"].map(format_hg38_source)

    index_df["src"] = genome + "." + index_df["src"]
    index_df_list.append(index_df)

    pd.concat(index_df_list).to_csv(index_file_path, index=False)


@task
def generate_vep_results(context, model: str, batch_size: int = 1, num_workers: int = 1):
    output_dir_path = config["data"]["result_dir"]

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Get list of variants to keep
    # Otherwise, results take up too much space
    file_paths = [config["data"]["processed_clinvar"], config["data"]["processed_dms"]]
    file_paths += glob.glob(config["data"]["gnomad_dir"] + "/*.csv.gz")

    variants_to_keep = []

    for file_path in file_paths:
        if file_path.endswith(".gz"):
            variant_df = pd.read_csv(file_path, compression="gzip")
        else:
            variant_df = pd.read_csv(file_path)
        
        if "ac" in variant_df.columns:
            variant_df["af"] = variant_df["ac"] / variant_df["an"]
            mask = variant_df["af"] >= 1e-4
            variant_df = variant_df.loc[mask]
        
        variants_to_keep.extend(variant_df["id"].tolist())

    with open(config["data"]["omim"], "r") as f:
        variants_to_keep += [line.strip() for line in f]

    model_name = model

    if model_name == "nucleotide_transformer_500m_human_reference":
        model = NucleotideTransformer(id_="500m-human-ref")
        chunk_size = 3000
        context_size = 6 * (1000 - 1) - chunk_size + 1
    elif model_name == "nucleotide_transformer_2.5b_multi_species":
        model = NucleotideTransformer(id_="2.5b-multi-species")
        chunk_size = 3000
        context_size = 6 * (1000 - 1) - chunk_size + 1
    elif model_name == "nucleotide_transformer_v2_50m":
        model = NucleotideTransformerV2(id_="50m")
        chunk_size = 3000
        context_size = 6 * (2048 - 1) - chunk_size + 1
    elif model_name == "nucleotide_transformer_v2_100m":
        model = NucleotideTransformerV2(id_="100m")
        chunk_size = 3000
        context_size = 6 * (2048 - 1) - chunk_size + 1
    elif model_name == "nucleotide_transformer_v2_250m":
        model = NucleotideTransformerV2(id_="250m")
        chunk_size = 3000
        context_size = 6 * (2048 - 1) - chunk_size + 1
    elif model_name == "nucleotide_transformer_v2_500m":
        model = NucleotideTransformerV2(id_="500m")
        chunk_size = 3000
        context_size = 6 * (2048 - 1) - chunk_size + 1
    elif model_name == "hyenadna_large":
        assert batch_size == 1
        chunk_size = 300000
        model = HyenaDNA(id_="large-1m", chunk_size=chunk_size)
        context_size = 2 * (1000000 - chunk_size) + 1
    elif model_name == "hyenadna_medium_450k":
        chunk_size = 3000
        model = HyenaDNA(id_="medium-450k", chunk_size=chunk_size)
        context_size = 2 * (450000 - chunk_size) + 1
    elif model_name == "hyenadna_medium_160k":
        chunk_size = 3000
        model = HyenaDNA(id_="medium-160k", chunk_size=chunk_size)
        context_size = 2 * (160000 - chunk_size) + 1
    elif model_name == "hyenadna_small":
        chunk_size = 3000
        model = HyenaDNA(id_="small-32k", chunk_size=chunk_size)
        context_size = 2 * (32768 - chunk_size) + 1
    elif model_name == "hyenadna_tiny":
        chunk_size = 300
        model = HyenaDNA(id_="tiny-1k", chunk_size=chunk_size)
        context_size = 2 * (1024 - chunk_size) + 1
    elif model_name == "caduceus_131k":
        chunk_size = 3000
        model = Caduceus(id_="131k", chunk_size=chunk_size)
        context_size = (131072 - 2) - chunk_size + 1
    elif model_name == "phylogpn":
        model = PhyloGPN()
        chunk_size = 10000
        context_size = model.context_size
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    chroms = [f"chr{x}" for x in range(1, 23)] + ["chrX", "chrY"]
    filter_ = lambda x: x in chroms

    for chrom in chroms:
        file_path = f"{output_dir_path}/{model_name}_{chrom}.csv"

        if os.path.exists(file_path):
            continue

        filter_ = lambda x: x == "hg38." + chrom
        dataset = GenomeChunkDataset(
            config["data"]["genome_chunk_dir"],
            config["data"]["genome_chunk_index"],
            chunk_size,
            context_size,
            filter_,
            pad_char="N",
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        current_idx = 0
        result_df_list = []

        for batch in tqdm(data_loader, desc=f"Processing {chrom}"):
            for result_df in model(batch):
                if not result_df.empty:
                    result_df["idx"] -= context_size // 2
                    mask = result_df["idx"] >= 0
                    mask &= result_df["idx"] < chunk_size
                    result_df = result_df.loc[mask].copy()
                    result_df["idx"] += current_idx
                    result_df_list.append(result_df)
                    
                current_idx += chunk_size

        result_df = pd.concat(result_df_list, ignore_index=True)
        result_df["id"] = f"{chrom}:g." + (result_df["idx"] + 1).astype(str) + result_df["ref"] + ">" + result_df["alt"]
        result_df = result_df[["id", "log_likelihood_ratio"]]
        variants_to_keep_ = set([x for x in variants_to_keep if x.startswith(chrom)])
        mask = result_df["id"].isin(variants_to_keep_)
        result_df = result_df.loc[mask]
        result_df.to_csv(file_path, index=False)


@task
def merge_clinvar_results(context):
    file_paths = glob.glob(f"{config['data']['result_dir_path']}/*.csv")
    file_paths = [x for x in file_paths if "full" not in x]
    model_to_result_df_list = defaultdict(list)

    for file_path in file_paths:
        base_name = file_path.split("/")[-1].rstrip(".csv")
        *model_name_parts, chrom = base_name.split("_")
        model_name = "_".join(model_name_parts)

        df = pd.read_csv(file_path)
        df["chrom"] = chrom
        model_to_result_df_list[model_name].append(df)

    model_to_result_df = {model_name: pd.concat(df_list) for model_name, df_list in model_to_result_df_list.items()}
    clinvar_df = pd.read_csv(config["data"]["processed_clinvar"])
    model_to_merged_df = {model_name: pd.merge(clinvar_df, df, on="id", how="inner") for model_name, df in model_to_result_df.items()}
    
    df_list = []

    cols = ["all", "noncoding", "intronic", "synonymous", "missense", "5_prime_utr", "3_prime_utr", "splice_donor",
            "splice_acceptor", "start_codon", "stop_lost", "upstream_of_transcript", "downstream_of_transcript"]
        
    for model_name, merged_df in model_to_merged_df.items():
        for col in cols:
            mask = (merged_df["num_stars"] > 0) & (
                merged_df["benign"] | merged_df["likely_benign"] |
                merged_df["pathogenic"] | merged_df["likely_pathogenic"]
            )

            if col != "all":
                mask &= (merged_df[col] == 1)

            masked_merged_df = merged_df.loc[mask]
            labels = masked_merged_df["benign"] | masked_merged_df["likely_benign"]
            fpr_list, tpr_list, thresholds = roc_curve(labels, masked_merged_df["log_likelihood_ratio"])

            df = pd.DataFrame({
                "model": model_name,
                "category": col,
                "fpr": fpr_list,
                "tpr": tpr_list,
                "threshold": thresholds
            })
            df_list.append(df)

    pd.concat(df_list, ignore_index=True).to_csv(config["data"]["clinvar_eval"], index=False)


def _process_omim_results(model_name, result_df, num_bins: int):
    with open(config["data"]["omim"], "r") as f:
        omim_variants = [line.strip() for line in f]

    positive_set_df = pd.DataFrame({"id": omim_variants})

    file_paths = glob.glob(config["data"]["gnomad_dir"] + "/*.csv.gz")
    maf_bins = np.logspace(-4, np.log10(0.5), num=num_bins + 1)[:-1]
    local_df_list = []

    for file_path in file_paths:
        gnomad_df = pd.read_csv(file_path, compression="gzip")
        gnomad_df = gnomad_df[["id", "ac", "an"]]
        gnomad_df["af"] = gnomad_df["ac"] / gnomad_df["an"]
        mask = gnomad_df["af"] > 0.5
        gnomad_df["maf"] = gnomad_df["af"]
        gnomad_df.loc[mask, "maf"] = 1 - gnomad_df.loc[mask, "af"]
        mask = gnomad_df["maf"] > maf_bins[0]
        negative_set_df = gnomad_df.loc[mask, ["id", "af"]].copy()

        for threshold in maf_bins:
            col = f"label_{np.log10(threshold):.4}"
            negative_set_df[col] = None
            positive_set_df[col] = 1
            mask = gnomad_df["maf"] > threshold
            negative_set_df.loc[mask, col] = 0

        variant_df = pd.concat([positive_set_df, negative_set_df], ignore_index=True)
        merged_df = pd.merge(
            variant_df,
            result_df,
            on="id",
            how="inner"
        )
        mask = merged_df["af"] > 0.5
        merged_df.loc[mask, "log_likelihood_ratio"] *= -1
        merged_df["model"] = model_name
        merged_df.drop(["af"], axis=1, inplace=True)
        local_df_list.append(merged_df)
    
    return pd.concat(local_df_list, ignore_index=True)


@task
def merge_omim_results(context, num_bins: int = 5):
    file_paths = glob.glob(f"{config['data']['result_dir_path']}/*.csv")
    file_paths = [x for x in file_paths if "full" not in x]
    model_to_result_df_list = defaultdict(list)

    for file_path in file_paths:
        base_name = file_path.split("/")[-1].rstrip(".csv")
        *model_name_parts, chrom = base_name.split("_")
        model_name = "_".join(model_name_parts)

        df = pd.read_csv(file_path)
        df["chrom"] = chrom
        model_to_result_df_list[model_name].append(df)

    model_to_result_df = {model_name: pd.concat(df_list) for model_name, df_list in model_to_result_df_list.items()}

    args_ = [(model_name, df, num_bins) for model_name, df in model_to_result_df.items()]

    with Pool(processes=4) as pool:
        df_list = pool.starmap(_process_omim_results, args_)

    label_df = pd.concat(df_list, ignore_index=True)
    cols = [col for col in label_df.columns if col.startswith("label")]

    results = defaultdict(list)

    for col in cols:
        threshold = float(col.split("_")[-1])
        mask = label_df[col].notna()

        for model, group_df in label_df.loc[mask].groupby("model"):
            labels = group_df[col].astype(int)
            preds = -group_df["log_likelihood_ratio"]
            results["model"].append(model)
            results["threshold"].append(threshold)
            results["auroc"].append(roc_auc_score(labels, preds))
            results["auprc"].append(average_precision_score(labels, preds))
    
    pd.DataFrame(results).to_csv(config["data"]["omim_eval"], index=False)


@task
def merge_dms_results(context):
    file_paths = glob.glob(f"{config['data']['result_dir_path']}/*.csv")
    file_paths = [x for x in file_paths if "full" not in x]
    model_to_result_df_list = defaultdict(list)

    for file_path in file_paths:
        base_name = file_path.split("/")[-1].rstrip(".csv")
        *model_name_parts, chrom = base_name.split("_")
        model_name = "_".join(model_name_parts)

        df = pd.read_csv(file_path)
        df["chrom"] = chrom
        model_to_result_df_list[model_name].append(df)

    model_to_result_df = {model_name: pd.concat(df_list) for model_name, df_list in model_to_result_df_list.items()}
    dms_df = pd.read_csv(config["data"]["processed_dms"])
    model_to_merged_df = {model_name: pd.merge(dms_df, df, on="id", how="inner") for model_name, df in model_to_result_df.items()}
    
    data = defaultdict(list)

    for model_name, merged_df in model_to_merged_df.items():
        for study, group_df in merged_df.groupby("study"):
            data["model"].append(model_name)
            data["study"].append(study)
            data["spearman_corr"].append(group_df["log_likelihood_ratio"].corr(group_df["label"], method="spearman"))

    pd.DataFrame(data).to_csv(config["data"]["dms_eval"], index=False)